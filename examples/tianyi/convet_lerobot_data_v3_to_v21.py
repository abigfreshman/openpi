"""Convert the Tienyi dual-arm / dex-hand "cloth_folding" data (LeRobot **v3.0** layout)
into a single openpi-compatible LeRobot **v2.1** dataset.

Why this script exists
-----------------------
openpi pins LeRobot at ``CODEBASE_VERSION == "v2.1"`` (see ``lerobot.common.datasets``),
but the data under ``data/cloth_folding`` is stored in the newer **v3.0** layout, where:

* tabular features live in ``data/chunk-XXX/file-YYY.parquet`` (many episodes per file),
* videos are *concatenated* per camera into ``videos/<key>/chunk-XXX/file-YYY.mp4``,
* per-episode ranges (row range + per-camera file index + timestamp window) live in
  ``meta/episodes/chunk-XXX/file-YYY.parquet``.

This script reads that raw v3.0 layout directly (pyarrow for tables, PyAV for video) and
re-emits a fresh v2.1 dataset that openpi can train on, with the standard openpi keys:

* ``observation.state``  -> proprio (puppet / follower), 16 dims
* ``action``             -> command (master / leader),   16 dims
* ``observation.images.cam_head``        (3rd-person / head camera)
* ``observation.images.cam_left_wrist``  (left camera)
* ``observation.images.cam_right_wrist`` (right camera)
* ``task``               -> language instruction (prompt)

State / action layout (16 = 7 + 1 + 7 + 1)
------------------------------------------
``[ left_arm(7), left_gripper(1), right_arm(7), right_gripper(1) ]``

* state  = ``puppet.arm_left_position_align`` (7), ``puppet.end_effector_left_position_align`` (1),
           ``puppet.arm_right_position_align`` (7), ``puppet.end_effector_right_position_align`` (1)
* action = the same fields from ``master.*`` (the teleoperator / leader command)

Usage
-----
uv run examples/tienyi/convert_tienyi_data_to_lerobot.py \
    --data-root data/cloth_folding \
    --repo-id tienyi/cloth_folding

# Quick smoke test on a single session, only a couple of episodes:
uv run examples/tienyi/convert_tienyi_data_to_lerobot.py \
    --data-root data/cloth_folding \
    --repo-id tienyi/cloth_folding_debug \
    --max-episodes-per-session 2

The resulting dataset is written to ``$HF_LEROBOT_HOME/<repo-id>``
(default ``~/.cache/huggingface/lerobot/<repo-id>``).
"""

import dataclasses
import logging
from pathlib import Path
import shutil

import av
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Source feature keys (v3.0 dataset) used to build state / action vectors.
# Order matters: [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)] == 16 dims.
PUPPET_KEYS = [
    ("puppet.arm_left_position_align.data", 7),
    ("puppet.end_effector_left_position_align.data", 1),
    ("puppet.arm_right_position_align.data", 7),
    ("puppet.end_effector_right_position_align.data", 1),
]
MASTER_KEYS = [
    ("master.arm_left_position_align.data", 7),
    ("master.end_effector_left_position_align.data", 1),
    ("master.arm_right_position_align.data", 7),
    ("master.end_effector_right_position_align.data", 1),
]
STATE_DIM = sum(d for _, d in PUPPET_KEYS)  # 16
ACTION_DIM = sum(d for _, d in MASTER_KEYS)  # 16

# Map source camera video keys -> openpi standard image keys.
CAMERA_MAP = {
    "camera_observations.color_images.camera_head": "observation.images.cam_head",
    "camera_observations.color_images.camera_left": "observation.images.cam_left_wrist",
    "camera_observations.color_images.camera_right": "observation.images.cam_right_wrist",
}


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    image_writer_processes: int = 8
    image_writer_threads: int = 4
    # Output video resolution (height, width). Native is 720x1280; openpi resizes to 224
    # internally so a smaller stored resolution keeps the dataset compact without hurting
    # training. Set to None to keep native resolution.
    resize_hw: tuple[int, int] | None = (480, 640)


def find_sessions(data_root: Path) -> list[Path]:
    """Find every v3.0 dataset root (the directory that contains ``meta/info.json``)."""
    sessions = sorted({p.parent.parent for p in data_root.glob("**/meta/info.json")})
    if not sessions:
        raise FileNotFoundError(f"No LeRobot datasets (meta/info.json) found under {data_root}")
    return sessions


def infer_prompt(session_dir: Path, default: str | None) -> str:
    """Derive a clean language instruction from the session path.

    The raw ``meta/tasks.parquet`` is unreliable in this dump (it sometimes contains an
    unrelated placeholder like ``ur2_place_eggplant_to_plate``), so we derive the prompt
    from the folder name, which encodes the real task (``fold_the_clothes`` / ``hang_clothes``).
    """
    if default is not None:
        return default
    name = str(session_dir).lower()
    if "hang" in name:
        return "hang the clothes"
    if "fold" in name:
        return "fold the clothes"
    return "manipulate the clothes"


def build_vectors(table_df, row_slice: slice, keys: list[tuple[str, int]]) -> np.ndarray:
    """Concatenate the given source columns into an (N, sum_dims) float32 array."""
    cols = []
    sub = table_df.iloc[row_slice]
    for key, dim in keys:
        arr = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in sub[key].to_numpy()])
        if arr.shape[1] != dim:
            raise ValueError(f"Column {key} has dim {arr.shape[1]}, expected {dim}")
        cols.append(arr)
    return np.concatenate(cols, axis=1).astype(np.float32)


def _resize(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    import cv2

    h, w = hw
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


class VideoFileReader:
    """Sequential frame reader over a single mp4 file.

    Multiple episodes are concatenated into one file; we keep one open decoder per camera and
    pull frames forward, opening a new file only when the per-episode file index changes.
    """

    def __init__(self, resize_hw: tuple[int, int] | None):
        self._resize_hw = resize_hw
        self._path: Path | None = None
        self._container = None
        self._iter = None

    def _open(self, path: Path):
        self.close()
        self._path = path
        self._container = av.open(str(path))
        self._iter = self._container.decode(self._container.streams.video[0])

    def read(self, path: Path, reset: bool, num_frames: int) -> np.ndarray:
        if reset or self._path != path:
            self._open(path)
        out = []
        for _ in range(num_frames):
            frame = next(self._iter)
            img = frame.to_ndarray(format="rgb24")
            if self._resize_hw is not None:
                img = _resize(img, self._resize_hw)
            out.append(img)
        return np.stack(out)

    def close(self):
        if self._container is not None:
            self._container.close()
        self._container = None
        self._iter = None
        self._path = None


def create_empty_dataset(repo_id: str, robot_type: str, cfg: DatasetConfig) -> LeRobotDataset:
    if cfg.resize_hw is not None:
        h, w = cfg.resize_hw
    else:
        h, w = 720, 1280

    motors = (
        [f"left_arm_{i}" for i in range(7)]
        + ["left_gripper"]
        + [f"right_arm_{i}" for i in range(7)]
        + ["right_gripper"]
    )
    features: dict = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": [motors]},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": [motors]},
    }
    for openpi_key in CAMERA_MAP.values():
        features[openpi_key] = {
            "dtype": "video",
            "shape": (3, h, w),
            "names": ["channels", "height", "width"],
        }

    out = HF_LEROBOT_HOME / repo_id
    if out.exists():
        shutil.rmtree(out)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=cfg.image_writer_processes,
        image_writer_threads=cfg.image_writer_threads,
    )


def process_session(
    dataset: LeRobotDataset,
    session_dir: Path,
    prompt: str,
    cfg: DatasetConfig,
    max_episodes: int | None,
) -> int:
    """Read one v3.0 session and append its episodes to `dataset`. Returns #episodes added."""
    meta_ep_files = sorted((session_dir / "meta" / "episodes").glob("**/*.parquet"))
    episodes_df = pq.read_table(meta_ep_files).to_pandas().sort_values("episode_index").reset_index(drop=True)

    # `dataset_from_index`/`dataset_to_index` are GLOBAL row indices across the whole session
    # (cumulative over every data parquet file). We load one data file at a time below, whose
    # rows are indexed locally from 0, so for sessions that span multiple data files the episodes
    # living in file-001+ would otherwise slice out of range (-> empty -> "need at least one array
    # to stack"). Convert to file-local indices by subtracting each data file's global start row.
    episodes_df["_file_start"] = episodes_df.groupby(["data/chunk_index", "data/file_index"])[
        "dataset_from_index"
    ].transform("min")

    # Cache loaded data parquet tables by (chunk, file) index.
    data_cache: dict[tuple[int, int], object] = {}

    def load_data_table(chunk_idx: int, file_idx: int):
        key = (chunk_idx, file_idx)
        if key not in data_cache:
            p = session_dir / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
            data_cache[key] = pq.read_table(p).to_pandas()
        return data_cache[key]

    readers = {src: VideoFileReader(cfg.resize_hw) for src in CAMERA_MAP}
    # Track the current video file index per camera to know when to reset the decoder.
    cur_file = {src: None for src in CAMERA_MAP}

    n_episodes = len(episodes_df) if max_episodes is None else min(max_episodes, len(episodes_df))
    added = 0
    for ep_pos in tqdm.tqdm(range(n_episodes), desc=session_dir.name):
        ep = episodes_df.iloc[ep_pos]
        length = int(ep["length"])
        d_chunk, d_file = int(ep["data/chunk_index"]), int(ep["data/file_index"])
        from_idx, to_idx = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        file_start = int(ep["_file_start"])  # global row where this data file begins

        df = load_data_table(d_chunk, d_file)
        # Convert global -> file-local row indices (see note where `_file_start` is computed).
        row_slice = slice(from_idx - file_start, to_idx - file_start)
        state = build_vectors(df, row_slice, PUPPET_KEYS)
        action = build_vectors(df, row_slice, MASTER_KEYS)
        assert state.shape == (length, STATE_DIM), (state.shape, length)
        assert action.shape == (length, ACTION_DIM), (action.shape, length)

        # Decode the matching frames for each camera.
        cam_frames: dict[str, np.ndarray] = {}
        for src, openpi_key in CAMERA_MAP.items():
            v_chunk = int(ep[f"videos/{src}/chunk_index"])
            v_file = int(ep[f"videos/{src}/file_index"])
            vpath = session_dir / "videos" / src / f"chunk-{v_chunk:03d}" / f"file-{v_file:03d}.mp4"
            reset = cur_file[src] != (v_chunk, v_file)
            cur_file[src] = (v_chunk, v_file)
            frames = readers[src].read(vpath, reset=reset, num_frames=length)
            if frames.shape[0] != length:
                raise ValueError(f"{vpath}: decoded {frames.shape[0]} frames, expected {length}")
            cam_frames[openpi_key] = frames

        for i in range(length):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": prompt,
            }
            for openpi_key, frames in cam_frames.items():
                frame[openpi_key] = frames[i]
            dataset.add_frame(frame)
        dataset.save_episode()
        added += 1

    for r in readers.values():
        r.close()
    return added


def main(
    data_root: str = "/media/DATA/put_clothes_in_washing_machine/lerobot_data_v3/tienyi_prod2_dualArm-gripper-3cameras_66/tienyi_prod2_dualArm-gripper-3cameras_66_Put_the_clothes_into_the_washing_machine_20260615_am",
    repo_id: str = "/media/leon/data/clothes_washing/success_data_0615am",
    robot_type: str = "tienyi_prod2_dualArm_dexHand",
    prompt: str | None = "Put the clothes in the washing machine",
    max_episodes_per_session: int | None = None,
    push_to_hub: bool = False,
    config: DatasetConfig = DatasetConfig(),
):
    data_root_path = Path(data_root)
    sessions = find_sessions(data_root_path)
    logger.info("Found %d sessions:", len(sessions))
    for s in sessions:
        logger.info("  - %s", s.relative_to(data_root_path) if data_root_path in s.parents else s)

    dataset = create_empty_dataset(repo_id, robot_type=robot_type, cfg=config)

    total = 0
    for session in sessions:
        ep_prompt = infer_prompt(session, prompt)
        logger.info("Processing %s  (prompt=%r)", session.name, ep_prompt)
        total += process_session(dataset, session, ep_prompt, config, max_episodes_per_session)

    logger.info("Done. Wrote %d episodes to %s", total, HF_LEROBOT_HOME / repo_id)

    if push_to_hub:
        dataset.push_to_hub(tags=["tienyi", "dual-arm", "dex-hand", "cloth"], private=True, push_videos=True)


if __name__ == "__main__":
    tyro.cli(main)