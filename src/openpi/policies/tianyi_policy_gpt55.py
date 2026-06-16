import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


TIENYI_ACTION_DIM = 16
TIENYI_STATE_DIM = 16


def make_tienyi_example() -> dict:
    """Creates a random input example for the Tienyi dual-arm dex-hand/gripper policy."""

    return {
        "observation.state": np.random.rand(TIENYI_STATE_DIM).astype(np.float32),
        "observation.images.cam_head": np.random.randint(
            256, size=(480, 640, 3), dtype=np.uint8
        ),
        "observation.images.cam_left_wrist": np.random.randint(
            256, size=(480, 640, 3), dtype=np.uint8
        ),
        "observation.images.cam_right_wrist": np.random.randint(
            256, size=(480, 640, 3), dtype=np.uint8
        ),
        "prompt": "fold clothes",
    }


def _parse_image(image) -> np.ndarray:
    """Convert image to uint8 HWC.

    LeRobot video/image tensors are often loaded as:
    - float32 in CHW format, range [0, 1]
    - uint8 in CHW format
    - uint8 in HWC format

    OpenPI policy input expects uint8 HWC.
    """

    image = np.asarray(image)

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (255 * image).astype(np.uint8)

    # CHW -> HWC
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")

    return image


def _get_first_existing(data: dict, keys: tuple[str, ...]):
    """Utility for supporting both slash-style and dot-style keys."""

    for key in keys:
        if key in data:
            return data[key]

    raise KeyError(f"None of the keys exist in input data: {keys}")


@dataclasses.dataclass(frozen=True)
class TienyiInputs(transforms.DataTransformFn):
    """Convert your Tienyi dual-arm dataset sample to OpenPI model input format.

    Expected raw LeRobot keys from your dataset:
    - observation.state
    - action
    - observation.images.cam_head
    - observation.images.cam_left_wrist
    - observation.images.cam_right_wrist

    Output keys expected by OpenPI model:
    - state
    - image/base_0_rgb
    - image/left_wrist_0_rgb
    - image/right_wrist_0_rgb
    - image_mask/*
    - actions
    - prompt
    """

    model_type: _model.ModelType
    default_prompt: str = "fold clothes"

    def __call__(self, data: dict) -> dict:
        # Support both original LeRobot dot keys and possible repacked slash keys.
        state = _get_first_existing(
            data,
            (
                "observation.state",
                "observation/state",
                "state",
            ),
        )

        base_image = _parse_image(
            _get_first_existing(
                data,
                (
                    "observation.images.cam_head",
                    "observation/image",
                    "image",
                    "cam_head",
                ),
            )
        )

        left_wrist_image = _parse_image(
            _get_first_existing(
                data,
                (
                    "observation.images.cam_left_wrist",
                    "observation/left_wrist_image",
                    "left_wrist_image",
                    "cam_left_wrist",
                ),
            )
        )

        right_wrist_image = _parse_image(
            _get_first_existing(
                data,
                (
                    "observation.images.cam_right_wrist",
                    "observation/right_wrist_image",
                    "right_wrist_image",
                    "cam_right_wrist",
                ),
            )
        )

        inputs = {
            # Do not rename this key.
            "state": np.asarray(state, dtype=np.float32),

            # These image names are OpenPI/pi0 convention.
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },

            # All three cameras exist in your dataset, so all masks are True.
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Training only.
        if "action" in data:
            inputs["actions"] = np.asarray(data["action"], dtype=np.float32)
        elif "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        # Prompt / language instruction.
        # Your dataset has total_tasks = 1, so fixed prompt is okay.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif "task" in data:
            inputs["prompt"] = data["task"]
        else:
            inputs["prompt"] = self.default_prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class TienyiOutputs(transforms.DataTransformFn):
    """Convert OpenPI model output back to your robot action format.

    Your robot expects 16 absolute action dimensions:
    left_arm_0 ... left_arm_6, left_gripper,
    right_arm_0 ... right_arm_6, right_gripper.
    """

    action_dim: int = TIENYI_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        # OpenPI model may output padded action dimension internally.
        # Only keep the first 16 dims for your robot.
        actions = np.asarray(data["actions"], dtype=np.float32)
        data["actions"] = actions[..., : self.action_dim]
        return data
