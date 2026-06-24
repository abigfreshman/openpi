from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("/media/leon/data/clothes_washing/part1")

print(len(ds))

# from pathlib import Path
# import pandas as pd

# root = Path("/media/leon/data/clothes_washing/part1")

# # 常见路径
# possible_files = [
#     root / "data" / "chunk-000" / "episode_000207.parquet",
#     root / "data" / "chunk-000" / "episode_207.parquet",
# ]

# for f in possible_files:
#     print(f, f.exists())

# episode_file = root / "data" / "chunk-000" / "episode_000207.parquet"

# df = pd.read_parquet(episode_file)

# print(df.columns)
# print(df.shape)

# # 查看 timestamp
# print(df[["episode_index", "frame_index", "timestamp"]].head(20))
# print(df[["episode_index", "frame_index", "timestamp"]].tail(20))

# # 找 timestamp 倒退的位置
# ts = df["timestamp"].to_numpy()
# diff = ts[1:] - ts[:-1]

# bad_indices = (diff < 0).nonzero()[0]

# print("bad_indices:", bad_indices)

# for i in bad_indices[:20]:
#     print("=" * 80)
#     print("bad at row:", i, "->", i + 1)
#     print("timestamps:", ts[i], "->", ts[i + 1], "diff:", diff[i])
#     print(df.iloc[max(0, i - 5): i + 6][["episode_index", "frame_index", "timestamp"]])

# from pathlib import Path
# import pandas as pd

# root = Path("/media/leon/data/clothes_washing/part1")

# for ep in [206, 207, 208, 209]:
#     pf = root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
#     print("\n" + "=" * 80)
#     print("episode", ep, pf, pf.exists())

#     if not pf.exists():
#         continue

#     df = pd.read_parquet(pf, columns=["episode_index", "frame_index", "timestamp", "index"])

#     print("shape:", df.shape)
#     print("episode_index unique:", df["episode_index"].unique())
#     print("frame_index min/max:", df["frame_index"].min(), df["frame_index"].max())
#     print("timestamp min/max:", df["timestamp"].min(), df["timestamp"].max())
#     print("index min/max:", df["index"].min(), df["index"].max())

#     print("head:")
#     print(df.head(3))

#     print("tail:")
#     print(df.tail(3))



# from pathlib import Path
# import json

# root = Path("/media/leon/data/clothes_washing/part1")
# episodes_path = root / "meta" / "episodes.jsonl"

# with open(episodes_path, "r") as f:
#     episodes = [json.loads(line) for line in f]

# print("num lines:", len(episodes))

# for i in [205, 206, 207, 208, 209]:
#     print("\n" + "=" * 80)
#     print("line index:", i)

#     if i >= len(episodes):
#         print("not exists in episodes.jsonl")
#         continue

#     print(json.dumps(episodes[i], indent=2, ensure_ascii=False))
