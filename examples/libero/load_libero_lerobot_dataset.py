from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset_path = "/home/ps/.cache/huggingface/lerobot/physical-intelligence/libero"

dataset = LeRobotDataset(dataset_path)

tasks = set()
episode_indices = set()

# (['image', 'wrist_image', 'state', 'actions', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index', 'task'])
for sample in dataset:
    # tasks.add(sample["task"])
    # episode_indices.add(sample["episode_index"])
    # print(sample["task"], sample["episode_index"])

    print(
        "image shape:", sample["image"].shape,
        "wrist image shape:", sample["wrist_image"].shape,
        "state shape:", sample["state"].shape,
        "actions shape:", sample["actions"].shape,
        "timestamp:", sample["timestamp"],
        "frame index:", sample["frame_index"],
        "episode index:", sample["episode_index"],
        "index:", sample["index"],
        "task index:", sample["task_index"],
        "task:", sample["task"],
    )

print("num frames:", len(dataset))
print("num episodes:", len(episode_indices))
print("unique tasks:", len(tasks))
