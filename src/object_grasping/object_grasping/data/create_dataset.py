from pathlib import Path

import numpy as np
import pandas as pd
import yaml

POSITION_COLS = ["x", "y", "z"]


def pair_images_and_poses(image_dir, poses_csv, dataset_csv):
    """Pairs images in the given directory to the corresponding pose at the nearest timestamp
    and saves to csv."""

    poses = pd.read_csv(poses_csv)
    images = sorted(Path(image_dir).glob("*.pt"))

    img_timestamps = [float(p.stem) / 1e9 for p in images]

    pairs = []
    for img_path, img_timestamp in zip(images, img_timestamps):
        idx = np.argmin(np.abs(poses["timestamp"] - img_timestamp))
        nearest_pose = poses.iloc[idx]
        time_diff = abs(nearest_pose["timestamp"] - img_timestamp)
        if time_diff < 0.05:
            pairs.append({"image": img_path.name, **nearest_pose})

    pd.DataFrame(pairs).to_csv(dataset_csv, index=False)
    print(f"{len(pairs)} matched pairs out of {len(images)} images")


def add_gripper_column(csv_path: str, yaml_path: str):
    """Adds gripper state to the dataset csv using the recorded trajectory json file."""

    df = pd.read_csv(csv_path)

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    recorded_at = df.iloc[0]["timestamp"]

    events = []
    for event in sorted(meta["gripper_events"], key=lambda e: e["time"]):
        events.append(
            {
                "time": recorded_at + event["time"],
                "action": event["action"],
            }
        )

    ACTION_TO_STATE = {"close": 1.0, "open": 0.0}
    gripper_state = 0.0
    event_idx = 0
    states = []

    for ts in df["timestamp"]:
        while event_idx < len(events) and events[event_idx]["time"] <= ts:
            gripper_state = ACTION_TO_STATE[events[event_idx]["action"]]
            event_idx += 1
        states.append(gripper_state)

    df["gripper"] = states
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows to {csv_path}")
    print(
        f"Gripper closed for {int(sum(states))} rows "
        f"({100 * sum(states) / len(states):.1f}%)"
    )


def assign_targets(
    csv_path,
    delta_t: float = 0.5,
    position_threshold=0.001,
    max_lookahead=50,
) -> None:
    """
    Adds a 'target' column to the CSV where each row's target is the index
    of the state ~delta_t seconds in the future. If that state has no
    meaningful change, skips forward until position or gripper changes.
    Rows with no valid target are assigned -1.
    """

    df = pd.read_csv(csv_path)
    targets = []

    for i in range(len(df)):
        current_time = df.iloc[i]["timestamp"]
        current_pos = df.iloc[i][POSITION_COLS].to_numpy(dtype=np.float32)
        current_gripper = df.iloc[i]["gripper"]

        target_time = current_time + delta_t

        # find first index >= target_time
        target = i
        while target < len(df) and df.iloc[target]["timestamp"] < target_time:
            target += 1

        if target >= len(df):
            targets.append(len(df) - 1)
            continue

        # check if that state is meaningful
        def is_meaningful(j):
            pos = df.iloc[j][POSITION_COLS].to_numpy(dtype=np.float32)
            gripper = df.iloc[j]["gripper"]

            return (
                gripper != current_gripper
                or np.linalg.norm(pos - current_pos) > position_threshold
            )

        # if not meaningful, skip forward
        steps = 0
        while target < len(df) - 1 and steps < max_lookahead:
            if is_meaningful(target):
                break
            target += 1
            steps += 1

        # if we failed to find anything meaningful, clamp to final index
        if target >= len(df):
            targets.append(len(df) - 1)
            continue

        targets.append(target)

    df["target"] = targets
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    for i in range(1, 20):
        demo_num = i
        poses_csv = f"/home/joeya/dataset/demo{demo_num}/poses.csv"
        dataset_csv = f"/home/joeya/dataset/demo{demo_num}/dataset.csv"
        image_dir = f"/home/joeya/dataset/demo{demo_num}"
        yaml_path = (
            f"/mnt/c/Users/joeya/Imitation Learning Demos/"
            f"trajectory_recordings/demo{demo_num}.json"
        )

        pair_images_and_poses(image_dir, poses_csv, dataset_csv)
        add_gripper_column(csv_path=dataset_csv, yaml_path=yaml_path)
        assign_targets(csv_path=dataset_csv)
        print("Saved final dataset")
