from pathlib import Path

import numpy as np
import pandas as pd

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

        target = i
        while target < len(df) and df.iloc[target]["timestamp"] < target_time:
            target += 1

        if target >= len(df):
            targets.append(len(df) - 1)
            continue

        def is_meaningful(j):
            pos = df.iloc[j][POSITION_COLS].to_numpy(dtype=np.float32)
            gripper = df.iloc[j]["gripper"]
            return (
                gripper != current_gripper
                or np.linalg.norm(pos - current_pos) > position_threshold
            )

        steps = 0
        while target < len(df) - 1 and steps < max_lookahead:
            if is_meaningful(target):
                break
            target += 1
            steps += 1

        if target >= len(df):
            targets.append(len(df) - 1)
            continue

        targets.append(target)

    df["target"] = targets
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    # for i in range(1, 20):
        demo_num = 1
        poses_csv = f"/home/joeya/pick_and_place_dataset/demo{demo_num}/poses_and_gripper.csv"
        dataset_csv = f"/home/joeya/pick_and_place_dataset/demo{demo_num}/dataset.csv"
        image_dir = f"/home/joeya/pick_and_place_dataset/demo{demo_num}"

        pair_images_and_poses(image_dir, poses_csv, dataset_csv)
        assign_targets(csv_path=dataset_csv)
        print("Saved final dataset")