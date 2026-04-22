import pandas as pd
import numpy as np
from pathlib import Path
import yaml

POSE_COLS = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
POSITION_COLS = ['x', 'y', 'z']

N_HISTORY = 1


def pair_images_and_poses(image_dir, poses_csv, dataset_csv):
    """Pairs images to the corresponding pose at the nearest timestamp"""
    
    poses = pd.read_csv(poses_csv)
    images = sorted(Path(image_dir).glob('*.pt'))

    img_timestamps = [float(p.stem) / 1e9 for p in images]

    pairs = []
    for img_path, img_timestamp in zip(images, img_timestamps):
        idx = np.argmin(np.abs(poses['timestamp'] - img_timestamp))
        nearest_pose = poses.iloc[idx]
        time_diff = abs(nearest_pose['timestamp'] - img_timestamp)
        if time_diff < 0.05:
            pairs.append({'image': img_path.name, **nearest_pose})

    pd.DataFrame(pairs).to_csv(dataset_csv, index=False)
    print(f'{len(pairs)} matched pairs out of {len(images)} images')


def add_gripper_column(csv_path: str, yaml_path: str,
                       lookahead_seconds: float = 2.0):
    """Adds gripper state to the dataset using the recorded trajectory."""
    df = pd.read_csv(csv_path)

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    recorded_at = df.iloc[0]['timestamp']

    events = []
    for event in sorted(meta['gripper_events'], key=lambda e: e['time']):
        events.append({
            'time':   recorded_at + event['time'] - lookahead_seconds,
            'action': event['action'],
        })

    ACTION_TO_STATE = {'close': 1.0, 'open': 0.0}
    gripper_state = 0.0
    event_idx = 0
    states = []

    for ts in df['timestamp']:
        while event_idx < len(events) and events[event_idx]['time'] <= ts:
            gripper_state = ACTION_TO_STATE[events[event_idx]['action']]
            event_idx += 1
        states.append(gripper_state)

    df['gripper'] = states
    df.to_csv(csv_path, index=False)
    print(f'Saved {len(df)} rows to {csv_path}')
    print(f'Gripper closed for {int(sum(states))} rows '
          f'({100 * sum(states) / len(states):.1f}%)')


def find_target_index(df: pd.DataFrame, i: int,
                       delta_t: float = 0.5,
                       position_threshold=0.001,
                       max_lookahead=50) -> int:
    """
    Target = state ~delta_t seconds in the future.
    If that state has no meaningful change, skip forward
    until position or gripper changes.
    """

    current_time = df.iloc[i]['timestamp']
    current_pos = df.iloc[i][POSITION_COLS].to_numpy(dtype=np.float32)
    current_gripper = df.iloc[i]['gripper']

    target_time = current_time + delta_t

    # find first index >= target_time
    target = i
    while target < len(df) and df.iloc[target]['timestamp'] < target_time:
        target += 1

    if target >= len(df):
        return -1

    # check if that state is meaningful
    def is_meaningful(j):
        pos = df.iloc[j][POSITION_COLS].to_numpy(dtype=np.float32)
        gripper = df.iloc[j]['gripper']

        return (
            gripper != current_gripper or
            np.linalg.norm(pos - current_pos) > position_threshold
        )

    # if not meaningful, skip forward
    steps = 0
    while target < len(df) - 1 and steps < max_lookahead:
        if is_meaningful(target):
            break
        target += 1
        steps += 1

    # if we failed to find anything meaningful, invalidate
    if target >= len(df):
        return -1

    return target


def find_history_indices(df: pd.DataFrame, i: int,
                         n_history: int, spacing_seconds: float = 0.5) -> list[int]:
    """Finds the indexes in the dataset that represent the history of the current state, which is the 
    current state plus 4 previous states spanning the last 2 seconds"""
    current_time = df.iloc[i]['timestamp']
    history = []

    for step in range(0, n_history):
        target_time = current_time - step * spacing_seconds
        # Find the row with timestamp closest to target_time
        idx = (df['timestamp'] - target_time).abs().argmin()
        # Don't go past the start of the demo
        idx = max(0, min(idx, i))
        history.append(idx)

    history.reverse()  # chronological order, oldest first
    return history


def precompute_samples(df: pd.DataFrame, n_history: int,
                       spacing_seconds: float = 0.5) -> list[dict]:
    """Creates the dataset of current index in the dataframe, the target index, and indexes of the history of the
    current state"""

    targets = [find_target_index(df, i) for i in range(len(df))]

    samples = []
    for i, target in enumerate(targets):
        if target == -1:
            continue
        history = find_history_indices(df, i, n_history, spacing_seconds)
        samples.append({
            'current_idx': i,
            'target_idx':  target,
            **{f'history_{j}': history[j] for j in range(n_history)},
        })

    return samples


if __name__ == '__main__':
    for i in range(1, 20):
        demo_num    = i
        poses_csv   = f'/home/joeya/dataset/demo{demo_num}/poses.csv'
        dataset_csv = f'/home/joeya/dataset/demo{demo_num}/dataset.csv'
        image_dir   = f'/home/joeya/dataset/demo{demo_num}'
        yaml_path   = (f'/mnt/c/Users/joeya/Imitation Learning Demos/'
                    f'trajectory_recordings/demo{demo_num}.json')
        targets_csv = f'/home/joeya/dataset/demo{demo_num}/targets.csv'

        pair_images_and_poses(image_dir, poses_csv, dataset_csv)
        add_gripper_column(csv_path=dataset_csv, yaml_path=yaml_path,
                        lookahead_seconds=2.0)

        df = pd.read_csv(dataset_csv)
        samples = precompute_samples(df, n_history=N_HISTORY)
        samples_df = pd.DataFrame([{
            'current_idx': s['current_idx'],
            'target_idx':  s['target_idx'],
            **{f'history_{j}': s[f'history_{j}'] for j in range(N_HISTORY)},
        } for s in samples])
        samples_df.to_csv(targets_csv, index=False)
        print(f'Saved final dataset with {len(samples_df)} rows')