import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import json


def add_gripper_column(csv_path: str, json_path: str,
                       output_path: str = None,
                       lookahead_seconds: float = 2.0) -> pd.DataFrame:
    """
    Add a gripper column to the dataset CSV.

    Gripper events are shifted back by lookahead_seconds so the network
    learns to anticipate the gripper action rather than react to it.
    i.e. if the gripper closes at t=10.0, rows from t=8.0 onwards will
    have gripper=1 (closed).

    Args:
        csv_path:          Path to the dataset CSV.
        json_path:         Path to the demo JSON with gripper_events.
        output_path:       Where to save the result. Defaults to csv_path.
        lookahead_seconds: How far back to shift gripper events (default 2s).

    Returns:
        DataFrame with gripper column added.
    """
    df = pd.read_csv(csv_path)

    with open(json_path) as f:
        meta = yaml.safe_load(f)

    recorded_at = df.iloc[0]['timestamp']

    # Convert relative event times to absolute timestamps,
    # then shift back by lookahead_seconds
    events = []
    for event in sorted(meta['gripper_events'], key=lambda e: e['time']):
        absolute_time  = recorded_at + event['time']
        effective_time = absolute_time - lookahead_seconds
        events.append({
            'time':   effective_time,
            'action': event['action'],
        })

    # Reconstruct gripper state at each row timestamp.
    # Gripper starts open (0). Each close event sets state=1,
    # each open event sets state=0.
    ACTION_TO_STATE = {'close': 1.0, 'open': 0.0}

    gripper_state = 0.0
    event_idx     = 0
    states        = []

    for ts in df['timestamp']:
        # Apply all events whose effective time has passed
        while event_idx < len(events) and events[event_idx]['time'] <= ts:
            gripper_state = ACTION_TO_STATE[events[event_idx]['action']]
            event_idx += 1
        states.append(gripper_state)

    df['gripper'] = states

    output_path = output_path or csv_path
    df.to_csv(output_path, index=False)
    print(f'Saved {len(df)} rows to {output_path}')
    print(f'Gripper closed for {int(sum(states))} rows '
          f'({100 * sum(states) / len(states):.1f}%)')
    return df


if __name__ == '__main__':
    df = add_gripper_column(
        csv_path='/home/joeya/dataset/demo1/dataset.csv',
        json_path='demo1.json',
        lookahead_seconds=2.0,
    )
    print(df[['timestamp', 'gripper']].head(20))