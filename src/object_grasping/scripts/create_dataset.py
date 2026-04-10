import pandas as pd
import numpy as np
from pathlib import Path
import os

poses = pd.read_csv('poses.csv')
images = sorted(Path('.').glob('*.jpg'))

img_timestamps = [float(p.stem) / 1e9 for p in images]

pairs = []
for img_path, img_timestamp in zip(images, img_timestamps):
    idx = np.argmin(np.abs(poses['timestamp'] - img_timestamp))
    nearest_pose = poses.iloc[idx]
    time_diff = abs(nearest_pose['timestamp'] - img_timestamp)
    if time_diff < 0.05:
        pairs.append({'image': img_path.name, **nearest_pose})

pd.DataFrame(pairs).to_csv('dataset.csv', index=False)
print(f'{len(pairs)} matched pairs out of {len(images)} images')