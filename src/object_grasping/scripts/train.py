import torch
from model.model import PolicyNetwork, PolicyNetworkLoss
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path

POSE_COLS = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
POSE_GRIPPER_COLS = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper']



def quaternion_delta(q_current: np.ndarray, q_next: np.ndarray) -> np.ndarray:
    r_current = Rotation.from_quat(q_current)
    r_next = Rotation.from_quat(q_next)
    r_delta = r_next * r_current.inv()
    delta_quat = r_delta.as_quat()
    return delta_quat


def find_target_index(df: pd.DataFrame, i: int, pose_threshold = 0.001) -> int:
    current_pose = df.iloc[i][POSE_COLS].to_numpy(dtype=np.float32)
    current_gripper = df.iloc[i]['gripper']

    target = i+1
    while target < len(df):
        target_pose = df.iloc[target][POSE_COLS].to_numpy(dtype=np.float32)
        target_gripper = df.iloc[target]['gripper']

        delta_pose = np.linalg.norm(current_pose - target_pose)
        gripper_changed = target_gripper != current_gripper

        if gripper_changed or delta_pose > pose_threshold:
            break

        target += 1

    if target == len(df):
        return -1
    else:
        return target
    

def precompute_index_chain(df: pd.DataFrame, n_history: int) -> list[dict]:
    
    # Pass 1: compute target for every row
    targets = [find_target_index(df, i) for i in range(len(df))]

    # Pass 2: build the ordered list of target indices (deduplicated, in order)
    # This is the "target chain" we walk back through for history
    target_chain = []
    seen = set()
    for t in targets:
        if t != -1 and t not in seen:
            target_chain.append(t)
            seen.add(t)
    
    print(f"Found {len(target_chain)} target indexes")

    # Build a lookup: row index -> position in target_chain
    chain_position = {target: idx for idx, target in enumerate(target_chain)}

    # Pass 3: for each valid (current, target) pair, walk back through
    # target_chain to find the previous n_history targets as history
    samples = []
    for i, target in enumerate(targets):
        if target == -1:
            continue

        # Find where this target sits in the chain
        pos = chain_position[target]

        # Need n_history previous entries in the chain before this target
        if pos < n_history:
            continue  # not enough history yet

        history_indices = target_chain[pos - n_history:pos]

        samples.append({
            'current_idx': i,
            'target_idx':  target,
            'history':     history_indices,  # list of n_history row indices
        })

    return samples


class ImitationLearningDataset(Dataset):
    def __init__(self, csv_path, image_dir, n_history):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.n_history = n_history

        self.preprocess = torchvision.models.ResNet18_Weights.DEFAULT.transforms()
        self.samples = precompute_index_chain(self.df, n_history)
        print(f'{len(self.samples)} valid samples from {len(self.df)} rows')

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        current_idx = sample['current_idx']
        target_idx = sample['target_idx']
        history = sample['history']

        image = Image.open(self.image_dir / self.df.iloc[current_idx]['image'])
        image = self.preprocess(image)

        history_list = self.df.iloc[history][POSE_GRIPPER_COLS].to_numpy(dtype=np.float32)
        history_vector = torch.tensor(history_list, dtype=torch.float32)

        current_pose = self.df.iloc[current_idx][POSE_COLS].to_numpy(dtype=np.float32)
        target_pose = self.df.iloc[target_idx][POSE_COLS].to_numpy(dtype=np.float32)
        
        delta_position = target_pose[:3] - current_pose[:3]
        delta_orientation = quaternion_delta(current_pose[3:], target_pose[3:])

        gripper_state = float(self.df.iloc[target_idx]['gripper'])

        return {
            'image': image,
            'history': history_vector,
            'delta_position': torch.tensor(delta_position, dtype=torch.float32),
            'delta_orientation': torch.tensor(delta_orientation, dtype=torch.float32),
            'gripper_state': torch.tensor([gripper_state], dtype=torch.float32),
        }
    



def train(
        csv_path,
        image_dir,
        n_history = 5,
        hidden_dim = 512,
        epochs = 50,
        batch_size = 32,
        lr = 1e-3
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}')

    dataset = ImitationLearningDataset(csv_path, image_dir, n_history)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = PolicyNetwork(n_history=n_history, hidden_dim=hidden_dim).to(device)
    loss_fn = PolicyNetworkLoss().to(device)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=lr)


    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            image = batch['image'].to(device)
            history = batch['history'].to(device)
            target = {
                'delta_position': batch['delta_position'].to(device),
                'delta_orientation': batch['delta_orientation'].to(device),
                'gripper_state': batch['gripper_state'].to(device)
            }

            optimizer.zero_grad()
            preds = model(image, history)
            losses = loss_fn(preds, target)
            losses['total'].backward()
            optimizer.step()

            train_losses.append(losses['total'].item())
        
        print(f'Epoch {epoch}: Loss is {np.mean(train_losses)}')


if __name__ == '__main__':
    train(
        csv_path='/home/joeya/dataset/demo1/dataset.csv',
        image_dir='/home/joeya/dataset/demo1'
    )