import torch
from torchvision.transforms import v2
from model.model import PolicyNetwork, PolicyNetworkLoss
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import yaml
import optuna
from optuna.visualization import plot_param_importances, plot_optimization_history
from optuna.trial import TrialState

POSE_COLS = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
POSE_GRIPPER_COLS = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper']
CONFIG_PATH='/home/joeya/Imitation-Learning/src/object_grasping/object_grasping/dataset_config.yaml'
N_HISTORY = 1
EPOCHS = 25

dataset = None


def quaternion_delta(q_current: np.ndarray, q_next: np.ndarray) -> np.ndarray:
    r_current = Rotation.from_quat(q_current)
    r_next = Rotation.from_quat(q_next)
    r_delta = r_next * r_current.inv()
    delta_quat = r_delta.as_quat()

    # Enforce consistent sign
    if delta_quat[3] < 0:  # qw < 0
        delta_quat = -delta_quat
    
    return delta_quat


class SingleDemoDataset(Dataset):
    def __init__(self, dataset_csv_path, targets_csv_path, image_dir, n_history):
        self.dataset_df = pd.read_csv(dataset_csv_path)
        self.targets_df = pd.read_csv(targets_csv_path)
        self.image_dir = Path(image_dir)
        self.n_history = n_history

        ## Set positive convention for quaternions
        qw_negative = self.dataset_df["qw"] < 0
        self.dataset_df.loc[qw_negative, ["qx", "qy", "qz", "qw"]] *= -1

        # print(self.dataset_df[0:10])

        # print(self.dataset_df[POSE_GRIPPER_COLS].agg(['min', 'max', 'mean']))

        self.augmentation = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomResizedCrop(
                size=(224, 224),
                scale=(0.9, 1.0),   # crop between 90% and 100% of the image area
                ratio=(0.9, 1.1),   # keep roughly square
            ),
            v2.RandomAffine(
                degrees=4,
                translate=(0.05, 0.05)
            ),
            v2.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.targets_df)
    
    def __getitem__(self, index):
        sample = self.targets_df.iloc[index]
        current_idx = sample['current_idx']
        target_idx = sample['target_idx']
        history  = sample[[f'history_{j}' for j in range(self.n_history)]].values

        image = torch.load(self.image_dir / Path(self.dataset_df.iloc[current_idx]['image']).with_suffix('.pt'))
        image = self.augmentation(image)

        history_list = self.dataset_df.iloc[history][POSE_GRIPPER_COLS].to_numpy(dtype=np.float32)
        history_vector = torch.tensor(history_list, dtype=torch.float32)

        current_pose = self.dataset_df.iloc[current_idx][POSE_COLS].to_numpy(dtype=np.float32)
        target_pose = self.dataset_df.iloc[target_idx][POSE_COLS].to_numpy(dtype=np.float32)
        
        delta_position = target_pose[:3] - current_pose[:3]
        delta_orientation = quaternion_delta(current_pose[3:], target_pose[3:])

        gripper_state = float(self.dataset_df.iloc[target_idx]['gripper'])

        return {
            'image': image,
            'history': history_vector,
            'delta_position': torch.tensor(delta_position, dtype=torch.float32) * 100,
            'delta_orientation': torch.tensor(delta_orientation, dtype=torch.float32),
            'gripper_state': torch.tensor([gripper_state], dtype=torch.float32),
        }
    

def load_dataset_from_yaml(config_path, n_history) -> ConcatDataset:
    """Expects YAML format:
        
        demos:
            - dataset_csv: path1
              targets_csv: targets_path1
              image_dir: image_path1
            - dataset_csv: path2
              targets_csv: targets_path2
              image_dir: image_path2
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    demos = config.get('demos', [])
    if not demos:
        raise ValueError(f'No demos found in {config_path}')

    print(f"Loading {len(demos)} demonstrations...")

    datasets = [
        SingleDemoDataset(
            dataset_csv_path=demo['dataset_csv'],
            targets_csv_path=demo['targets_csv'],
            image_dir=demo['image_dir'],
            n_history=n_history
        )
        for demo in demos
    ]

    combined = ConcatDataset(datasets)
    print(f"Total samples across all demos: {len(combined)}")
    return combined



def train_and_save(
        hidden_dim = 256,
        batch_size = 32,
        lr = 0.0015
):
    dataset = load_dataset_from_yaml(CONFIG_PATH, N_HISTORY)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}')

    training_set, validation_set = random_split(dataset, [0.85, 0.15])
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

    model = PolicyNetwork(n_history=N_HISTORY, hidden_dim=hidden_dim).to(device)
    loss_fn = PolicyNetworkLoss().to(device)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=lr)


    for epoch in range(EPOCHS):
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
    
    ## Final training loss
    model.eval()
    final_training_losses = []

    with torch.no_grad():
        for batch in train_loader:
            image = batch['image'].to(device)
            history = batch['history'].to(device)
            target = {
                'delta_position': batch['delta_position'].to(device),
                'delta_orientation': batch['delta_orientation'].to(device),
                'gripper_state': batch['gripper_state'].to(device)
            }
            preds = model(image, history)
            losses = loss_fn(preds, target)
            final_training_losses.append(losses['total'].item())
        
    final_training_loss = np.mean(final_training_losses)
    print(f"Final training loss: {final_training_loss}")


    ## Validation loop
    model.eval()
    val_losses = []

    with torch.no_grad():
        for batch in validation_loader:
            image = batch['image'].to(device)
            history = batch['history'].to(device)
            target = {
                'delta_position': batch['delta_position'].to(device),
                'delta_orientation': batch['delta_orientation'].to(device),
                'gripper_state': batch['gripper_state'].to(device)
            }
            preds = model(image, history)
            losses = loss_fn(preds, target)
            val_losses.append(losses['total'].item())
        
    validation_loss = np.mean(val_losses)
    print(f"Validation loss: {validation_loss}")

    torch.save(model.state_dict(), '/home/joeya/Imitation-Learning/src/object_grasping/object_grasping/Graspingv1.pth')


def objective(trial: optuna.Trial) -> float:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    learning_rate = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    hidden_dim = trial.suggest_int('hidden_dim', low=100, high=500, step=10)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    training_set, validation_set = random_split(dataset, [0.85, 0.15])
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

    model = PolicyNetwork(n_history=N_HISTORY, hidden_dim=hidden_dim).to(device)
    loss_fn = PolicyNetworkLoss().to(device)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=learning_rate)

    for epoch in range(EPOCHS):

        ## Training loop
        model.train()

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
        

        ## Validation loop
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in validation_loader:
                image = batch['image'].to(device)
                history = batch['history'].to(device)
                target = {
                    'delta_position': batch['delta_position'].to(device),
                    'delta_orientation': batch['delta_orientation'].to(device),
                    'gripper_state': batch['gripper_state'].to(device)
                }
                preds = model(image, history)
                losses = loss_fn(preds, target)
                val_losses.append(losses['total'].item())

        
        validation_loss = np.mean(val_losses)

        trial.report(validation_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return validation_loss
    

if __name__ == '__main__':
    # study = optuna.create_study(
    #     direction='minimize',
    #     pruner=optuna.pruners.MedianPruner(),
    #     study_name='imitation_learning'
    # )

    # dataset = load_dataset_from_yaml(CONFIG_PATH, N_HISTORY)

    # study.optimize(objective, n_trials=20, show_progress_bar=True)

    # pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    # complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # print("Study statistics: ")
    # print("  Number of finished trials: ", len(study.trials))
    # print("  Number of pruned trials: ", len(pruned_trials))
    # print("  Number of complete trials: ", len(complete_trials))

    # print("Best trial:")
    # trial = study.best_trial

    # print("  Value: ", trial.value)

    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))


    # fig_importance = plot_param_importances(study)
    # fig_history = plot_optimization_history(study)

    # fig_importance.show()
    # fig_history.show()

    # fig_importance.write_html('param_importances_nhistory1.html')
    # fig_history.write_html('optimization_history_nhistory1.html')


    # train_and_save(trial.params['hidden_dim'], trial.params['batch_size'], trial.params['lr'])


    hidden_dim = 500
    batch_size = 64
    lr = 0.001
    train_and_save(hidden_dim, batch_size, lr)