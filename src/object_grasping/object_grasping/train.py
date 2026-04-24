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
from optuna.visualization import plot_param_importances, plot_optimization_history, plot_contour
from optuna.trial import TrialState

POSE_COLS = ['x', 'y', 'z']
POSE_GRIPPER_COLS = ['x', 'y', 'z', 'gripper']
CONFIG_PATH='/home/joeya/Imitation-Learning/src/object_grasping/object_grasping/dataset_config.yaml'
EPOCHS = 40

dataset = None


class SingleDemoDataset(Dataset):
    def __init__(self, dataset_csv_path, image_dir):
        self.dataset_df = pd.read_csv(dataset_csv_path)
        self.image_dir = Path(image_dir)

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
        return len(self.dataset_df)
    
    
    def __getitem__(self, index):
        target_idx = self.dataset_df.iloc[index]['target']

        image = torch.load(
            self.image_dir /
            Path(self.dataset_df.iloc[index]['image']).with_suffix('.pt')
        )
        image = self.augmentation(image)

        
        current_pose = self.dataset_df.iloc[index][POSE_GRIPPER_COLS].to_numpy(dtype=np.float32)
        target_pose = self.dataset_df.iloc[target_idx][POSE_GRIPPER_COLS].to_numpy(dtype=np.float32)


        delta_position = target_pose[:3] - current_pose[:3]

        gripper_state = float(self.dataset_df.iloc[target_idx]['gripper'])

        return {
            'image': image,
            'current_state': current_pose,
            'delta_position': torch.tensor(delta_position, dtype=torch.float32) * 100,
            'gripper_state': torch.tensor([gripper_state], dtype=torch.float32),
        }
    

def load_dataset_from_yaml(config_path) -> ConcatDataset:
    """Expects YAML format:
        
        demos:
            - dataset_csv: path1
              image_dir: image_path1
            - dataset_csv: path2
              image_dir: image_path2
    """

    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    demos = config.get('demos', [])
    if not demos:
        raise ValueError(f'No demos found in {config_path}')

    print(f"Loading {len(demos)} demonstrations...")
    print(f"Using demo {len(demos)} as a test set")

    test_demo = demos[len(demos) - 1]
    demos = demos[0:len(demos) - 1]

    datasets = [
        SingleDemoDataset(
            dataset_csv_path=demo['dataset_csv'],
            image_dir=demo['image_dir']
        )
        for demo in demos
    ]

    test_set = SingleDemoDataset(dataset_csv_path=test_demo['dataset_csv'], image_dir=test_demo['image_dir'])

    combined = ConcatDataset(datasets)
    print(f"Total samples across all demos (excluding test): {len(combined)}")
    return combined, test_set



def train_and_save(
        hidden_dim,
        lr,
        weights_path
):
    dataset, test_set = load_dataset_from_yaml(CONFIG_PATH)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}')

    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=12, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=12, pin_memory=True)

    model = PolicyNetwork(hidden_dim=hidden_dim).to(device)
    loss_fn = PolicyNetworkLoss().to(device)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=lr)


    for epoch in range(EPOCHS):
        model.train()
        train_losses = []

        for batch in train_loader:
            image = batch['image'].to(device)
            current_state = batch['current_state'].to(device)
            target = {
                'delta_position': batch['delta_position'].to(device),
                'gripper_state': batch['gripper_state'].to(device)
            }

            optimizer.zero_grad()
            preds = model(image, current_state)
            losses = loss_fn(preds, target)
            losses['total'].backward()
            optimizer.step()

            train_losses.append(losses['total'].item())
        print(f'Epoch {epoch}: Loss is {np.mean(train_losses)}')


    ## Test loop
    model.eval()
    test_losses = []

    with torch.no_grad():
        for batch in test_loader:
            image = batch['image'].to(device)
            current_state = batch['current_state'].to(device)
            target = {
                'delta_position': batch['delta_position'].to(device),
                'gripper_state': batch['gripper_state'].to(device)
            }
            preds = model(image, current_state)
            losses = loss_fn(preds, target)
            test_losses.append(losses['total'].item())
        
    test_loss = np.mean(test_losses)
    print(f"Test loss: {test_loss}")

    torch.save(model.state_dict(), weights_path)


def objective(trial: optuna.Trial) -> float:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    learning_rate = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    hidden_dim = trial.suggest_int('hidden_dim', low=30, high=200, step=10)

    training_set, validation_set = random_split(dataset, [0.85, 0.15])
    train_loader = DataLoader(training_set, batch_size=64, shuffle=True, num_workers=12, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=64, shuffle=True, num_workers=12, pin_memory=True)

    model = PolicyNetwork(hidden_dim=hidden_dim).to(device)
    loss_fn = PolicyNetworkLoss().to(device)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=learning_rate)

    for epoch in range(EPOCHS):

        ## Training loop
        model.train()

        for batch in train_loader:
            image = batch['image'].to(device)
            current_state = batch['current_state'].to(device)
            target = {
                'delta_position': batch['delta_position'].to(device),
                'gripper_state': batch['gripper_state'].to(device)
            }

            optimizer.zero_grad()
            preds = model(image, current_state)
            losses = loss_fn(preds, target)
            losses['total'].backward()
            optimizer.step()
        

        ## Validation loop
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in validation_loader:
                image = batch['image'].to(device)
                current_state = batch['current_state'].to(device)
                target = {
                    'delta_position': batch['delta_position'].to(device),
                    'gripper_state': batch['gripper_state'].to(device)
                }
                preds = model(image, current_state)
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

    # dataset, test_set = load_dataset_from_yaml(CONFIG_PATH, N_HISTORY)

    # study.optimize(objective, n_trials=30, show_progress_bar=True)

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
    # fig_contour = plot_contour(study, params=["lr", "hidden_dim"])

    # fig_importance.show()
    # fig_history.show()
    # fig_contour.show()

    # fig_importance.write_html('plots/param_importances_no_orientations.html')
    # fig_history.write_html('plots/optimization_history_no_orientations.html')
    # fig_contour.write_html('plots/contour.html')


    # train_and_save(trial.params['hidden_dim'], trial.params['lr'], '/home/joeya/Imitation-Learning/src/object_grasping/object_grasping/trained_models/Graspingv7.pth')


    hidden_dim = 200
    lr = 0.002
    train_and_save(hidden_dim, lr, '/home/joeya/Imitation-Learning/src/object_grasping/object_grasping/trained_models/Graspingv7.pth')