from model.model import PolicyNetwork, PolicyNetworkLoss
import torch
from torch.utils.data import DataLoader
from train import load_dataset_from_yaml
import torch.nn.functional as F

N_HISTORY = 1
HIDDEN_DIM = 256
MODEL_WEIGHTS_PATH = '/home/joeya/Imitation-Learning/src/object_grasping/object_grasping/Graspingv1.pth'
CONFIG_PATH = '/home/joeya/Imitation-Learning/src/object_grasping/object_grasping/dataset_config.yaml'

model = PolicyNetwork(n_history=N_HISTORY, hidden_dim=HIDDEN_DIM)
state_dict = torch.load(MODEL_WEIGHTS_PATH, weights_only=True)
model.load_state_dict(state_dict)

dataset = load_dataset_from_yaml(CONFIG_PATH, N_HISTORY)
validation_loader = DataLoader(dataset, shuffle=True)

loss_fn = PolicyNetworkLoss()

model.eval()
with torch.no_grad():
    i = 0
    for batch in validation_loader:
        if i == 5:
            break
        
        image = batch['image']
        history = batch['history']
        target = {
            'delta_position': batch['delta_position'],
            'delta_orientation': batch['delta_orientation'],
            'gripper_state': batch['gripper_state']
        }
        preds = model(image, history)
        losses = loss_fn(preds, target)

        preds['delta_position'] /= 100
        target['delta_position'] /= 100

        preds['delta_orientation'] = F.normalize(preds['delta_orientation'], dim=-1)

        ## enforce positive quaternion convention
        if preds['delta_orientation'][0][3] < 0:
            preds['delta_orientation'] *= -1

        print("   -------------------------------------------   \n")
        print(f"Predictions: {preds}")
        print(f"Target: {target}")
        print(f"Losses: {losses}")
        print("   -------------------------------------------   \n")


        i+=1
