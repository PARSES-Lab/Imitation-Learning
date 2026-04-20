from model.model import PolicyNetwork, PolicyNetworkLoss
import torch
from torch.utils.data import DataLoader
from train import load_dataset_from_yaml
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
import numpy as np

N_HISTORY = 1
HIDDEN_DIM = 450
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


with torch.no_grad():

    image_path = '/home/joeya/dataset/demo1/1775166724_106757568.pt'
    image = torch.load(image_path)

    preprocess = v2.Compose([
            v2.ToImage(),
            v2.Resize(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
    
    image = preprocess(image).unsqueeze(0)

    current_gripper = 0.0
    current_translation = [-0.1968760705967593,0.0841186242806411,0.4108139484172628]
    current_rotation = [0.7318828101672257,0.6808134555411351,-0.0252402552586923,-0.0142660593837099]

    current_rotation = np.array(current_rotation)
    if current_rotation[3] < 0:
        current_rotation *= -1

    current_rotation = current_rotation.tolist()
    
    latest_state = current_translation + current_rotation + [current_gripper]
    latest_state = torch.tensor(latest_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 8)

    preds = model(image, latest_state)

    preds['delta_position'] /= 100

    preds['delta_orientation'] = F.normalize(preds['delta_orientation'], dim=-1)

    ## enforce positive quaternion convention
    if preds['delta_orientation'][0][3] < 0:
        preds['delta_orientation'] *= -1

    print("   -------------------------------------------   \n")
    print("Start")
    print(f"Predictions: {preds}")
    print("   -------------------------------------------   \n")




    image_path = '/home/joeya/dataset/demo9/1775167627_293373535.pt'
    image = torch.load(image_path)

    preprocess = v2.Compose([
            v2.ToImage(),
            v2.Resize(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
    
    image = preprocess(image).unsqueeze(0)

    current_gripper = 0.0
    current_translation = [-0.2383464235542431,0.0428426408910344,0.3096116306828636]
    current_rotation = [0.6764321183170762,0.7363468340868781,-0.0138491869693604,-0.0064132097729874]

    current_rotation = np.array(current_rotation)
    if current_rotation[3] < 0:
        current_rotation *= -1

    current_rotation = current_rotation.tolist()
    
    latest_state = current_translation + current_rotation + [current_gripper]
    latest_state = torch.tensor(latest_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 8)

    preds = model(image, latest_state)

    preds['delta_position'] /= 100

    preds['delta_orientation'] = F.normalize(preds['delta_orientation'], dim=-1)

    ## enforce positive quaternion convention
    if preds['delta_orientation'][0][3] < 0:
        preds['delta_orientation'] *= -1

    print("   -------------------------------------------   \n")
    print("moving down")
    print(f"Predictions: {preds}")
    print("   -------------------------------------------   \n")




    image_path = '/home/joeya/dataset/demo9/1775167632_530427979.pt'
    image = torch.load(image_path)

    preprocess = v2.Compose([
            v2.ToImage(),
            v2.Resize(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
    
    image = preprocess(image).unsqueeze(0)

    current_gripper = 0.0
    current_translation = [-0.2487118554305202,0.0462103540892433,0.2108969684218323]
    current_rotation = [0.6839428042715993,0.7283843833174137,-0.0295233996815815,-0.0284042161598112]

    current_rotation = np.array(current_rotation)
    if current_rotation[3] < 0:
        current_rotation *= -1

    current_rotation = current_rotation.tolist()
    
    latest_state = current_translation + current_rotation + [current_gripper]
    latest_state = torch.tensor(latest_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 8)

    preds = model(image, latest_state)

    preds['delta_position'] /= 100

    preds['delta_orientation'] = F.normalize(preds['delta_orientation'], dim=-1)

    ## enforce positive quaternion convention
    if preds['delta_orientation'][0][3] < 0:
        preds['delta_orientation'] *= -1

    print("   -------------------------------------------   \n")
    print("About to grasp")
    print(f"Predictions: {preds}")
    print("   -------------------------------------------   \n")





    image_path = '/home/joeya/dataset/demo9/1775167636_533322021.pt'
    image = torch.load(image_path)

    preprocess = v2.Compose([
            v2.ToImage(),
            v2.Resize(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
    
    image = preprocess(image).unsqueeze(0)

    current_gripper = 1.0
    current_translation = [-0.2579776250946557,0.043385502598997,0.2859757745033345]
    current_rotation = [.6837276611784321,0.7279804953156057,-0.0385539640842316,-0.0327791951229465]

    current_rotation = np.array(current_rotation)
    if current_rotation[3] < 0:
        current_rotation *= -1

    current_rotation = current_rotation.tolist()
    
    latest_state = current_translation + current_rotation + [current_gripper]
    latest_state = torch.tensor(latest_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 8)

    preds = model(image, latest_state)

    preds['delta_position'] /= 100

    preds['delta_orientation'] = F.normalize(preds['delta_orientation'], dim=-1)

    ## enforce positive quaternion convention
    if preds['delta_orientation'][0][3] < 0:
        preds['delta_orientation'] *= -1

    print("   -------------------------------------------   \n")
    print("grasped, moving up")
    print(f"Predictions: {preds}")
    print("   -------------------------------------------   \n")
