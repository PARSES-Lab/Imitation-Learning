import torch
import torch.nn as nn
from .spatial_softmax import SpatialSoftArgmax
import torchvision.models as models

class PolicyNetwork(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        ## Resnet18 encoder
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        ## Freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        ## Spatial softmax
        self.spatial_softmax = SpatialSoftArgmax(normalize=True)

        ## Feedforward layers
        encoder_output_dim = 512 * 2
        ff_input_dim = encoder_output_dim + 3

        self.feedforward = nn.Sequential(
            nn.Linear(ff_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    
    def forward(self, image: torch.Tensor, pose_history: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = image.shape[0]

        with torch.no_grad():
            encoder_output = self.encoder(image)

        keypoints = self.spatial_softmax(encoder_output)
        poses_flat = pose_history.view(batch_size, -1)
        ff_input = torch.cat([keypoints, poses_flat], dim=1)

        output = self.feedforward(ff_input)

        return {
            'delta_position': output[:, :3],
            'gripper_state': output[:, 3:4]
        }
    
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
    


class PolicyNetworkLoss(nn.Module):
    def __init__(self, position_weight=1, gripper_weight=.8):
        super().__init__()
        self.position_weight = position_weight
        self.gripper_weight = gripper_weight
        self.huber_loss = nn.HuberLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
            self,
            prediction: dict[str, torch.Tensor],
            target: dict[str, torch.Tensor],
    ):
        position_loss = self.huber_loss(prediction['delta_position'], target['delta_position'])
        gripper_loss = self.bce(prediction['gripper_state'], target['gripper_state'])
        total = self.position_weight * position_loss + self.gripper_weight * gripper_loss

        return {
            'position_loss': position_loss,
            'gripper_loss': gripper_loss,
            'total': total
        }
    

if __name__ == '__main__':
    model = PolicyNetwork(5, 256)
    loss_fn = PolicyNetworkLoss()

    images = torch.randn(5, 3, 224, 224)
    pose_history = torch.randn(5, 5, 8)
    preds = model(images, pose_history)
    print(preds)

    target_delta_position = torch.randn(5, 3)
    target_delta_orientation = torch.randn(5, 4)
    target_gripper_state = torch.randint(0, 2, (5, 1)).float()

    loss = loss_fn(
        preds,
        {'delta_position': target_delta_position,
         'delta_orientation': target_delta_orientation,
         'gripper_state': target_gripper_state}
    )
    print(loss)