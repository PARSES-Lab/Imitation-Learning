import torch
from model.model import PolicyNetwork, PolicyNetworkLoss
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = PolicyNetwork(5, 512)
loss_fn = PolicyNetworkLoss()

learning_rate = 1e-2
epochs = 100
optimizer = torch.optim.Adam(model.trainable_parameters(), lr=learning_rate)

