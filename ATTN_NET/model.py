import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

INPUT_CHANNELS = 3
NUM_CLASSES = 3


# initialize a class for the Neural Network model
class CNN_model(nn.Module):
  def __init__(self):
    # call super to initialize nn.Module parent class
    super(CNN_model, self).__init__()
    # init trainable layers
    self.conv1= nn.Conv2d(INPUT_CHANNELS, 32, 3, 1, padding = 1)
    self.conv2= nn.Conv2d(32, 64, 3, 1, padding = 1)
    self.fc1 = nn.Linear(64*64*64,512)
    self.fc2 = nn.Linear(512,NUM_CLASSES)

  def forward(self, x):
    # conv1 -> relu -> maxpool
    # x shape: [N, 3, 256, 256]
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x,2)
    # x shape: [N, 32, 128, 128]
    # conv2 -> relu -> maxpool 
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x,2)
    # x shape: [N, 64, 64, 64]
    # flatten
    x = torch.flatten(x,1)
    # x shape: [N, 64^3] 
    # fc1 -> relu
    x = F.relu(self.fc1(x))
    # fc2 -> no activation
    x = self.fc2(x)
    return x

# generate the model to pass to the training function
def gen_model():
  # build and return model
  m = CNN_model()
  return m


