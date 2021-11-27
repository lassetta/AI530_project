import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch

INPUT_CHANNELS = 3
NUM_CLASSES = 2


# initialize a class for the Neural Network model
class res_model(nn.Module):
  def __init__(self):
    # call super to initialize nn.Module parent class
    super(res_model, self).__init__()
    self.fc1 = nn.Linear(3*64*64,512)
    self.fc2 = nn.Linear(512,512)
    self.fc3 = nn.Linear(512,512)
    self.fc4 = nn.Linear(512,512)
    self.fc5 = nn.Linear(512,512)
    self.fc6 = nn.Linear(512,512)
    self.fc7 = nn.Linear(512,NUM_CLASSES)

  def forward(self, x):
    x = F.max_pool2d(x,2)
    x = F.max_pool2d(x,2)
    x = torch.flatten(x,1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = F.relu(self.fc5(x))
    x = F.relu(self.fc6(x))
    # fc2 -> no F.relu
    x = self.fc7(x)
    return x


# generate the model to pass to the training function

m = res_model()
img = torch.randn((4,3,256,256))
m(img)





def gen_model():
  # build and return model
  m = res_model()
  return m


