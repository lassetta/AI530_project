import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

INPUT_CHANNELS = 3
NUM_CLASSES = 2


# initialize a class for the Neural Network model
class MLP_model(nn.Module):
  def __init__(self):
    # call super to initialize nn.Module parent class
    super(MLP_model, self).__init__()
    self.fc1 = nn.Linear(3*256*256,512)
    self.fc2 = nn.Linear(512,512)
    self.fc3 = nn.Linear(512,512)
    self.fc4 = nn.Linear(512,512)
    self.fc5 = nn.Linear(512,NUM_CLASSES)
    # init trainable layers

  def forward(self, x, activation):
    x = torch.flatten(x,1)
    x = activation(self.fc1(x))
    x = activation(self.fc2(x))
    x = activation(self.fc3(x))
    x = activation(self.fc4(x))
    x = self.fc5(x)
    return x

# generate the model to pass to the training function
'''
m = MLP_model()
img = torch.randn(4,3,256,256)
out = m(img, F.relu)
print(out.shape)
'''


def gen_model():
  # build and return model
  m = MLP_model()
  return m


