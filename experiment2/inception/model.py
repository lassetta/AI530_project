import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

INPUT_CHANNELS = 3
NUM_CLASSES = 35

class inception(nn.Module):
  def __init__(self, channels, outchannels):
    # call super to initialize nn.Module parent class
    super(inception, self).__init__()
    # init trainable layers
    # dimenstion reduction layers
    self.dim_red1 = nn.Conv2d(channels, int(channels / 2), 1, 1)
    self.dim_red2 = nn.Conv2d(channels, int(outchannels/16), 1, 1)
    # convolutional layers
    self.conv1 = nn.Conv2d(channels, int(outchannels/4), 1, 1)
    self.conv2 = nn.Conv2d(int(channels / 2), int(outchannels/2), 3, 1, padding = 1)
    self.conv3_1 = nn.Conv2d(int(outchannels/16), int(outchannels/8), 3, 1, padding = 1)
    self.conv3_2 = nn.Conv2d(int(outchannels/8), int(outchannels / 8), 3, 1, padding = 1)
    self.conv4 = nn.Conv2d(channels, int(outchannels / 8), 1, 1)

    # maxpool layer
    self.maxpool = nn.MaxPool2d(3,1,padding = 1)
  def forward(self, x):
    # Route 1
    x1 = F.relu(self.conv1(x))
    # Route 2
    x2 = F.relu(self.dim_red1(x))
    x2 = F.relu(self.conv2(x2))
    # Route 3
    x3 = F.relu(self.dim_red2(x))
    x3 = F.relu(self.conv3_1(x3))
    x3 = F.relu(self.conv3_2(x3))
    # Route 4 
    x4 = self.maxpool(x)
    x4 = F.relu(self.conv4(x4))

    # Route 5
    out = torch.cat([x1,x2,x3,x4], dim = 1)
    return out


# initialize a class for the Neural Network model
class CNN_model(nn.Module):
  def __init__(self):
    # call super to initialize nn.Module parent class
    super(CNN_model, self).__init__()
    # init trainable layers
    self.conv1= nn.Conv2d(INPUT_CHANNELS, 32, 3, 1, padding = 1)
    self.inception1 = inception(32,128)
    self.inception2 = inception(128,256)
    self.fc1 = nn.Linear(256*4*4,512)
    self.fc2 = nn.Linear(512,NUM_CLASSES)

  def forward(self, x):
    # conv1 -> relu -> maxpool
    # x shape: [N, 3, 128, 128]
    x = F.max_pool2d(x,2)
    x = F.max_pool2d(x,2)
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x,2)
    # x shape: [N, 64, 32, 32]
    # conv2 -> relu -> maxpool 
    x = self.inception1(x)
    x = F.max_pool2d(x,2)
    # x shape: [N, 128, 16, 16]
    x = self.inception2(x)
    x = F.max_pool2d(x,2)
    # x shape: [N, 256, 8, 8]
    # flatten
    x = torch.flatten(x,1)
    # x shape: [N, 256*8*8] 
    # fc1 -> relu
    x = F.relu(self.fc1(x))
    # fc2 -> no activation
    x = self.fc2(x)
    return x

m = CNN_model()
n = torch.randn((4,3,128,128))
m(n)
# generate the model to pass to the training function
def gen_model():
  # build and return model
  m = CNN_model()
  n = torch.randn((4,3,128,128))

  return m


