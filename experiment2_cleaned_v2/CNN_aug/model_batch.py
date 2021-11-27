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
    # init trainable layers
    self.conv_pre= nn.Conv2d(INPUT_CHANNELS, 16, 3, 1, padding = 1)
    self.conv1= nn.Conv2d(16, 16, 3, 1, padding = 1)
    self.conv1_2= nn.Conv2d(16, 16, 3, 1, padding = 1)
    self.conv1_3= nn.Conv2d(16, 16, 3, 1, padding = 1)
    self.conv1_4= nn.Conv2d(16, 16, 3, 1, padding = 1)
    self.conv1to2 = nn.Conv2d(16,32,3,2,padding = 1)
    self.conv1to2_2= nn.Conv2d(32, 32, 3, 1, padding = 1)
    self.conv_block1 = nn.Conv2d(16,32,1,2)


    self.conv2= nn.Conv2d(32, 32, 3, 1, padding = 1)
    self.conv2_2= nn.Conv2d(32, 32, 3, 1, padding = 1)
    self.conv2_3= nn.Conv2d(32, 32, 3, 1, padding = 1)
    self.conv2_4= nn.Conv2d(32, 32, 3, 1, padding = 1)
    self.conv2to3 = nn.Conv2d(32,64,3,2,padding = 1)
    self.conv2to3_2 = nn.Conv2d(64,64,3,1,padding = 1)
    self.conv_block2 = nn.Conv2d(32,64,1,2)

    self.conv3= nn.Conv2d(64, 64, 3, 1, padding = 1)
    self.conv3_2= nn.Conv2d(64, 64, 3, 1, padding = 1)
    self.conv3_3= nn.Conv2d(64, 64, 3, 1, padding = 1)
    self.conv3_4= nn.Conv2d(64, 64, 3, 1, padding = 1)
    self.fc1 = nn.Linear(64*8*8,1024)
    self.fc2 = nn.Linear(1024,1024)
    self.fc3 = nn.Linear(1024,NUM_CLASSES)

    self.drop1 = nn.Dropout(p=0.3)
    self.drop2 = nn.Dropout(p=0.3)

    self.batch_norm1 = nn.BatchNorm2d(16)
    self.batch_norm2 = nn.BatchNorm2d(16)
    self.batch_norm3 = nn.BatchNorm2d(16)
    self.batch_norm4 = nn.BatchNorm2d(16)
    self.batch_norm5 = nn.BatchNorm2d(16)

    self.batch_norm6 = nn.BatchNorm2d(32)
    self.batch_norm7 = nn.BatchNorm2d(32)
    self.batch_norm8 = nn.BatchNorm2d(32)
    self.batch_norm9 = nn.BatchNorm2d(32)
    self.batch_norm10 = nn.BatchNorm2d(32)
    self.batch_norm11 = nn.BatchNorm2d(32)

    self.batch_norm12 = nn.BatchNorm2d(64)
    self.batch_norm13 = nn.BatchNorm2d(64)
    self.batch_norm14 = nn.BatchNorm2d(64)
    self.batch_norm15 = nn.BatchNorm2d(64)
    self.batch_norm16 = nn.BatchNorm2d(64)
    self.batch_norm17 = nn.BatchNorm2d(64)

    self.batch_norm18 = nn.BatchNorm1d(1024)
    self.batch_norm19 = nn.BatchNorm1d(1024)

  def forward(self, x):
    # conv1 -> relu -> maxpool
    # x shape: [N, 3, 256, 256]
    x = F.max_pool2d(x,2)
    x = F.max_pool2d(x,2)
    x = F.relu(self.batch_norm1(self.conv_pre(x)))
    x = F.relu(self.batch_norm2(self.conv1(x)))
    x = F.relu(self.batch_norm3(self.conv1_2(x)))
    x = F.relu(self.batch_norm4(self.conv1_3(x)))
    x = F.relu(self.batch_norm5(self.conv1_4(x)))
    x = F.relu(self.batch_norm6(self.conv1to2(x)))
    x = F.relu(self.batch_norm7(self.conv1to2_2(x)))
    # x shape: [N, 32, 128, 128]
    # conv2 -> relu -> maxpool 
    x = F.relu(self.batch_norm8(self.conv2(x)))
    x = F.relu(self.batch_norm9(self.conv2_2(x)))
    x = F.relu(self.batch_norm10(self.conv2_3(x)))
    x = F.relu(self.batch_norm11(self.conv2_4(x)))
    x = F.relu(self.batch_norm12(self.conv2to3(x)))
    x = F.relu(self.batch_norm13(self.conv2to3_2(x)))
    # x shape: [N, 64, 64, 64]
    # flatten
    x = F.relu(self.batch_norm14(self.conv3(x)))
    x = F.relu(self.batch_norm15(self.conv3_2(x)))
    x = F.relu(self.batch_norm16(self.conv3_3(x)))
    x = F.relu(self.batch_norm17(self.conv3_4(x)))
    #x = F.relu(self.conv3to4(x))
    # x shape: [N, 64, 64, 64]
    '''
    x = F.relu(self.conv4(x))
    x = F.relu(self.conv4_2(x))
    x = F.relu(self.conv4_3(x))
    x = F.relu(self.conv4_4(x))
    '''
    x = F.max_pool2d(x,2)

    x = torch.flatten(x,1)
    # x shape: [N, 64^3] 
    # fc1 -> relu
    x = F.relu(self.batch_norm18(self.fc1(x)))
    x = self.drop1(x)
    x = F.relu(self.batch_norm19(self.fc2(x)))
    x = self.drop2(x)
    # fc2 -> no F.relu
    x = self.fc3(x)
    return x


# generate the model to pass to the training function

m = res_model()
img = torch.randn((4,3,256,256))
print(img.shape)
out = m(img)





def gen_model():
  # build and return model
  m = res_model()
  return m


