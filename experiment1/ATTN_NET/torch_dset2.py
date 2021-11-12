import torch
import numpy as np
import pandas as pd
import glob
from torch.utils.data import Dataset, DataLoader
import cv2

## Dataset class for reading the images from the 
## Data folder
class Dset(Dataset):

  # init constructer for the dataset
  def __init__(self):
    self.img_shape = (256,256)
    # Create a list to store the data
    self.data = []
    # Create a list to store class names:
    self.classes = []
    # define the root path of the images
    self.root_path = "../../data/"
    # get the class folders for the data 
    class_folders = glob.glob(self.root_path + "*")
    # iterate through the class folders
    for cfold in class_folders:
      # get the class label
      label = cfold.split("/")[-1]
      self.classes.append(label)
      # for every image of that label
      for img in glob.glob(cfold + "/*.png"):
        # append the img,label to the dataset
        img = cv2.imread(img)
        img = torch.from_numpy(img)
        self.data.append([img, label])
    # initialize class map dictionary
    self.cmap = {}
    # assign mappings
    for i in range(len(self.classes)):
      cname = self.classes[i]
      self.cmap[cname] = i
  
  # return dataset length
  def __len__(self):
    return len(self.data)

  # get item from the dataset
  def __getitem__(self, i):
    # get the path and label from the selected index
    img, label = self.data[i]
    # read the image
    # img = cv2.imread(img)
    # map the label to its class number
    label_idx = self.cmap[label]
    # convert to pytorch tensors
    label = torch.tensor([label_idx])
    return img, label




# Train network function. Iterates through the network
# and trains for MAX_EPOCHS epochs
def train_network(DL):

  # load imgs and labels from DL
  for imgs, labels in DL:
    # reshaping the images from (N,H,W,C)
    # to (N,C,H,W) for consistency with pytorch 
    # schema
    imgs = imgs.permute(0,3,1,2)
    print(imgs.shape)
    print(labels.shape)
    sys.exit(1)

def get_dset():
  dset = Dset()
  return dset


if __name__ == "__main__":
  # construct dataset
  dset = construct_Dset()
  # construct dataloader
  DL = DataLoader(dset, batch_size=4, shuffle = True)
  # train network
  train_network(DL)





q = Dset()
