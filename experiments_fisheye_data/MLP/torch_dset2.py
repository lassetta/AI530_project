import torch
import numpy as np
import pandas as pd
import glob
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt

dogs = ['american', 'beagle', 'basset', 'boxer', 'chihuahua', 'english',
    'german', 'great', 'havanese', 'japanese', 'keeshond', 'leonberger', 
    'miniature', 'newfoundland', 'pomeranian', 'pug', 'saint', 'samoeyed', 
    'scottish', 'staffordshire', 'wheaten', 'yorkshire']
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
    self.root_path = "../../Oxford_pets_fish/"
    # get the class folders for the data 
    class_folders = glob.glob(self.root_path + "*")
    print(class_folders)
    # iterate through the class folders
    for cfold in class_folders:
      # get the class label
      label = cfold.split("/")[-1]
      print(label)
      self.classes.append(label)
      # for every image of that label
      for img in glob.glob(cfold + "/*.jpg"):
        # append the img,label to the dataset
        img = cv2.imread(img)
        img = torch.from_numpy(img)
        img = img.permute(2,0,1) / 255.
        img = img*2 - 1
        #self.data.append([img, label])
        if label in dogs:
          self.data.append([img, 0])
        else:
          self.data.append([img, 1])
    # initialize class map dictionary
    self.cmap = {}
    # assign mappings
    '''
    for i in range(len(self.classes)):
      cname = self.classes[i]
      self.cmap[cname] = i
    '''
  
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
    # label_idx = self.cmap[label]
    # convert to pytorch tensors
    # label = torch.tensor([label_idx])
    label = torch.tensor([label])
    return img, label




# Train network function. Iterates through the network
# and trains for MAX_EPOCHS epochs
def train_network(DL):

  # load imgs and labels from DL
  for imgs, labels in DL:
    # reshaping the images from (N,H,W,C)
    # to (N,C,H,W) for consistency with pytorch 
    # sche
    if labels == 10:
      imgs = imgs[0]
      imgs = imgs.permute(1,2,0) 
      plt.imshow(imgs)
      plt.show()

def get_dset():
  dset = Dset()
  print("yay")
  return dset


if __name__ == "__main__":
  # construct dataset
  dset = get_dset()
  # construct dataloader
  DL = DataLoader(dset, batch_size=1, shuffle = True)
  # train network
  train_network(DL)





