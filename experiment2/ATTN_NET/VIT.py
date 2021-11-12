import torch
import numpy as np

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_dset2 import get_dset
from torch.autograd import Variable
from model import gen_model 
import torch.optim as optim
from tqdm import trange
#from vit_pytorch import ViT
from torchvision import transforms
from pytorch_pretrained_vit import ViT

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

trans = transforms.Compose([transforms.Resize(384)])

def eval_network(net, crit, DL):
  # Put in evaluation mode to remove issues
  # such as dropout
  net.eval()
  # count samples
  total_samples = 0
  # count correct
  correct = 0
  # track loss
  total_loss = 0
  for i, (imgs, labels) in enumerate(DL):
    # reshaping the images from (N,H,W,C)
    # to (N,C,H,W) for consistency with pytorch 
    # schema
    #imgs = trans(imgs)
    imgs = torch.div(imgs.permute(0,3,1,2),255.)
    imgs = trans(imgs)
    # converting the data to the device
    imgs, labels = Variable(imgs).to(device), Variable(labels[:,0]).to(device)

    # get the output
    out = net(imgs)
    # get the class with the max
    _, pred = torch.max(out.data, 1)
    # get total samples
    total_samples += labels.size(0)
    # sum number of correct
    correct += (pred == labels.data).sum()
    # get loss
    loss = crit(out, labels)
    # add loss
    total_loss += loss.item()
  return total_loss / (i+1), 100*correct.float() / total_samples









def train_network(DL_train, DL_val,net,crit,opt):
  MAX_EPOCHS = 100
  t = trange(MAX_EPOCHS, desc = 'Loss')
  transform = transforms.Compose(
      [ transforms.Resize(384),
        transforms.RandomCrop(384, padding = 15),
       transforms.RandomHorizontalFlip(p=0.5)])
  running_loss = 0

  for _ in t:
    tot_loss = 0
    net.train()
    for i, (imgs, labels) in enumerate(DL_train, 0):
      # reshaping the images from (N,H,W,C)
      # to (N,C,H,W) for consistency with pytorch 
      # schema
      imgs = imgs.permute(0,3,1,2) / 255.
      imgs = transform(imgs)
      imgs, labels = Variable(imgs).to(device), Variable(labels[:,0]).to(device)

      # zero out optimizers gradient
      opt.zero_grad()
      out = net(imgs)

      # get loss
      loss = crit(out, labels)
      #tot_loss += loss
      #train_loss = float(tot_loss) / (i+1)

      # get backpropagate
      loss.backward()
      # update optimizer
      opt.step()

    train_loss, train_acc = eval_network(net,crit,DL_train)
    val_loss, val_acc = eval_network(net,crit,DL_val)
    t.set_description('Training Loss:{0:.5f}, Training Acc:{1:.2f}%, Validation Loss:{2:.5f}, Validation Acc:{3:.2f}%'.format(train_loss, train_acc, val_loss, val_acc))

      


if __name__ == "__main__":
  # BUILD dataset and dataloader
  dset = get_dset()
  # build the model
  #net = gen_model().to(device) 
  #net = ViT(image_size = 128, patch_size = 8, num_classes=35, dim =256, depth=12, heads=16, mlp_dim = 2048, dropout = 0.0, emb_dropout = 0.0)

  total = len(dset)
  train_size = int(0.8 * total)
  test_size = int(0.2 * total)
  torch.manual_seed(19345678)
  train_dset, val_dset = torch.utils.data.random_split(dset, [train_size, test_size])
  DL_train = DataLoader(train_dset, batch_size=8, shuffle = True)
  DL_val = DataLoader(val_dset, batch_size=8, shuffle = True)
  net = ViT('B_16_imagenet1k', pretrained = True)
  print(net)
  for param in net.parameters():
    param.requires_grad = False
  requires_grad = True
  net.fc = nn.Linear(768, 2)
  net = net.to(device) 
  crit = torch.nn.CrossEntropyLoss()
  opt = optim.Adam(net.parameters(), lr = 1e-3)
  train_network(DL_train, DL_val, net,crit,opt)

