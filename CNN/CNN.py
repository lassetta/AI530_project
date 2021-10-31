import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_dset import get_dset
from model import gen_model 
import torch.optim as optim
from tqdm import trange


device = torch.device("cuda")

def train_network(DL,net,crit,opt):
  MAX_EPOCHS = 100
  t = trange(MAX_EPOCHS, desc = 'Loss')

  for _ in t:
    tot_loss = 0
    for i, (imgs, labels) in enumerate(DL):
      # reshaping the images from (N,H,W,C)
      # to (N,C,H,W) for consistency with pytorch 
      # schema
      net.train()
      imgs = imgs.permute(0,3,1,2) / 255.
      imgs, labels = imgs.to(device), labels.to(device)

      # zero out optimizers gradient
      opt.zero_grad()
      out = net(imgs)

      # get loss
      loss = crit(out, labels[:,0])
      tot_loss += loss
      train_loss = float(tot_loss) / (i+1)

      # get backpropagate
      loss.backward()
      # update optimizer
      opt.step()

      t.set_description('Training Loss:{0:.5f}'.format(train_loss))

      


if __name__ == "__main__":
  # BUILD dataset and dataloader
  dset = get_dset()
  DL = DataLoader(dset, batch_size=32, shuffle = True)
  # build the model
  net = gen_model().to(device) 
  crit = torch.nn.CrossEntropyLoss()
  opt = optim.Adam(net.parameters(), lr = 1e-3)
  train_network(DL,net,crit,opt)

