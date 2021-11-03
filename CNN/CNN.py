import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_dset import get_dset
from torch.autograd import Variable
from model import gen_model 
import torch.optim as optim
from tqdm import trange
from torchsummary import summary


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

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
    imgs = imgs.permute(0,3,1,2) / 255.
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
  return total_loss / total_samples, 100*correct.float() / total_samples









def train_network(DL,net,crit,opt):
  MAX_EPOCHS = 100
  t = trange(MAX_EPOCHS, desc = 'Loss')

  for _ in t:
    tot_loss = 0
    net.train()
    for i, (imgs, labels) in enumerate(DL):
      # reshaping the images from (N,H,W,C)
      # to (N,C,H,W) for consistency with pytorch 
      # schema
      imgs = imgs.permute(0,3,1,2) / 255.
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

    train_loss, train_acc = eval_network(net,crit,DL)
    t.set_description('Training Loss:{0:.5f}, Training Acc:{1:.2f}%'.format(train_loss, train_acc))

      


if __name__ == "__main__":
  # BUILD dataset and dataloader
  dset = get_dset()
  DL = DataLoader(dset, batch_size=32, shuffle = True)
  # build the model
  net = gen_model().to(device) 
  crit = torch.nn.CrossEntropyLoss()
  opt = optim.Adam(net.parameters(), lr = 1e-3)
  train_network(DL,net,crit,opt)

