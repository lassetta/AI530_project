import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_dset2 import get_dset
from torch.autograd import Variable
from model_batch import gen_model 
import torch.optim as optim
from tqdm import trange
from torchsummary import summary
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time


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
  #transform2 = transforms.Compose(
  #      [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  for i, (imgs, labels) in enumerate(DL):
    # reshaping the images from (N,H,W,C)
    # to (N,C,H,W) for consistency with pytorch 
    # schema
    #imgs = imgs.permute(0,3,1,2) / 255.
    # converting the data to the device
    '''
    imgs2 = imgs[0]
    imgs2 = imgs2.permute(1,2,0)
    plt.imshow(imgs2)
    plt.show()
    '''
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









def train_network(DL_train, DL_val,net,crit,opt,aug,fs):
  MAX_EPOCHS = 400
  t = trange(MAX_EPOCHS, desc = 'Loss')
  transform = transforms.Compose(
      [transforms.RandomCrop(256, padding = 25),
       transforms.RandomHorizontalFlip(p=0.5),
       transforms.RandomVerticalFlip(p=0.25),
       transforms.RandomRotation(90),
       transforms.GaussianBlur(7,(0.1,1.5)),
       transforms.ColorJitter(brightness = 0.2, hue = 0.2)])
  running_loss = 0
  running_time = 0

  accuracy_res = np.zeros((MAX_EPOCHS,4))

  for j in t:
    tot_loss = 0
    net.train()
    for i, data in enumerate(DL_train, 0):

      imgs, labels = data


      # reshaping the images from (N,H,W,C)
      # to (N,C,H,W) for consistency with pytorch 
      # schema
      #imgs = imgs.permute(0,3,1,2) / 255.
      imgs, labels = Variable(imgs).to(device), Variable(labels[:,0]).to(device)
      if aug == True:
        imgs = transform(imgs)


      # zero out optimizers gradient
      start = time.perf_counter()
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
      running_loss += loss.item()
      running_time += time.perf_counter() - start

    train_loss, train_acc = eval_network(net,crit,DL_train)
    accuracy_res[j,0] = train_acc
    val_loss, val_acc = eval_network(net,crit,DL_val)
    accuracy_res[j,1] = val_acc
    accuracy_res[j,2] = train_loss
    accuracy_res[j,3] = val_loss
    t.set_description('Training Loss:{0:.5f}, Training Acc:{1:.2f}%, Validation Loss:{2:.5f}, Validation Acc:{3:.2f}%'.format(train_loss, train_acc, val_loss, val_acc))
  np.savetxt(fs+"acc.csv", accuracy_res, delimiter = ',')

      


if __name__ == "__main__":
  # BUILD dataset and dataloader
  dset = get_dset()
  total = len(dset)
  train_size = int(0.8 * total)
  val_size = int(0.2 * total)
  torch.manual_seed(19345678)
  train_dset, val_dset = torch.utils.data.random_split(dset, [train_size, val_size])
  DL_train = DataLoader(train_dset, batch_size=32, shuffle = True)
  DL_val = DataLoader(val_dset, batch_size=32, shuffle = True)
  # build the model
  aug = ['aug', 'no_aug']
  aug2 = [True, False]
  for i in range(2):
    net = gen_model().to(device) 
    crit = torch.nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr = 1e-4)
    train_network(DL_train, DL_val,net,crit,opt,aug2[i],aug[i])

