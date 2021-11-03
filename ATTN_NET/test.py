import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange

p = 4
img = cv2.imread("../data/CheckercubeAred/CheckercubeA_0002.png")
#print(img)
#plt.imshow(img)
#plt.show()

BATCH_SIZE = 4
NUM_CHANNELS = 3
IMG_WIDTH = 256
IMG_HEIGHT = 256
ps = 4

img = torch.rand((BATCH_SIZE,NUM_CHANNELS,IMG_WIDTH,IMG_HEIGHT))

# use einops for rearranging the vision transformer

# reshape 

# transform the image tensor (BatchSize, Channel, W, H) into flattened patches
# of size (BatchSize, h*w/p^2, channels*p^2)
#num_patches = np.square(ps) * NUM_CHANNELS 
#patched = rearrange(img, 'b c (h ph) (w pw) -> b (h w) (ph pw c)', ph = ps, pw = ps)
#print(patched.shape)
#encoder_


class Embed(nn.Module):
  def __init__(
      self,
      img_shape = (256, 256),
      embed_dim = 128,
      ps = 4,
      ):
    super(Embed, self).__init__()
    # patch size
    self.Spatch = ps
    # Number of patches
    self.Npatch = (img_shape[0] // ps) ** 2
    # Dimension of patch
    self.Dpatch = np.square(ps) * NUM_CHANNELS
    # Linear projection
    self.project = nn.Linear(self.Dpatch, embed_dim)
    # Positional Classification Tokens
    self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
    # Positional Embeddings
    self.position_embed = nn.Parameter(torch.randn((1, self.Npatch + 1, embed_dim)))

  def forward(self, x):
    batch_size = x.shape[0]
    x = rearrange(x, 'b c (h ph) (w pw) -> b (h w) (ph pw c)', ph = self.Spatch, pw = self.Spatch)
    x = self.project(x)
    cls_tok_exp = self.cls_token.repeat(batch_size, 1, 1)
    x = torch.cat([cls_tok_exp, x], dim = 1)
    x += self.position_embed

    return x

img = torch.rand((BATCH_SIZE,NUM_CHANNELS,IMG_WIDTH,IMG_HEIGHT))
m = Embed()
x = m(img)
print(x.shape)
m2 = nn.TransformerEncoderLayer(d_model = 128, nhead = 4)
m3 = nn.TransformerEncoder(m2, num_layers=2)
y = m3(x)
y = y.mean(dim = 1)
print(y.shape)
m4 = nn.Linear(128, 3)
z = m4(y)
print(z.shape)





