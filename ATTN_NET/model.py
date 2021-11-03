import torch
import torch.nn as nn
import cv2
import numpy as np
from einops import rearrange

INPUT_CHANNELS = 3
NUM_CLASSES = 3

# Create Vision Xformer Class
class VisionXformer(nn.Module):
  def __init__(
      self,
      img_shape = (256, 256),
      embed_dim = 128,
      ps = 16,
      n_channels = 3,
      n_xform_layers = 2, 
      n_heads = 4,
      xform_dim = 128,
      n_classes = 3,
      ):
    super(VisionXformer, self).__init__()
    # patch size
    self.Spatch = ps
    # Number of patches
    self.Npatch = (img_shape[0] // ps) ** 2
    # Dimension of patch
    self.Dpatch = np.square(ps) * n_channels
    # Linear projection
    self.project = nn.Linear(self.Dpatch, embed_dim)
    # Positional Classification Tokens
    self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
    # Positional Embeddings
    self.position_embed = nn.Parameter(torch.randn((1, self.Npatch + 1, embed_dim)))


    # Transformer Encoder 
    enc_layer = nn.TransformerEncoderLayer(d_model = xform_dim, nhead = n_heads)
    self.xformer = nn.TransformerEncoder(enc_layer, num_layers = n_xform_layers)

    # Classification Head
    self.class_head = nn.Linear(xform_dim, n_classes)




  def forward(self, x):
    # determine feedforward batch size
    batch_size = x.shape[0]
    # rearrange input data
    x = rearrange(x, 'b c (h ph) (w pw) -> b (h w) (ph pw c)', ph = self.Spatch, pw = self.Spatch)
    # project the patches to the embedded dimension
    x = self.project(x)
    # get the token embeddings
    cls_tok_exp = self.cls_token.repeat(batch_size, 1, 1)
    # concatenate the tokene embeddings and x
    x = torch.cat([cls_tok_exp, x], dim = 1)
    # add the position embeddings to x
    x += self.position_embed

    # send x through the transformer encoder
    x = self.xformer(x)

    # take the mean of x
    x = x.mean(dim = 1)
    # send x through the classification head
    x = self.class_head(x)
    return x

# generate the model to pass to the training function
def gen_model():
  # build and return model
  m = VisionXformer()
  return m


