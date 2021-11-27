import pandas as pd
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

from wand.image import Image

def fish_xform(x,y,rad,dist):
  if 1-dist*(rad**2) == 0:
    return x, y
  else:
    ret_x = x / (1 - (dist * (rad**2)))
    ret_y = y / (1 - (dist * (rad**2)))
    return ret_x, ret_y


path = "../Oxford_pets256/Maine/Maine_Coon_1.jpg"
img = cv2.imread(path)
img = cv2.resize(img, (512,512))

w,h = img.shape[0], img.shape[1]

with Image(filename = path) as img:
  print(img.size)
  img.resize(512,512)
  img.virtual_pixel = 'transparent'
  img.distort('barrel', (5.4, 0.0, 0.0, 1))
  img.save(filename = 'check.jpg')


img = cv2.imread("check.jpg")
img = img[112:400, 112:400]
img = cv2.resize(img, (256,256))


plt.imshow(img)
plt.show()

img = cv2.imread(path)
plt.imshow(img)
plt.show()



