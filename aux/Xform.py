import pandas as pd
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

def fish_xform(x,y,rad,dist):
  if 1-dist*(rad**2) == 0:
    return x, y
  else:
    ret_x = x / (1 - (dist * (rad**2)))
    ret_y = y / (1 - (dist * (rad**2)))
    return ret_x, ret_y


path = "doggo.jpg"
img = cv2.imread(path)

w,h = img.shape[0], img.shape[1]


#img = np.dstack((img, np.full((w,h),255)))
#print(img.shape)

dest = np.zeros_like(img)
print(dest.shape)

dist = .3

for x in tqdm(range(dest.shape[0])):
  for y in range(dest.shape[1]):
    norm_x = float((2*x - w)/w)
    norm_y = float((2*y - h)/h)

    rad = np.sqrt(norm_x ** 2 + norm_y ** 2)

    dest_x, dest_y = fish_xform(norm_x, norm_y, rad, dist)

    dest_x = int(((dest_x + 1)*w)/2)
    dest_y = int(((dest_y + 1)*w)/2)

    if 0 <= dest_x and dest_x < img.shape[0] and 0 <= dest_y and dest_y < img.shape[1]:
      dest[x,y] = img[dest_x,dest_y]

plt.imshow(dest[:,:,0:3])
plt.show()

plt.imshow(img)
plt.show()

