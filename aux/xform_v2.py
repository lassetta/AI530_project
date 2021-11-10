import pandas as pd
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interp2d, RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import cdist

def transform(img,focus_x, focus_y, mag_rad, mag_d, demag_w):
  print(focus_x, focus_y, mag_rad)
  # define parameters
  # outlined in observablehq.com
  xform_mat = np.zeros((2,2))
  xform_mat[0,0] = demag_w**2 / 2.
  xform_mat[0,1] = 1 - (demag_w*(mag_d + 1)/(mag_d*demag_w + 1))
  xform_mat[1,0] = demag_w
  xform_mat[1,1] = -1*(mag_d + 1) / (mag_d * demag_w + 1)**2 

  bias = np.zeros((2,1))
  bias[0] = 1 - demag_w
  bias[1] = -1
  
  A = np.matmul(np.linalg.inv(xform_mat),bias) 
  print(A)
  xc = 1 - demag_w - (A[0]*(demag_w**2))/2

  # piecewise f function inverse
  f1 = lambda x: (mag_d + 1)*x / ((mag_d * x) + A[1])
  f2 = lambda x: 1 + (1/A[0]) + ((1 / A[0]**2) + (2*(1-x)/A[0]))**0.5
  f_inv1 = lambda x: x - (A[0]*(1-x)**2)/2
  f_inv2 = lambda x: A[1]*x / (mag_d * (1-x) + 1)




  coordinates = []
  # iterate through all i that is in the magnification radius (x-axis)
  for i in range(max(0, int(focus_x-mag_rad)), min(img.shape[0], int(focus_x + mag_rad))):
    # iterate through all j that is in the magnification radius (x-axis)
    for j in range(max(0, int(focus_y-mag_rad)), min(img.shape[1], int(focus_y + mag_rad))):
      # calculate the distance from the center of the magnification
      dist_from_center = ((focus_x - i)**2 + (focus_y - j)**2)**.5
      # if the distance from the center is less than the magnification radius,
      # add the coords
      if dist_from_center < mag_rad:
        #print(i,j)
        coordinates.append((i,j))
  coords = np.array(coordinates)

  #get angle by taking the arctangenct of y / x
  angle = np.arctan2(coords[:,1]-focus_y, coords[:,0] - focus_x)
 
  # get the element wise distance from the center 
  # sqrt((coord_x - center_x)**2 + (coord_y - center_y)**2)
  focus = np.expand_dims(np.array([focus_x, focus_y]), axis = 0)
  norms = np.linalg.norm(coords - focus, axis = 1).flatten() / mag_rad

  # inv_fishey
  # inverse of f given by the piecewise function
  fish_inv = np.zeros_like(norms)
  fish_inv[norms <= (1-demag_w)] = f_inv2(norms[norms <= (1-demag_w)])
  idcs2 = np.logical_and(norms > (1-demag_w), norms < 1)
  fish_inv[idcs2] = f_inv1(norms[idcs2])

  # get the new positions from spherical (angle and radius)
  new_coords = np.zeros_like(coords)
  # x
  new_coords[:,0] = focus_x + np.cos(angle) * mag_rad * fish_inv 
  # y
  new_coords[:,1] = focus_x + np.sin(angle) * mag_rad * fish_inv 

  result = np.zeros_like(img)
  for i in range(img.shape[2]):
    xform = RectBivariateSpline(np.arange(img.shape[0]), np.arange(img.shape[1]), img[:,:,i], kx=2, ky=2)
    transform = xform(new_coords[:,0].flatten(), new_coords[:,1].flatten(), grid = False)
    result[:,:,i] = img[:,:,i]
    result[coords[:,0], coords[:,1], i] = transform
  return result



if __name__ == "__main__":
  path = "doggo.jpg"
  img = cv2.imread(path)
  print(img.shape)
  img2 = transform(img,img.shape[0]/2.,img.shape[1]/2.,min(img.shape[0]//2, img.shape[1]//2),4,.4)
  plt.figure(1)
  plt.imshow(img2)
  plt.show()
