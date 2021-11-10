# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 23:09:16 2021

@author: Michael
"""
import torch
import torch.nn.functional as F

class transforms():
    
    def __init__(self, dimensional_transform, image_transform):
     
        self.dimensional_transform = dimensional_transform
        self.image_transform = image_transform
    
    def __call__(self, image, segmentation_mask = None):
        
        #concatenate image and segmentation mask together
        
        if segmentation_mask != None:
            
            cat_image_mask = torch.cat((image, segmentation_mask))
            
            transformed_image_mask = self.dimensional_transform(cat_image_mask)
            
            transformed_image = transformed_image_mask[0:3,:,:]
            transformed_segmentation_mask = transformed_image_mask[3:,:,:]
            
            transformed_image = self.image_transform(transformed_image)
            
            return transformed_image, transformed_segmentation_mask
        
        else:
            
            transformed_image_tensor = self.dimensional_transform(image)
            transformed_image_tensor = self.image_transform(image)
            
            return transformed_image_tensor         
        
class scale():

    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, tensor):
        return tensor/self.scale_factor
        
# class fisheye_generator():
    
#     def __init__(self, parameters):
        
        
#     def __call__(self, image):
        
#         assuming optical center is at the center of the image
        
#         (channel, H, W) = image.shape
        
#         y_center = H/2
#         x_center = W/2
        
#         sample_grid = torch.zeros((H,W,2))
        
#         # for every point in the disroted image, compute the corresponding
#         # undistorted image sample point
        
#         for y in range(H):
#             for x in range(W):
                
#                 r_u = r_d*()
                
#                 sample_grid[y,x,0] = sample_y/H
#                 sample_grid[y,x,1] = sample_x/W
                    
#         # expand to channel size since sample point the same across channels
        
#         sample_grid = sample_grid.unsqueeze()
#         sample_grid = sample_grid.repeat((channel,1,1))
        
#         # interpolate fisheye image 
        
#         fisheye_image = F.grid_sample(image,sample_grid)
        
#         return fisheye_image
    
#     def undistorted_location(self, x, y):
        
#         return undistroted_x, undistroted_y

        
    
    
        
        
        
        
               
            
                
            
            
        
        