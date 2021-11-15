import scipy 
import numpy as np
import torch
import math

class fisheye_generator():
    
    def __init__(self, sphere_radius, focal_distance = None):
        
        self.sphere_radius = sphere_radius
        self.focal_distance = focal_distance
        if self.focal_distance == None:
            self.focal_distance = self.sphere_radius

 def fisheye_transform(self, image):
        
        image_dims = len(image.shape)
        
        # add batch and channel dimension if missing
        if len(image.shape) <4:
            image = torch.unsqueeze(image, 0)
        
        if len(image.shape) <4:
            image = torch.unsqueeze(image, 0)
        
        (b,c,h,w) = image.shape
        
        h_points = torch.linspace(-1,1,h)
        w_points = torch.linspace(-1,1,w)
        
        h_grid, w_grid = torch.meshgrid(h_points, w_points)
        
        C_grid = 1/torch.sqrt(self.sphere_radius**2 - (h_grid**2 + w_grid**2))*self.focal_distance
        
        h_sample_grid = C_grid*h_grid
        w_sample_grid = C_grid*w_grid
        
        sample_tensor = torch.stack((w_sample_grid, h_sample_grid),2)
        sample_tensor = torch.unsqueeze(sample_tensor, 0)
        sample_tensor = torch.tile(sample_tensor, (b,1,1,1))
        
        fisheye_images = torch.nn.functional.grid_sample(image, sample_tensor)
        
        # crop image
        
        h_top = math.sin(math.atan(self.focal_distance))*(h/2)
        w_top = math.sin(math.atan(self.focal_distance))*(w/2)
        
        
        fisheye_images = fisheye_images[:,:,math.ceil(h/2-h_top):math.floor(h/2+h_top),math.ceil(w/2-w_top):math.floor(w/2+w_top)]
        
        # resize images
        
        fisheye_images = torchvision.transforms.Resize((h,w))(fisheye_images)
        
        if image_dims <= 3:
            fisheye_images = torch.squeeze(fisheye_images, dim=0)
        if image_dims == 2:
            fisheye_images = torch.squeeze(fisheye_images, dim=0)
            
        
        return fisheye_images
        
        
        
        
