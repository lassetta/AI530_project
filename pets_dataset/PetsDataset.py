# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 15:10:34 2021

@author: Michael
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import os
import csv
import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms
import torchvision.utils
import torch


# class for operations with pets dataset, give base folder path
# which contains two sub folders, \images\ and \annotations\.
# \images\ contains images and \annotations\ contains image information
# including test.txt and trainval.txt which list the training and testing sets
# segmentation information is provided in \trimaps\ subfolder of annotations and xmls
# contains face boxes of pets 

class PetsDataset(Dataset):
    
    # set base folder path, of pets_dataset and generate sample list from csv_file_path
    # assuming relative address from base_folder_path
    
    def __init__(self, base_folder_path, csv_file_path, class_ids = None, species_ids = None, transform = None):
        self.base_folder_path = base_folder_path
        self.csv_file_path = csv_file_path
        
        # if sets are not given make loader load all image classes
        if class_ids != None:
            self.class_ids = class_ids
        else:
            self.class_ids = set(range(1, 37))
        
        if species_ids != None:
            self.species_ids = species_ids
        else:
            self.species_ids = {1,2}    
            
        self.transform = transform
        self.sample_list(csv_file_path)
        
    # generate list of samples 
    def sample_list(self, sample_list_file_name):
        
        self.sample_list_file_name = sample_list_file_name
        self.sample_list_file_path = self.base_folder_path + self.sample_list_file_name
            
        self.sample_list = []
        
        with open(self.sample_list_file_path) as csv_sample_list_file:
            csv_reader = csv.reader(csv_sample_list_file, delimiter=' ')
            
            for row in csv_reader:
                
                # extract CSV file information 
                
                image_name = row[0] + ".jpg"
                image_path = self.base_folder_path + "images\\" + image_name
                class_id = int(row[1])
                species_id = int(row[2])
                
                if species_id == 1:
                    species = "cat"
                if species_id == 2:
                    species = "dog"
                
                breed_id = int(row[3])
                split_str = row[0].split('_')
                breed = split_str[0]
                segmentation_mask_path = self.base_folder_path + "annotations\\trimaps\\" + row[0] + ".png"
                
                # only add to sample list the class id and species id which are contained in the class_ids and species_ids sets
                
                if (class_id in self.class_ids) and (species_id in self.species_ids):      
                    # generate dictionary for each sample containing sample information
                    sample = {"image_name":image_name, "image_path":image_path, "class_id":class_id, "species_id":species_id,
                              "species":species, "breed_id":breed_id, "breed":breed, "segmentation_mask_path":segmentation_mask_path}

                    self.sample_list.append(sample)
                
        return self.sample_list
    
    
    # defines length method for custom data loader
    
    def __len__(self):
        return len(self.sample_list)
    
    # defines __getitem__ for custom data loader  
    def __getitem__(self, idx):
        
        image_sample = matplotlib.image.imread(self.sample_list[idx]["image_path"])
        segmentation_mask = matplotlib.image.imread(self.sample_list[idx]["segmentation_mask_path"])  
        
        image_sample = torch.Tensor(image_sample)
        
        
        # permute image to be in the form CxHxW
        image_sample = image_sample.permute((2,0,1))
        
        image_sample_shape = image_sample.shape
        if image_sample_shape[0] > 3:
            print("sample image has 4 dimensions")
            print(self.sample_list[idx]["image_path"])
            print("removing 4th channel")
            image_sample = image_sample[0:3,:,:]

        
        segmentation_mask_shape = segmentation_mask.shape
        if len(segmentation_mask_shape) > 2:
            print("sample mask loaded wrong")
            print(self.sample_list[idx]["segmentation_mask_path"])
        
        # generate mask for each class
        segmentation_mask = torch.Tensor(segmentation_mask)
        
        # matplotlib scales png to 0 to 1, must reverse
        segmentation_mask = segmentation_mask*255
        
        mask_1 = (segmentation_mask  == 1)
        mask_2 = (segmentation_mask  == 2)
        mask_3 = (segmentation_mask  == 3)
        
        mask_1 = mask_1.float()
        mask_2 = mask_2.float()
        mask_3 = mask_3.float()
        
        # stack masks into ClassxHxW tensor
        
        segmentation_mask = torch.stack((mask_1, mask_2, mask_3))
        
        if self.transform:
            image_sample, segmentation_mask = self.transform(image_sample, segmentation_mask)
            
        sample = {'image':image_sample, 'segmentation_mask':segmentation_mask}
        
        # add image information to dictionary containing image and segmentation mask
        sample.update(self.sample_list[idx])
        
        return sample
    
    
    # return an untransformed sample from given index
    
    def getsample(self, idx):
    
        image_sample = matplotlib.image.imread(self.sample_list[idx]["image_path"]) 
        segmentation_mask = matplotlib.image.imread(self.sample_list[idx]["segmentation_mask_path"])  
        
        sample = {'image':image_sample, 'segmentation_mask':segmentation_mask}
        sample.update(self.sample_list[idx])
        
        return sample


if __name__ == "__main__":
    
    PATH = "C:\\Users\\Michael\\Desktop\\Pets_Dataset\\Pets_Dataset\\"
    test_CSV = "annotations\\test.txt"
    train_CSV = "annotations\\trainval.txt"
    
    classes = {1}
    
    target_size = [100,100]
    
    # can use torchvision.Compose(transform_list) to combine additional transforms
    # for both dimensional_transforms and image_transforms
    

    dimensional_transforms = torchvision.transforms.Resize(target_size)
    image_transforms = transforms.scale(255)
    
    # custom transforms class which applies dimensional transforms to image and mask
    # while only appling image transforms to image
    
    transform = transforms.transforms(dimensional_transforms, image_transforms)
    
    train_dataset = PetsDataset(PATH, train_CSV, transform = transform)
    test_dataset = PetsDataset(PATH, test_CSV, transform = transform)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=1)
    
    # note that the DataLoader adds a new first dimension to each dictionary value
    # for each sample in the batch
    
    for batch_num, sample_batch in enumerate(train_loader):
        
        #plot the first image of each batch 
        plt.figure()
        plt.imshow(sample_batch["image"][0].permute((1,2,0)))
        plt.figure()
        plt.imshow(sample_batch["segmentation_mask"][0][0,:,:])
        
        print(sample_batch["breed"][0])


    

    
    
    

