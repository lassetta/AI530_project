# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 15:10:34 2021

@author: Michael
"""

import numpy as np
import matplotlib
from matplotlib import image
import os
import csv 
from PIL import Image



# class for operations with pets dataset, give base folder path
# which contains two sub folders, \images\ and \annotations\.
# \images\ contains images and \annotations\ contains image information
# including test.txt and trainval.txt which list the training and testing sets
# segmentation information is provided in \trimaps\ subfolder and xmls
# contains face boxes of pets 

class PetsDataset:
    
    # class to hold info for each picture 
    class sample:    
        def __init__(self, image_path, class_id, species_id, breed_id, segmentation_path):
            
            self.image_path = file_path
            self.image_name = os.path.basename(image_path)
            self.class_id = class_id
            self.species_id = species_id
            self.breed_id = breed_id
            
            parsed_image_name = self.image_name.split('_')
            self.breed = parsed_image_name[0]
            self.segmentation_path = segmentation_path
    
    # set base folder path, of pets_dataset and generate sample list from csv_file_path
    # assuming relative address from base_folder_path
    def __init__(self, base_folder_path, csv_file_path):
        self.base_folder_path = base_folder_path
        self.csv_file_path = csv_file_path
        self.sample_list(csv_file_path)
        
        
    # 
    def sample_list(self, sample_list_file_name):
        
        self.entry_file = entry_file_name
        self.entry_file_path = self.base_folder_path + self.entry_file
            
        entry_list = []
        
        with open(self.entry_file_path) as csv_train_file:
            csv_reader = csv.reader(csv_train_file)
            
            for row in csv_reader:
                
                image_name = row[0] + ".jpg"
                image_path = self.base_folder_path + "\\images\\" + image_name
                class_id = row[1]
                species_id = row[2]
                breed_id = row[3]
                
                trimap_path = self.base_folder_path + "\\annotations\\trimaps\\" + image_name
                new_entry = entry(image_path, class_id, species_id, breed_id,trimap_path)
                entry_list.append(new_entry)
                
        
        return entry_list
    
    
    # defines length method for custom data loader
    
    def __len__(self):
        return len(self.entry_list)
    
    # defines __getitem__ for custom data loader
    
    def __getitem__(self, idx):
        
        image = image.imread(self.entry_list[idx].image_path)
        segmentation_mask = image.imread(self.entry_list[idx].segmetnation_path)  
        sample = {'image':image, 'image_info': self.entry_list[idx], 'segmentation_mask':segmentation_mask}
        
        return sample




    

    

C
    
    
        
        
    
        
    
        
        
        
        
        


PATH = "C:\\Users\\Michael\\Desktop\\Pets_Dataset\\Pets_Dataset\\"

images = "images\\"
annotations = "annotations\\"

test_CSV = "test.txt"
train_CSV = "trainval.txt"

train_csv_file = open(PATH+train_CSV)
trasin_csv_reader = csv.reader(csv_file)

train_list =  []

# generate list of dictionary consisting of information for each file
for row in csv_reader:
    filename = row[0]
    class_id = int(row[1])
    species = int(row[2])
    breed = int(row[3])
    
    entry = {"filename" : filename,
             "class" : class_id,
             "species" : species,
             "breed" : breed}
    
    train_list.append(entry)
    

 
    



first_file = 


imageread = cv2.imread(PATH + images + "\\Abyssinian_1.jpg")
#displaying the image as the output on the screen
cv2.imshow('Display_image', imageread)
np_image = np.array(imageread)
cv2.imshow("np image", np_image)


cv2.waitKey(0)

