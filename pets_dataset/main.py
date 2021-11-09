# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 21:51:54 2021

@author: Michael
"""

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms 
import torchvision.utils 
import torch.nn as nn
import torch.optim as optim
import torch
import transforms
import model 
import PetsDataset


# define dictionaries to go between species and class index
# the goal of this model is to predict the correct species of animal

species_to_class = {"dog":0, "cat":1}
class_to_species = {0:"dog", 1:"cat"}

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

def main():
    
    PATH = "C:\\Users\\Michael\\Desktop\\Pets_Dataset\\Pets_Dataset\\"
    test_CSV = "annotations\\test.txt"
    train_CSV = "annotations\\trainval.txt"
    
    target_size = [128,128]
    
    # can use torchvision.Compose(transform_list) to combine additional transforms
    # for both dimensional_transforms and image_transforms
    
    dimensional_transforms = torchvision.transforms.Resize(target_size)
    image_transforms = transforms.scale(256)
    
    # custom transforms class which applies dimensional transforms to image and mask
    # while only applying image transforms to image
    
    transform = transforms.transforms(dimensional_transforms, image_transforms)
    
    # create data loaders
    
    train_dataset = PetsDataset.PetsDataset(PATH, train_CSV, transform = transform)
    test_dataset = PetsDataset.PetsDataset(PATH, test_CSV, transform = transform)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=1)
    
    # instantiate model
    
    NN_model = model.NeuralNetwork().to(device)    
    
    # define crossentropy loss which accepts un_normalized input
    
    loss = nn.CrossEntropyLoss()

    # define optimizer
    
    optimizer = optim.Adam(NN_model.parameters())
    
    for epoch in range(10):
    
        print("starting epoch " + str(epoch))
        train(train_loader, NN_model, loss, optimizer)
        print("starting test")
        test(test_loader, NN_model, loss)
    
    
# define training loop
def train(dataloader, model, loss_fn, optimizer):
    
    training_sample_count = len(dataloader.dataset)
    
    model.train()
    for batch, samples in enumerate(dataloader):
        
        images = samples["image"]
        labels = samples["species"]
        
        (batch_size, channel, h, w) = images.shape
        
        # create label tensor with probabilities for loss function 
        # of form batchxclasses
        
        label_tensor = torch.zeros((batch_size,2))
        for index in range(batch_size):
            label_tensor[index, species_to_class[labels[index]]] = 1  
        
        images, label_tensor = images.to(device), label_tensor.to(device)
        
        output = model(images)
        loss = loss_fn(output, label_tensor)
        loss.backward()
        
        optimizer.step()
     
        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size
            print(f"loss: {loss:>7f}, iteration [{current:>5d}/{training_sample_count:>5d}]")
        

def test(dataloader, model, loss_fn):
    
    test_sample_count = len(dataloader.dataset)
    
    model.eval()
    
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        
        for batch, samples in enumerate(dataloader):
            
            images = samples["image"]
            labels = samples["species"]
            
            (batch_size, channel, h, w) = images.shape
            
            # create label tensor with probabilities for loss function 
            # of form batchxclasses
            
            label_tensor = torch.zeros((batch_size,2))
            for index in range(batch_size):
                label_tensor[index, species_to_class[labels[index]]] = 1   
                
            images, label_tensor = images.to(device), label_tensor.to(device)
            
            output = model(images)
            loss = loss_fn(output, label_tensor)
            
            test_loss += loss
            
            class_predictions = torch.argmax(output, dim=1)
            
            for predicted_class, label in zip(class_predictions, labels):
                if class_to_species[int(predicted_class)] == label:
                    correct += 1;
                    
            correct_percentage = 100*(correct/test_sample_count)
        
    print(f"test loss:{test_loss:>7f}, percentage correct is {correct_percentage:>4f}")

    



if __name__ == "__main__":
    main()