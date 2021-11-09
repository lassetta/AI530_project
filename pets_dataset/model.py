# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 21:51:26 2021

@author: Michael
"""

import torch
import torch.nn as nn
import torch.nn.functional

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 5, 5, padding='same')
        self.avgpool1 = nn.AvgPool2d(4,2, padding=1) 
        self.conv2 = nn.Conv2d(5, 10, 5, padding='same')
        self.avgpool2 = nn.AvgPool2d(4,2, padding=1)
        self.conv3 = nn.Conv2d(10, 20, 5, padding='same')
        self.avgpool3 = nn.AvgPool2d(4,2, padding=1) 
        self.conv4 = nn.Conv2d(20,40, 5, padding='same')
        self.avgpool4 = nn.AvgPool2d(4,2, padding=1) 
        self.conv5 = nn.Conv2d(40,80, 5, padding='same')
        self.avgpool5 = nn.AvgPool2d(4,2, padding=1)
        self.conv6 = nn.Conv2d(80,160, 5, padding='same')
        self.avgpool6 = nn.AvgPool2d(4,2, padding=1)
        self.conv7 = nn.Conv2d(160,1,1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(4,2)
        
        

    def forward(self, image):
        out1 = nn.ReLU()(self.avgpool1(self.conv1(image)))
        out2 = nn.ReLU()(self.avgpool2(self.conv2(out1)))
        out3 = nn.ReLU()(self.avgpool3(self.conv3(out2)))
        out4 = nn.ReLU()(self.avgpool4(self.conv4(out3)))
        out5 = nn.ReLU()(self.avgpool5(self.conv5(out4)))
        out6 = nn.ReLU()(self.avgpool6(self.conv6(out5)))
        out7 = self.conv7(out6)
        out7 = torch.squeeze(out7,1)
        out8 = self.flatten(out7)    
        
        out9 =  self.linear(out8)

        return out9
