# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:18:26 2019

@author: zhejun
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class ConvNet(nn.Module):
    def __init__(self, channel1 = 64, channel2 = 64, channel3 = 64):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, channel1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel1),
            nn.Dropout2d(0.5),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(channel1, channel2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel2),
            nn.Dropout2d(0.5),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(channel2, channel3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel3),
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(16*16*64, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 360)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

    
class ClassNet(nn.Module):
    def __init__(self):
        super(ClassNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(16*16*64, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    
    
def process_data(file_name):
    dataset = np.load(file_name).item()
    y = np.array(dataset['training_regress'])
    x = np.array(dataset['training_x'])
    c = np.array(dataset['training_class'])
    
    y, x, c = shuffle(y, x, c)
    print(len(x))
    
    mask = np.isnan(y)
    if len(x.shape) == 3:
        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis = 1)
        c = np.expand_dims(c, axis = 1)
    mean = np.mean(x, axis=(0,2,3),keepdims = True)
    print(mean)
    std = np.std(x, axis=(0,2,3), keepdims = True)
    print(std)
    x = (x - mean)/std

    test_index = int(len(x) * 0.9)
    test_x, test_y, test_c, test_mask = x[test_index:], y[test_index:], c[test_index:], mask[test_index:]
            
    class_dict = dict()
    class_dict['x'] = torch.Tensor(x)
    class_dict['y'] = torch.LongTensor(c)
    
    x, y, c, mask = x[:test_index], y[:test_index], c[:test_index], mask[:test_index]
    reg_dict = dict()
    reg_x = x[~mask]
    reg_y = y[~mask]
    reg_y = np.array(np.rad2deg(reg_y)).astype(int)
    print(reg_y.shape)
    reg_dict['x'] = torch.Tensor(reg_x)
    reg_dict['y'] = torch.LongTensor(reg_y)
    
    test_dict = dict()
    test_dict['x'] = torch.Tensor(test_x[~test_mask])
    test_y = np.array(np.rad2deg(test_y[~test_mask])).astype(int)
    test_dict['y'] = torch.LongTensor(test_y)
    return reg_dict, class_dict, test_dict
