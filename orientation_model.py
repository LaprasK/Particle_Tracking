# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:18:26 2019

@author: zhejun
"""

import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, channel1 = 32, channel2 = 64, channel3 = 64):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, channel1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel1),
            nn.Dropout2d(0.25),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(channel1, channel2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel2),
            nn.Dropout2d(0.25),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(channel2, channel3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel3),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(16*16*64, 1024)
        self.dropout = nn.Dropout(0.25)
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