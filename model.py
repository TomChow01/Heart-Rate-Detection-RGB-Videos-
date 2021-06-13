# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 23:18:05 2021

@author: hp
"""
import os
import sys
import numpy as np
#import pandas as pd
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
from torch.optim import lr_scheduler



class SampleCNN(nn.Module):
    def __init__(self):
        super(SampleCNN, self).__init__()

        #self.to_linear = None

        # 59049 x 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU())
        # 19683 x 128
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))

        self.linear_1 = nn.Linear(128*4, 128)
        self.linear_2 = nn.Linear(128, 1)
        #self.activation = nn.Sigmoid()
    
    def forward(self, x):

        out = self.conv1(x)
        #print(out.size())
        out = self.conv2(out)

        
        out = out.view(x.shape[0], out.size(1) * out.size(2))
        out = self.linear_1(out)
        out = self.linear_2(out)

        #logit = self.activation(logit)

        return out