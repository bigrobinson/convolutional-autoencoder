#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:45:41 2019

@author: Brian Robinson
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):

    def __init__(self, num_logits):
        super(Autoencoder, self).__init__()
        self.n_logits = num_logits

        # Encoder methods
        self.conv1 = nn.Conv2d( 3, 6,  kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d( 6, 8,  kernel_size=7, padding=3)
        self.conv3 = nn.Conv2d( 8, 12, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(12, 16, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(16, 18, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(9*16*18, 256)
        self.fc2 = nn.Linear(256, self.n_logits)

        # Decoder methods
        self.fc3 = nn.Linear(self.n_logits, 256)
        self.fc4 = nn.Linear(256, 9*16*18)
        self.deconv1 = nn.ConvTranspose2d(18, 16, kernel_size=2, stride=2, output_padding=(0,0))
        self.deconv2 = nn.ConvTranspose2d(16, 12, kernel_size=2, stride=2, output_padding=(0,0))
        self.deconv3 = nn.ConvTranspose2d(12,  8, kernel_size=2, stride=2, output_padding=(0,0))
        self.deconv4 = nn.ConvTranspose2d( 8,  6, kernel_size=2, stride=2, output_padding=(0,0))
        self.deconv5 = nn.ConvTranspose2d( 6,  3, kernel_size=2, stride=2, output_padding=(0,0))

        # Common methods
        self.pool  = nn.MaxPool2d(2,2)

    def forward(self, x):

        x = self.pool(F.leaky_relu(self.conv1(x), negative_slope=0.01))
        x = self.pool(F.leaky_relu(self.conv2(x), negative_slope=0.01))
        x = self.pool(F.leaky_relu(self.conv3(x), negative_slope=0.01))
        x = self.pool(F.leaky_relu(self.conv4(x), negative_slope=0.01))
        x = self.pool(F.leaky_relu(self.conv5(x), negative_slope=0.01))
        x = x.view(-1, 9*16*18)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        code = F.leaky_relu(self.fc2(x), negative_slope=0.01)

        x = F.leaky_relu(self.fc3(code), negative_slope=0.01)
        x = F.leaky_relu(self.fc4(x), negative_slope=0.01)
        x = x.view(-1, 18, 16, 9)
        x = F.leaky_relu(self.deconv1(x), negative_slope=0.01)
        x = F.leaky_relu(self.deconv2(x), negative_slope=0.01)
        x = F.leaky_relu(self.deconv3(x), negative_slope=0.01)
        x = F.leaky_relu(self.deconv4(x), negative_slope=0.01)
        img_out = torch.sigmoid(self.deconv5(x))

        return code, img_out
