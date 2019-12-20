#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:53:15 2019

@author: Brian Robinson
"""

from __future__ import print_function, division
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import fb_model as model
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import imread

class EnvironDataset(torch.utils.data.Dataset):

    def __init__(self, image_names, root_dir, transform=None):

        """
        Args:
            img_names (string): name of text file with image filenames
            root_dir (string): directory with images and img_names
            transform (callable, optional): Optional transform applied to image
        """
        self.image_names = pd.read_csv(image_names, header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_names.iloc[idx,0])
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image


def find_stats(img_path):

    """Find mean and standard deviation of image data set"""

    images = os.listdir(img_path)
    img = imread(os.path.join(img_path, images[0])).astype(np.float)
    img_sq = np.square(img)
    pix_count = img.shape[0]*img.shape[1]

    for i in images[1:]:
        img_add = imread(os.path.join(img_path, i)).astype(np.float)
        assert img_add.shape == img.shape, "Image shape is inconsistent"
        img_sq += np.square(img_add)
        img += img_add

    img_mean = np.sum(img, axis=(0,1))/(len(images)*pix_count)
    img_mean_sq = np.sum(img_sq, axis=(0,1))/(len(images)*pix_count)
    img_var = img_mean_sq - np.square(img_mean)
    img_std = np.sqrt(img_var)

    return tuple(img_mean), tuple(img_std)


def split_data(dataset, val_split, shuffle_dataset):
    """
    Split data into training and validation sets
    Args:
        dataset (object): EnvironDataset object
        val_split (float): proportion of data to hold out for validation
        shuffle_dataset (Bool): whether to shuffle data before splitting
    """
    dataset_size = dataset.__len__()
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, valid_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    return train_sampler, valid_sampler


def load_data(dataset, train_sampler, valid_sampler, batch_size):
    """Load split datasets"""

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=8)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, num_workers=8)
    return train_loader, validation_loader


class Rescale(object):

    """Rescale the image in a sample to a given size."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        img = img

        return img

class ToTensor(object):
    """ Convert image to C x H x W tensor """

    def __call__(self, image):
        image = image.transpose((2, 0, 1))

        return torch.from_numpy(image)


def cae_loss_fcn(code, img_out, img_in, lamda=1e-4, device=torch.device('cuda')):

    assert img_out.shape == img_in.shape, "img_out.shape : {} != img_in.shape : {}".format(img_out.shape, img_in.shape)

    # First term in the loss function, for ensuring representational fidelity
    criterion=nn.MSELoss()
    loss1 = criterion(img_out, img_in)

    # Second term in the loss function, for enforcing contraction of representation
    code.backward(torch.ones(code.size()).to(device), retain_graph=True)
    # Frobenius norm of Jacobian of code with respect to input image
    loss2 = torch.sqrt(torch.sum(torch.pow(img_in.grad, 2)))
    img_in.grad.data.zero_()

    # Total loss, the sum of the two loss terms, with weight applied to second term
    loss = loss1 + (lamda*loss2)

    return loss

def cae_loss_fcn_2(model, img_in, img_out, h, lamda=1e-4):

    assert img_out.shape == img_in.shape, "img_out.shape : {} != img_in.shape : {}".format(img_out.shape, img_in.shape)

    criterion = nn.MSELoss()
    loss1 = criterion(img_out, img_in)

    W = model.module.fc2.weight.data
    dh = h*(1-h)
    w_sum = torch.sum(Variable(W)**2, dim=1)
    w_sum = w_sum.unsqueeze(1)
    loss2 = torch.sum(torch.mm(dh**2, w_sum), 0)

    return loss1 + loss2.mul_(lamda)

def load_image(image_path, vres, hres):

    composed = transforms.Compose([Rescale((vres,hres)), transforms.ToTensor()])
    image = io.imread(image_path)
    image = composed(image).float()

    return image.cuda()

def save_model(saved_model, optimizer, running_loss_history, val_loss_history, model_path):

    now = datetime.datetime.now()
    PATH = os.path.join(model_path, str('AE_' + now.strftime("%Y-%m-%d-%H-%M")))
    torch.save({
                'model_state_dict' : saved_model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'training_loss_history' : running_loss_history,
                'validation_loss_history' : val_loss_history
                }, PATH)
    print('Model saved successfully')

    return

def load_model(model_path, num_logits):

    loaded_model = model.Autoencoder(num_logits)
    if torch.cuda.device_count() > 1:
        print("Loading model onto", torch.cuda.device_count(), "GPU's")
        loaded_model = torch.nn.DataParallel(loaded_model)
    checkpoint = torch.load(model_path)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.cuda()

    print('Model loaded successfully')

    return loaded_model

def forward_image(model, image_path, output_path, vres, hres):

    img_in = load_image(image_path, vres, hres)
    _, img_out = model(img_in.view(-1,3,vres,hres))
    img_out = img_out.cpu().detach().numpy()
    img_out = np.squeeze(img_out)
    img_out = np.transpose(img_out, (1,2,0))
    fig=plt.figure()
    plt.imshow(img_out)
    fig.savefig(output_path)

    return img_out
