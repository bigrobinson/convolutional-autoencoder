#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:16:00 2019

@author: Brian Robinson
"""

import torch
import utils
import model
from torchvision import transforms
import os
import argparse

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=500, help="number of epochs to train")
    parser.add_argument("--custom_loss", type=bool, default=False, help="Use custom loss?")
    parser.add_argument("--batch_size", type = int, default=4, help="batch size")
    parser.add_argument("--val_split", type=float, default=0.2, help="ratio of training data to use for validation and testing")
    parser.add_argument("--num_logits", type=int, default=16, help="number of units in autoencoder bottleneck/code layer")
    parser.add_argument("--img_dir", type=str, default='./train_data/flappy_bird', help="training data root directory")
    parser.add_argument("--img_names", type=str, default='./train_data/flappy_bird_images.txt', help="text file of image names")
    parser.add_argument("--model_dir", type=str, default='./saved_models', help="saved models directory")
    parser.add_argument("--model_name", type=str, default='AE_2019-12-11-23-25')
    parser.add_argument("--resume", type=bool, default=False, help="Resume training from saved model")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD optimizer")
    parser.add_argument("--SGD", type=bool, default=False, help="Use SGD optimizer?")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam")
    parser.add_argument("--lr0", type=float, default=0.01, help="base learning rate for SGD")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay for Adam or SGD")
    parser.add_argument("--nesterov", type=bool, default=True, help="Use Nesterov momentum with SGD?")
    parser.add_argument("--burn_in", type=int, default=1, help="number of epochs for burn-in with SGD")
    parser.add_argument("--burn_in_scale", type=float, default=1e-2, help="scale factor by which to reduce lr for burn-in with SGD")
    parser.add_argument("--steps", type=int, default=33, help="number of epochs after which to reduce lr with SGD")
    parser.add_argument("--scales", type=float, default=0.1, help="scale factor by which to reduce lr with SGD")
    parser.add_argument("--lamda", type=float, default=1e-4, help="weight of regularizer in custom loss")

    opt = parser.parse_args()

    return opt

def train(opt):

    # Get training data statistics
    img_mean, img_std = utils.find_stats(opt.img_dir)
    print('\nMean RGB pixel values: {:.4f}, {:.4f}, {:.4f}'.format(*img_mean))
    print('Standard deviation of RGB pixel values: {:.4f}, {:.4f}, {:.4f}\n'.format(*img_std))

    # Model inputs are 512 x 288 to match Flappy Bird screen
    composed = transforms.Compose([utils.Rescale((512,288)), utils.ToTensor()])

    dataset = utils.EnvironDataset(opt.img_names, opt.img_dir, transform = composed)
    train_sampler, valid_sampler = utils.split_data(dataset, opt.val_split, shuffle_dataset=True)
    train_loader, validation_loader = utils.load_data(dataset, train_sampler, valid_sampler, opt.batch_size)

    # Check for multiple GPU's
    device = torch.device("cuda")

    # Instantiate model
    if opt.resume:
        model_path = os.path.join(opt.model_dir, opt.model_name)
        model = utils.load_model(model_path, opt.num_logits)
    else:
        model = model.Autoencoder(opt.num_logits)
        if torch.cuda.device_count() > 1:
            print("Training with", torch.cuda.device_count(), "GPU's")
            model = torch.nn.DataParallel(model)
        # Put model on GPU's
        model.to(device)

    if not opt.SGD:
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    if not opt.custom_loss:
        # Builtin loss
        criterion = torch.nn.MSELoss()

    running_loss_history = []
    val_loss_history = []

    for epoch in range(opt.num_epochs):
        running_loss = 0.0
        val_running_loss = 0.0

        if opt.SGD:
            if epoch < opt.burn_in:
                lr = opt.burn_in_scale*opt.lr0
                optimizer = torch.optim.SGD(model.parameters(),
                                            lr=lr,
                                            momentum=opt.momentum,
                                            weight_decay=opt.weight_decay,
                                            nesterov=opt.nesterov)
                lr = opt.lr0
            elif epoch % steps == 0:
                lr = opt.scales*lr
                optimizer = torch.optim.SGD(model.parameters(),
                                            lr=lr,
                                            momentum=opt.momentum,
                                            weight_decay=opt.weight_decay,
                                            nesterov=opt.nesterov)

        for data in train_loader:

            img_in = data.to(device)
            img_in = img_in.float()

            # forward loss calculation
            code, img_out = model(img_in)
            loss = criterion(img_out, img_in)

            if opt.custom_loss:
                # Custom loss
                loss = utils.cae_loss_fcn(code, img_in, img_out, lamda=opt.lamda)

            # back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # running loss
            running_loss += loss.item()

        else:
            with torch.no_grad():
                for data in validation_loader:
                    val_img_in = data.to(device)
                    val_img_in = val_img_in.float()
                    val_code, val_img_out = model(val_img_in)

                    if not opt.custom_loss:
                        val_loss = criterion(val_img_out, val_img_in)
                    else:
                        val_loss = utils.cae_loss_fcn_2(model, val_img_in, val_img_out, val_code, lamda=opt.lamda)

                    val_running_loss += val_loss.item()

            epoch_loss = running_loss/len(train_loader)
            running_loss_history.append(epoch_loss)
            val_epoch_loss = val_running_loss/len(validation_loader)
            val_loss_history.append(val_epoch_loss)

            print('\nepoch:', epoch)
            print('training loss: {:.6f}'.format(epoch_loss))
            print('validation loss: {:.6f}'.format(val_epoch_loss))

        if (epoch+1)%100==0:
            utils.save_model(model, optimizer, running_loss_history, val_loss_history, "/home/brian/IRAD/CogAIRAD/saved_models")

if __name__ == "__main__":

    opt=get_args()
    train(opt)
