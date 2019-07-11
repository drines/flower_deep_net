#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER:       Daniel Rines - drines(at)gmail(dot)com
# DATE CREATED:     2019.07.10
# REVISED DATE:     2019.07.10
# PURPOSE:  Program for training a convolutional neural network.
#
# INPUT:    
#           Command Line Arguments:
#               Required:
#                   1. Data Directory         <string>
#               Optional:
#                   2. Checkpoint Directory   --save_dir <string>
#                   3. NN Architecture        --arch <string>
#                   4. Learning Rate:         --learning_rate <float>
#                   5. Hidden Units:          --hidden_units <int>
#                   6. Epochs:                --epochs <int>
#                   7. Use GPU:               --gpu
# OUTPUT:   
#           Program outputs the following during training:
#               training loss
#               validation loss
#               Validation accuracy
#
#   Example call:
#       Basic usage: 
#           python train.py data_directory
#       Options:
#           python train.py --save_dir <dir> --arch <'vgg16'> --learning_rate <value>
##

#
# Import libraries
#

# for taking command line arguments in parameters
import argparse

# Torch library imports
from torch import torch, nn, optim
import torch.nn.functional as F

# Torchvision imports for loading transforms, models and images
from torchvision import transforms, datasets, models

# Collections and JSON libraries
import json
from collections import OrderedDict

# Numpy library
import numpy as np


#
# Function definitions
#

def get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. If the user fails to provide 
    some or all of the arguments, then the default values are used for the 
    missing arguments. 

    Available Command Line Arguments:
      Required:
        1. Data Directory         data_dir <string>
      Optional:
        2. Checkpoint Directory   --save_dir <string>
        3. NN Architecture        --arch <string>
        4. Learning Rate:         --learning_rate <float>
        5. Hidden Units:          --hidden_units <int>
        6. Epochs:                --epochs <int>
        7. Use GPU:               --gpu
    
    Parameters:
        None - using argparse module to store command line arguments
    Returns:
        parse_args() -data structure that stores the command line arguments
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description='Trains a Convolutional'+
             'Neural Network on a provided set of images.')
    # Create the command line arguments as mentioned above
    parser.add_argument('data_dir',
                        type=str,
                        default='.',
                        help='Data Dir.: (default: .).')
    parser.add_argument('--save_dir',
                        type=str,
                        default='.',
                        help='Checkpoint Dir.: (default: .).')
    parser.add_argument('--arch',
                        type=str,
                        default='vgg16',
                        help='Architecture: (default: vgg16).')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.0001,
                        help='Learning Rate: (default: 0.0001).')
    parser.add_argument('--hidden_units',
                        type=int,
                        default=2048,
                        help='Hidden Units: (default: 2048).')
    parser.add_argument('--epochs',
                        type=int,
                        default=6,
                        help='Epochs: (default: 6).')
    parser.add_argument('--gpu',
                        type=bool,
                        default=False,
                        help='GPU: (default: False).')

    # Return the parsed arguments back to the calling function
    return parser.parse_args()


class CNN(object):
    """
    Class definition for the convolutional neural network (CNN)
    training program. The neural network model takes in a number of
    optional parameters at time of instantiation:
    """

    def __init__(self, arch='vgg16', inputs=25088, hiddens=2048, outputs=102, drop=0.3, lr=0.0001, image_root='flowers'):
        """
        Initialization and class related variables.
        INPUTS:
            1. Model Architecture:      <string>
            2. Input size:              <int>
            3. Hidden size:             <int>
            4. Output size:             <int>
            5. Dropout rate:            <float>
            6. Learning rate:           <float>
        RETURNS:
        """
        self.arch = arch
        self.input_size = inputs
        self.hidden_size = hiddens
        self.output_size = outputs
        self.dropout_rate = drop
        self.learning_rate = lr
        self.data_root = image_root
        self.train_loader = ''
        self.valid_loader = ''
        self.test_loader = ''
        self.cat_to_name = {}
    

    def set_data_loaders(self, train_dir='/train', valid_dir='/valid', test_dir='/test', batch_size=128):
        """
        Initialize and define the different transforms along with the associate
        data loaders.
        INPUTS:
            1. train_dir    <str>
            2. valid_dir    <str>
            3. test_dir     <str>
            4. batch_size   <int>
        RETURNS:
            None
        """
        # resize the images to square (255 pixels) prior to cropping the images to 224x224 pixels
        # also perform data augmentation with randomization (rotation, resize, flipping)
        # normalization is performed on all sets of images
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.RandomVerticalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

        # for validation and testing, no random augmentation is applied, just resizing, cropping 
        # and normalization
        test_transforms =  transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

        # Pass dataset locations to apprioriate ImageFolder definitions
        train_dataset = datasets.ImageFolder(self.data_root + train_dir, transform=train_transforms)
        valid_dataset = datasets.ImageFolder(self.data_root + valid_dir, transform=test_transforms)
        test_dataset  = datasets.ImageFolder(self.data_root + test_dir, transform=test_transforms)

        # Load data with DataLoader definitions for each image datasets using the correct trainform
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
        self.test_loader =  torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    

    def set_label_dict(self, label_file='cat_to_name.json'):
        """
        Loads in a mapping JSON file for associating category label to category name.
        This creates a dictionary which maps the integer encoded categories to the actual names of the flowers.
        INPUTS:
            1. label_file   <str>
        OUTPUTS:
            None
        """
        # load and assign image truth values to a dictionary for training and testing
        with open(label_file, 'r') as f:
            self.cat_to_name = json.load(f)





def main():
    """
    The main function.
    PARAMETERS: None
    RETURNS:    None
    """
    # parse in the input arguments
    in_args = get_input_args()


# Call to main function to run the program
if __name__ == "__main__":
    main()