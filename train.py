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
#import torch.nn.functional as F

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
                        default='flowers',
                        help='Data Dir.: (default: flowers).')
    parser.add_argument('--save_dir',
                        type=str,
                        default='',
                        help='Checkpoint Dir.: (default: None).')
    parser.add_argument('--arch',
                        type=str,
                        default='vgg16',
                        help='Architecture: (default: "VGG16").')
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
    parser.add_argument('--gpu', action='store_true',
                        default=False,
                        dest='gpu',
                        help='GPU switch: (default: Off).')

    # Return the parsed arguments back to the calling function
    return parser.parse_args()


class FCNN(object):
    """
    Class definition for the fully connected convolutional neural network (CNN)
    training program. The neural network model takes in a number of
    optional parameters at time of instantiation:
    """

    def __init__(self, args):
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
            None
        """
        # default to CPU for device computation
        self.gpu = args.gpu

        # assign the model parameters, instantiate the model object and load 
        # the desired pretrained model into this object
        self.arch = args.arch.lower()
        self.input_size = 25088
        self.hidden_size = args.hidden_units
        self.output_size = 102
        self.dropout_rate = 0.3
        self.learning_rate = args.learning_rate
        self.model, self.criterion, self.optimizer = self.set_network_model()
        
        # initalize the dataset loaders and add the image truth dict to the 
        # model for saving and inference predictions later
        self.train_loader = ''
        self.valid_loader = ''
        self.test_loader = ''
        self.model.class_to_idx = self.set_data_loaders(args.data_dir)
     
        # assign the flower human readable names with associated index numbers
        # to the cat_to_name dictionary
        self.cat_to_name = self.set_label_dict()
        
        # variables for keeping track of training performance
        self.epochs = args.epochs
        self.training_loss = 0
        

    def gpu_status(self):
        """
        Method for checking on the status of a GPU before model device assignment.
        INPUTS:
            1. Command line GPU switch  <bool>
        RETURNS:
            None
        """
        # check if the GPU is currently available and set device flag appropriately
        gpu_status = "cuda:0" if torch.cuda.is_available() else "cpu"

        # if GPU switch was selected during program initiation
        if self.gpu and gpu_status == "cuda:0":
            print("Activating GPU...")
            return torch.device("cuda:0")
        
        # GPU switch not requested of device not available
        else:
            print("GPU not available or not requested, using CPU...")
            self.gpu = False
            return torch.device("cpu")


    def set_network_model(self):
        """
        Loads the pretrained network model.

        INPUTS:
            None
        RETURNS:
            1. model object
            2. criterion
            3. optimizer object
        """
        # load a pre-trained network model based on the command line argument, if supplied
        if self.arch == 'vgg13':
            model = models.vgg13(pretrained=True)
            self.input_size = 25088
        elif self.arch == 'vgg16':
            model = models.vgg16(pretrained=True)
            self.input_size = 25088
        elif self.arch == 'densenet121':
            model = models.densenet121(pretrained=True)
            self.input_size = 1024
        else:
            self.arch = "vgg16"
            model = models.vgg16(pretrained=True)
            self.input_size = 25088
            print("Pretrained model architecture not recoginized or supported. \n"
                  "Using default VGG16 instead. Available architectures: VGG13, \n"
                  "VGG16, and DenseNet121.\n")
        
        # freeze the networks parameters so no backprop occurs
        for param in model.parameters():
            param.requires_grad = False

        # define the feed forward network as a classifier
        # use rectified-linear activation functions and include dropouts (20%)
        # output size based on 102 flower categories specified above
        model.classifier = nn.Sequential(nn.Linear(self.input_size, self.hidden_size),
                                         nn.ReLU(),
                                         nn.Dropout(self.dropout_rate),
                                         nn.Linear(self.hidden_size, self.output_size),
                                         nn.LogSoftmax(dim=1))

        # prepare to train the model using NLLLoss, Adam - for momentum & a learning rate of 0.001
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=self.learning_rate)
        
        return model, criterion, optimizer
   

    def set_data_loaders(self, data_dir, train_dir='/train', valid_dir='/valid', test_dir='/test', batch_size=128):
        """
        Initialize and define the different transforms along with the associate
        data loaders.
        INPUTS:
            1. train_dir    <str>
            2. valid_dir    <str>
            3. test_dir     <str>
            4. batch_size   <int>
        RETURNS:
            1. class_to_idx <dict>
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
        train_dataset = datasets.ImageFolder(data_dir + train_dir, transform=train_transforms)
        valid_dataset = datasets.ImageFolder(data_dir + valid_dir, transform=test_transforms)
        test_dataset  = datasets.ImageFolder(data_dir + test_dir, transform=test_transforms)

        # Load data with DataLoader definitions for each image datasets using the correct trainform
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
        self.test_loader =  torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_dataset.class_to_idx
    

    def set_label_dict(self, label_file='cat_to_name.json'):
        """
        Loads in a mapping JSON file for associating category label to category name.
        This creates a dictionary which maps the integer encoded categories to the actual names of the flowers.
        INPUTS:
            1. label_file   <str>
        RETURNS:
            1. cat_to_name  <dict>
        """
        # load and assign image truth values to a dictionary for training and testing
        # TODO: trap for exceptions
        with open(label_file, 'r') as f:
            return json.load(f)


    def train_network(self, top_k=1):
        """
        Method for peforming training of the network's features only.
        INPUTS:
            None
        RETURNS:
            None
        """
        # now for the training and validation process
        print(f"Training the model with the following parameters:\n"
              f"\tmodel: \t\t\t{self.arch}\n"
              f"\thidden units: \t\t{self.hidden_size}\n"
              f"\tlearning rate: \t\t{self.learning_rate}\n"
              f"\tepochs: \t\t{self.epochs}\n"
              f"\tGPU processing: \t{self.gpu}\n"
             ) 

        # push model to correct processor
        device = self.gpu_status()
        self.model.to(device)

        # loop thru based on total epochs desired
        for epoch in range(self.epochs):
            
            # reset loss and accuracy counters for each new epoch
            self.training_loss = 0
            valid_loss = 0
            valid_accuracy = 0
            
            # train the model with the training data 
            self.model.train()
            for inputs, labels in self.train_loader:
                
                # forward pass with inputs and labels to correct environment (GPU vs CPU)
                inputs, labels = inputs.to(device), labels.to(device)

                # turn off gradient
                self.optimizer.zero_grad()
                
                # use network outputs to calculate loss
                log_ps = self.model.forward(inputs)
                loss = self.criterion(log_ps, labels)

                # perform backward pass to calc gradients and take step to update weights
                loss.backward()
                self.optimizer.step()
                
                self.training_loss += loss.item()
            
            else:
                # evaluate the model with the validation data
                # turn off gradient calc for validation
                self.model.eval()
                with torch.no_grad():

                    # test the model with the validation data
                    for inputs, labels in self.valid_loader:

                        # send inputs and labels to correct environment (GPU vs CPU)
                        inputs, labels = inputs.to(device), labels.to(device)

                        # perform a forward pass & get loss rate for batch
                        logps = self.model.forward(inputs)
                        batch_loss = self.criterion(logps, labels)

                        # accumulate validation total loss from current batch
                        valid_loss += batch_loss.item()

                        # Calculate accuracy values
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(top_k, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # output the results as we go to watch for productive progress
                print(f"Epoch {epoch + 1} / {self.epochs}.. "
                      f"Train loss: {self.training_loss / len(self.train_loader):.3f}.. "
                      f"Valid loss: {valid_loss / len(self.valid_loader):.3f}.. "
                      f"Valid accuracy: {valid_accuracy / len(self.valid_loader):.2f}")


    def save_network(self, save_dir='.', checkpoint_file='checkpoint.pth'):
        """
        Method for saving the neural network to a checkpoint file so it can be
        reloaded again without the need to re-train the network.
        INPUTS:
            1. Checkpoint file name     <str>
        RETURNS:
            None
        """
        # define the checkpoint dict for saving, loading and inference later
        checkpoint = {'input_size' : self.input_size,
                      'hidden_size' : self.hidden_size,
                      'output_size' : self.output_size,
                      'classifier' : self.model.classifier,
                      'criterion' : self.criterion,
                      'learning_rate' : self.learning_rate,
                      'epochs' : self.epochs,
                      'loss' : self.training_loss,
                      'class_to_idx' : self.model.class_to_idx,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict()}

        # save the model to the specified folder and file name
        torch.save(checkpoint, save_dir + "/" + checkpoint_file)


# Entry point into program
if __name__ == "__main__":
     # parse in the input arguments
    in_args = get_input_args()

    # instantiate the fully-connected neural network
    model = FCNN(in_args)

    # train the model
    model.train_network()

    # save the network to a checkpoint file
    if in_args.save_dir != '':
        model.save_network(in_args.save_dir, 'checkpoint.pth')
        print("Trained model saved to: {}".format(in_args.save_dir + '/checkpoint.pth'))
    else:
        print("No model file name argument supplied, no checkpoint.pth file created at this time.")