#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER:       Daniel Rines - drines(at)gmail(dot)com
# DATE CREATED:     2019.07.10
# REVISED DATE:     2019.07.13
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
# IMPORT STATEMENTS
#
# taking command line arguments in parameters
import argparse

# load object model with associated methods
from fcnn import FCNN


#
# FUNCTION DEFINITIONS
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
    parser = argparse.ArgumentParser(description='Trains a Convolutional '+
             'Neural Network on a provided set of images.')
    # Create the command line arguments as mentioned above
    parser.add_argument('data_dir',
                        type=str,
                        default='flowers',
                        help='Data Dir.: (default: flowers).')
    parser.add_argument('--save_dir',
                        type=str,
                        default='/home/workspace/ImageClassifier',
                        help='Checkpoint Dir.: (default: "/home/workspace/ImageClassifier").')
    parser.add_argument('--arch',
                        type=str,
                        default='vgg16',
                        help='Architecture: (options: "VGG13", "VGG16", "DenseNet121", default: "VGG16").')
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


# Entry point into program
if __name__ == "__main__":
     # parse in the input arguments
    in_args = get_input_args()

    # instantiate the fully-connected neural network
    model = FCNN(in_args)

#    load and assign image truth values to a dictionary for training and testing
    # TODO: trap for exceptions
    with open(label_file, 'r') as f:
        return json.load(f)

    # train the model
    model.train_network()

    # save the network to a checkpoint file
    model.save_checkpoint(in_args.save_dir, 'checkpoint.pth')
