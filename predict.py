#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER:       Daniel Rines - drines(at)gmail(dot)com
# DATE CREATED:     2019.07.12
# REVISED DATE:     2019.07.18
# PURPOSE:  Program for using a convolutional neural network to predict
#           the identity (class) with an associated likelihood probability.
#
# INPUT:    
#           Command Line Arguments:
#               Required:
#                   1. input:                 <string> Image File
#                   2. Checkpoint             <string> Checkpoint.pth filename
#               Optional:
#                   3. Top K                  --top_k <int>
#                   4. Category Names         --category_names <string>
#                   5. Use GPU:               --gpu
# OUTPUT:   
#           Program outputs the following from the inference determination:
#               Top-K likely identities with catagories
#               Real name for each category
#
#   Example call:
#       Basic usage: 
#           python predict.py /path/to/image checkpoint
#       Options:
#           python predict.py input checkpoint --category_names cat_to_name.json --gpu
##

#
# IMPORT STATEMENTS
#
# for taking command line arguments in parameters
import argparse

# Collections and JSON libraries
import json

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
    
    INPUTS:
        None - using argparse module to store command line arguments
    RETURNS:
        parse_args() -data structure that stores the command line arguments
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description='Infers the identity '+
        'of an object in an image (e.g. flower) using a trained network '+
        'that was previously encapsulated a checkpoint.pth file. Use '+
        'train.py to train the network and output the checkpoint file.')
    # Create the command line arguments
    parser.add_argument('input',
                        type=str,
                        default='',
                        help='Enter an input image filename and location.')
    parser.add_argument('checkpoint',
                        type=str,
                        default='',
                        help='Enter a checkpoint file name and location.')
    parser.add_argument('--top_k',
                        type=int,
                        default=5,
                        help='Top K: (default: 5).')
    parser.add_argument('--category_names',
                        type=str,
                        default='cat_to_name.json',
                        help='Category names: (default: cat_to_name.json).')
    parser.add_argument('--gpu', action='store_true',
                        default=False,
                        dest='gpu',
                        help='GPU switch: (default: Off).')

    # Return the parsed arguments back to the calling function
    return parser.parse_args()


def main(in_args):
    """
    Main function for program execution.
    INPUTS:
        1. Cmd line arguments   <in_args data structure>
    RETURNS:
        None
    """
    # initialize the fcnn model
    network = FCNN()
    model, optimizer = network.load_checkpoint(in_args.gpu, in_args.checkpoint)

    # process the image (in numpy format) for a pytorch inference
    img = network.process_image(in_args.input).unsqueeze(0)

    # run the file through the NN model
    probs, classes = network.predict(img, model, in_args.gpu, topk=in_args.top_k)

    # the classes list includes the value, not the key
    # need to swap the key value to match the cat_to_name keys
    idx_to_class = dict((value, key) for key, value in model.class_to_idx.items())

    # convert the class indices into flower names
    # load and assign image truth values to a dictionary for training and testing
    try:
        f = open(in_args.category_names, 'r')
        network.cat_to_name = json.load(f)
    except Exception as error:
        print("While loading '{}', the following error occurred: {}; check the spelling and file location.".format(in_args.category_names, error))
        return
    else:
        f.close()

    print("The network selected these top: {} identities for image: {}.".format(in_args.top_k, in_args.input))
    idx = 0
    for index in classes:
        print("{0:2d}: {1:} :: {2:3.2f}%".format(idx+1, network.cat_to_name[idx_to_class[index]].capitalize(), probs[idx]*100))
        idx += 1
    
    return


# Entry point into program
if __name__ == "__main__":
     # parse in the input arguments and pass to main
    main(get_input_args())
