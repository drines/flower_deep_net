#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER:       Daniel Rines - drines(at)gmail(dot)com
# DATE CREATED:     2019.07.12
# REVISED DATE:     2019.07.17
# PURPOSE:  Program for using a convolutional neural network to predict.
#
# INPUT:    
#           Command Line Arguments:
#               Required:
#                   1. Image File             <string>
#                   2. Checkpoint             <string>
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
    parser.add_argument('input',
                        type=str,
                        default='./flowers/test/1/image_06743.jpg',
                        help='Image file: (default: flowers/test/1/image_06743.jpg).')
    parser.add_argument('checkpoint',
                        type=str,
                        default='/home/workspace/ImageClassifier/checkpoint.pth',
                        help='Checkpoint file: (default: "/home/workspace/ImageClassifier/checkpoint.pth").')
    parser.add_argument('--top_k',
                        type=int,
                        default=5,
                        help='Top K: (default: 5).')
    parser.add_argument('--category_names',
                        type=str,
                        default='cat_to_names.json',
                        help='Category names: (default: cat_to_names.json).')
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

    # # let's grab a random image file from the test folder
    # flower_num = str(np.random.randint(low=1, high=103))
    # image_dir = './flowers/test/' + flower_num + '/'
    # flower_list = [x for x in os.listdir(image_dir) if x.endswith('.jpg')]
    # rand_flower = np.random.randint(low=0, high=len(flower_list))
    # image_file = image_dir + flower_list[rand_flower]

    # initialize the fcnn model
    network = FCNN()
    model, optimizer = network.load_checkpoint(in_args.checkpoint)

    # process the image (in numpy format) for a pytorch inference
    img = network.process_image(in_args.input).unsqueeze(0)

    # run the file through the NN model
    probs, classes = network.predict(img, model, topk=in_args.top_k)

    # the classes list includes the value, not the key
    # need to swap the key value to match the cat_to_name keys
    idx_to_class = dict((value, key) for key, value in model.class_to_idx.items())

    # convert the class indices into flower names
    # load and assign image truth values to a dictionary for training and testing
    try:
        with open(in_args.category_names, 'r') as f:
            network.cat_to_name = json.load(f)
    except ValueError:
        print("There was an error loading {}.".format(in_args.category_names))

    names = []
    for index in classes:
        names.append(str(idx_to_class[index]) + " : " + network.cat_to_name[idx_to_class[index]])
    print(names)