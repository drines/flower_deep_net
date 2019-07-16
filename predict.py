#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER:       Daniel Rines - drines(at)gmail(dot)com
# DATE CREATED:     2019.07.12
# REVISED DATE:     2019.07.12
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



def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    img = Image.open(image_path)
    width = img.size[0]
    height = img.size[1]
    
    # scale the image
    resize_dim = 256
    if width > height:
        percent = float(resize_dim) / float(height)
        resize_width = int(width * percent)
        img = img.resize((resize_width, resize_dim), Image.BILINEAR)
    else:
        percent = float(resize_dim) / float(width)
        resize_height = int(height * percent)
        img = img.resize((resize_dim, resize_dim), Image.BILINEAR)
    
    # crop the image object
    crop_size = 224
    left = (img.size[0] - crop_size) / 2
    upper = (img.size[1] - crop_size) / 2
    right = left + crop_size
    lower = upper + crop_size
    img = img.crop((left, upper, right, lower))
    
    # normalize the pixel values 
    # adjust values to be between 0 - 1 instead of 0 - 255
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406]) # mean as provided above with Transform
    std = np.array([0.229, 0.224, 0.225])  # std dev as provided above with Transform
    img = (img - mean) / std  # normalize
    
    # PyTorch expects the color channel to be the first dimension but it's the third dimension
    # moving the third index to the first, and shifting the other two indices
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).type(torch.FloatTensor) 
    
    # return the Pytorch tensor (image)
    return img


# Entry point into program
if __name__ == "__main__":
     # parse in the input arguments
    in_args = get_input_args()

    # let's grab a random image file from the test folder
    flower_num = str(np.random.randint(low=1, high=103))
    image_dir = './flowers/test/' + flower_num + '/'
    flower_list = [x for x in os.listdir(image_dir) if x.endswith('.jpg')]
    rand_flower = np.random.randint(low=0, high=len(flower_list))
    image_file = image_dir + flower_list[rand_flower]

    # run the file through the NN model
    probs, classes = predict(image_file, model)

    # the classes list includes the value, not the key
    # need to swap the key value to match the cat_to_name keys
    idx_to_class = dict((value, key) for key, value in model.class_to_idx.items())

    # convert the class indices into flower names
    names = []
    for index in classes:
        names.append(str(idx_to_class[index]) + " : " + cat_to_name[idx_to_class[index]])
    print(names)