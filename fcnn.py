#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER:       Daniel Rines - drines(at)gmail(dot)com
# DATE CREATED:     2019.07.13
# REVISED DATE:     2019.07.17
# PURPOSE:  Object model and associated methods for the fully connected
#           neural network that is used by train.py and predict.py.
#

#
# IMPORT STATEMENTS
#
# taking command line arguments in parameters
import argparse

# Torch library imports
from torch import torch, nn, optim
#import torch.nn.functional as F

# Torchvision imports for loading transforms, models and images
from torchvision import transforms, datasets, models

# Pillow library
from PIL import Image

# Numpy library
import numpy as np


#
# CLASS DEFINITIONS
#
class FCNN(object):
    """
    Class definition for the fully connected convolutional neural network (CNN)
    training program. The neural network model takes in a number of
    optional parameters at time of instantiation:
    """

    def __init__(self, data_dir='./'):
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
        self.gpu = 'cpu'

        # assign the model parameters, instantiate the model object and load 
        # the desired pretrained model into this object
        self.arch = ''
        self.input_size = 0
        self.hidden_size = 0
        self.output_size = 0
        self.dropout_rate = 0.0
        self.learning_rate = 0.0
        self.batch_size = 128
        
        # initalize the dataset loaders and add the image truth dict to the 
        # model for saving and inference predictions later
        self.train_loader = ''
        self.valid_loader = ''
        self.test_loader = ''
        self.class_to_idx = self.set_data_loaders(data_dir)
     
        # assign the flower to human readable names with associated index numbers
        # to the cat_to_name dictionary
        self.cat_to_name = {}
        
        # variables for keeping track of training performance
        self.epochs = 1 # args.epochs
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


    def set_data_loaders(self, data_dir, 
                         train_dir='/train', 
                         valid_dir='/valid', 
                         test_dir='/test'):
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
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size)
        self.test_loader =  torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        return train_dataset.class_to_idx


    def set_network_model(self, arch='vgg16',
                          learning_rate=0.0001, 
                          hidden_size=2048,
                          dropout_rate=0.3,
                          ):
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
        if arch.lower() == 'vgg13':
            self.arch = arch
            model = models.vgg13(pretrained=True)
            self.input_size = 25088
        elif arch.lower() == 'vgg16':
            self.arch = arch.lower()
            model = models.vgg16(pretrained=True)
            self.input_size = 25088
        elif arch.lower() == 'densenet121':
            self.arch = arch.lower()
            model = models.densenet121(pretrained=True)
            self.input_size = 1024
        else:
            self.arch = "vgg16"
            model = models.vgg16(pretrained=True)
            self.input_size = 25088
            print("Pretrained model architecture not recoginized or supported. \n"
                  "Using default VGG16 instead. Available architectures: VGG13, \n"
                  "VGG16, and DenseNet121.\n")
        
        # update other training parameters
        self.output_size = 102
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

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
   

    def train_network(self, model, optimizer, criterion, epochs=1, top_k=1):
        """
        Method for peforming training of the network's features only.
        INPUTS:
            None
        RETURNS:
            None
        """
        # update the requested epochs value
        self.epochs = epochs

        # now for the training and validation process
        print(f"Training the model with the following parameters:\n"
              f"\tmodel archit.: \t{self.arch}\n"
              f"\thidden units: \t{self.hidden_size}\n"
              f"\tlearning rate: \t{self.learning_rate}\n"
              f"\ttotal epochs: \t{self.epochs}\n"
              f"\tGPU processing: \t{self.gpu}\n"
             ) 

        # push model to correct processor
        device = self.gpu_status()
        model.to(device)

        print("Starting epoch: 1 of {}...\n".format(self.epochs))
        
        # loop thru based on total epochs desired
        for epoch in range(self.epochs):
            
            # reset loss and accuracy counters for each new epoch
            self.training_loss = 0
            valid_loss = 0
            valid_accuracy = 0
            
            # train the model with the training data 
            model.train()
            for inputs, labels in self.train_loader:
                
                # forward pass with inputs and labels to correct environment (GPU vs CPU)
                inputs, labels = inputs.to(device), labels.to(device)

                # turn off gradient
                optimizer.zero_grad()
                
                # use network outputs to calculate loss
                log_ps = model.forward(inputs)
                loss = criterion(log_ps, labels)

                # perform backward pass to calc gradients and take step to update weights
                loss.backward()
                optimizer.step()
                self.training_loss += loss.item()
            
            else:
                # evaluate the model with the validation data
                # turn off gradient calc for validation
                model.eval()
                with torch.no_grad():

                    # test the model with the validation data
                    for inputs, labels in self.valid_loader:

                        # send inputs and labels to correct environment (GPU vs CPU)
                        inputs, labels = inputs.to(device), labels.to(device)

                        # perform a forward pass & get loss rate for batch
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        # accumulate validation total loss from current batch
                        valid_loss += batch_loss.item()

                        # Calculate accuracy values
                        ps = torch.exp(logps)
                        _, top_class = ps.topk(top_k, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # output the results as we go to watch for productive progress
                print(f"Epoch {epoch + 1} / {self.epochs}.. "
                      f"Train loss: {self.training_loss / len(self.train_loader):.3f}.. "
                      f"Valid loss: {valid_loss / len(self.valid_loader):.3f}.. "
                      f"Valid accuracy: {valid_accuracy / len(self.valid_loader):.2f}")


    def save_checkpoint(self, model, optimizer, 
                        save_dir='/home/workspace/ImageClassifier', 
                        checkpoint_file='checkpoint.pth'):
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
                      'classifier' : model.classifier,
                      'learning_rate' : self.learning_rate,
                      'epochs' : self.epochs,
                      'loss' : self.training_loss,
                      'class_to_idx' : self.class_to_idx,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}

        # save the model to the specified folder and file name
        torch.save(checkpoint, save_dir + "/" + checkpoint_file)
        print("Trained model saved to: {}".format(save_dir + '/checkpoint.pth'))


    # Method that loads a checkpoint and rebuilds the model
    def load_checkpoint(self, checkpoint_file):
        """
        Method to load a checkpoint file and reassign required variables.
        INPUT:
            1. Checkpoint file:     <string>
        OUTPUT:
            1. Model                <object>
            2. Optimizer            <object>
        """
        # check if the GPU is currently available and set device flag appropriately
        dev_location = "cuda:0" if torch.cuda.is_available() else "cpu"

        # load the old model state
        checkpoint = torch.load(checkpoint_file, map_location=dev_location)
        model = models.vgg16(pretrained=True)
        
        # freeze the networks parameters so no backprop occurs
        for param in model.parameters():
            param.requires_grad = False

        # in case more training is desired, assign needed values
        self.epoch = checkpoint['epoch']
        self.training_loss = checkpoint['loss']
        model.classifier = checkpoint['classifier']
        #criterion = checkpoint['criterion']

        # prepare to train the model using NLLLoss, Adam - for momentum & a learning rate of 0.001
        optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
        
        # ressign the state dictionaries and label indices
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
            
        return model, optimizer


    def process_image(self, image_path):
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
            img = img.resize((resize_dim, resize_height), Image.BILINEAR)
        
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


    def predict(self, img, model, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        # Predict the class from an image file
        
        # push model to correct processor
        device = self.gpu_status()
        model.to(device)

        # turn off gradient calc for testing
        model.eval()
        with torch.no_grad():

            # send image to correct environment (GPU vs CPU) and
            # perform a forward pass
            logps = model.forward(img.to(device))
            
            # get the log softmax values and convert to probabilities
            probs = torch.exp(logps)
        
        # convert to the top (k) labels
        probs, classes = probs.topk(topk)

        return probs.tolist()[0], classes.tolist()[0]