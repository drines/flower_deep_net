#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER:       Daniel Rines - drines(at)gmail(dot)com
# DATE CREATED:     2019.07.13
# REVISED DATE:     2019.07.18
# PURPOSE:  Object model and associated methods for creating the fully connected
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
    training program.
    """

    def __init__(self, data_dir='./'):
        """
        Initialization of class related variables.
        INPUTS:
            1. Data directory: <string>
        RETURNS:
            None
        """
        # assign the model parameters, instantiate the model object and load 
        # assume cpu processing gpu selected by command line argument
        self.device_location = 'cpu'

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
        self.class_to_idx = {}
     
        # assign the flower to human readable names with associated index numbers
        # to the cat_to_name dictionary
        self.cat_to_name = {}
        
        # variables for keeping track of training performance
        self.epochs = 1
        self.training_loss = 0
        

    def gpu_status(self, gpu_arg):
        """
        Checks on the status of a GPU for model device assignment.
        INPUTS:
            1. Command line GPU switch argument:    <bool>
        RETURNS:
            1. PyTorch device type available:       <device object>
        """
        # check if the GPU is currently available and set device flag appropriately
        self.device_location = "cuda:0" if torch.cuda.is_available() else "cpu"

        # if GPU switch was selected during program initiation
        if gpu_arg == True and self.device_location == "cuda:0":
            return torch.device(self.device_location)
        
        # GPU switch not requested of device not available
        else:
            if gpu_arg:
                print("Sorry, GPU not available, using CPU instead...")
            self.device_location = "cpu"
            return torch.device(self.device_location)


    def set_data_loaders(self, data_dir, 
                         train_dir='/train', 
                         valid_dir='/valid', 
                         test_dir='/test'):
        """
        Initializes and defines the different transforms along with the associated
        data loaders.
        INPUTS:
            1. Data dir root:   <str>
            2. Training dir:    <str>
            3. Validation dir:  <str>
            4. Test dir:        <str>
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
        train_dataset = datasets.ImageFolder(data_dir + train_dir, transform=train_transforms)
        valid_dataset = datasets.ImageFolder(data_dir + valid_dir, transform=test_transforms)
        test_dataset  = datasets.ImageFolder(data_dir + test_dir, transform=test_transforms)

        # Load data with DataLoader definitions for each image datasets using the correct trainform
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size)
        self.test_loader =  torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        self.class_to_idx = train_dataset.class_to_idx


    def set_network_model(self, arch='vgg16',
                          learning_rate=0.0001, 
                          hidden_size=2048,
                          dropout_rate=0.3,
                          ):
        """
        Loads a user selected network model architecture.
        INPUTS:
            1. Predefined model arch:       <str>
            2. Learning rate:               <float>
            3. Num of hidden units:         <int>
            4. Percent of dropouts:         <float>
        RETURNS:
            1. Selected model definition:   <model object>
            2. Predefined loss def:         <criterion object>
            3. Gradient descent def:        <optimizer object>
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
   

    def train_network(self, gpu_arg, model, optimizer, criterion, epochs=1, top_k=1):
        """
        Performs training of the network's features only along with validation
        accuracy and loss rates while conducting the training operation.
        INPUTS:
            1. GPU user selection:      <bool>
            2. Network model:           <model object>
            3. Gradient descent def:    <optimizer object>
            4. Predefined loss def:     <criterion object>
            5. User's epoch value:      <int>
            6. Top K id's for error:    <int>
        RETURNS:
            None
        """
        # update the requested epochs value
        self.epochs = epochs

        # push model to correct processor
        device = self.gpu_status(gpu_arg)
        model.to(device)

        # now for the training and validation process
        print(f"Training the model with the following parameters:\n"
              f"\tmodel archit.: \t{self.arch}\n"
              f"\thidden units: \t{self.hidden_size}\n"
              f"\tlearning rate: \t{self.learning_rate}\n"
              f"\ttotal epochs: \t{self.epochs}\n"
              f"\tGPU processing: \t{self.device_location.upper()}\n"
             ) 

        print("Starting epoch: 1 of {}...".format(self.epochs))
        
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
        Saves the neural network to a checkpoint file so it can be
        reloaded again without the need to re-train the network.
        INPUTS:
            1. Network model:           <model object>
            2. Gradient descent def:    <optimizer object>
            3. URL for checkpoint file: <str>
            4. Checkpoint file name     <str>
        RETURNS:
            None
        """
        # define the checkpoint dict for saving, loading and inference later
        checkpoint = {'arch' : self.arch,
                      'input_size' : self.input_size,
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
        try:
            torch.save(checkpoint, save_dir + "/" + checkpoint_file)
        except Exception as error:
            print("The following error: {} occurred while saving the checkpoint file to: {}".format(error, save_dir + "/" + checkpoint_file))
        else:
            print("Trained model saved to: {}".format(save_dir + "/" + checkpoint_file))


    # Method that loads a checkpoint and rebuilds the model
    def load_checkpoint(self, gpu_arg, checkpoint_file):
        """
        Method to load a checkpoint file and reassign required variables.
        INPUT:
            1. GPU user selection:          <bool>
            2. Checkpoint file:             <string>
        RETURNS:
            1. Selected model definition:   <model object>
            2. Gradient descent def:        <optimizer object>
        """
        # check if the GPU is currently available and set device flag appropriately
        _ = self.gpu_status(gpu_arg)
        
        # load the old model state
        checkpoint = torch.load(checkpoint_file, map_location=self.device_location)
        
        # load a pre-trained network model based on the command line argument, if supplied
        if checkpoint['arch'] == 'vgg13':
            model = models.vgg13(pretrained=True)
        elif checkpoint['arch'] == 'vgg16':
            model = models.vgg16(pretrained=True)
        elif checkpoint['arch'] == 'densenet121':
            model = models.densenet121(pretrained=True)
        else:
            model = models.vgg16(pretrained=True)
            print("Checkpoint model architecture not recoginized or supported. \n"
                  "Using default VGG16 instead. Available architectures: VGG13, \n"
                  "VGG16, and DenseNet121.\n")

        # freeze the networks parameters so no backprop occurs
        for param in model.parameters():
            param.requires_grad = False

        # in case more training is desired, assign needed values
        self.arch = checkpoint['arch']
        self.epochs = checkpoint['epochs']
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
        '''
        Scales, crops, and normalizes a PIL image for a PyTorch model,
        and returns an Numpy array.
        INPUTS:
            1. Relative image path and file name:   <str>
        RETURNS:
            1. Numpy array
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


    def predict(self, img, model, gpu_arg, topk=5):
        """
        Predicts the class (or classes) of an image using a trained deep learning model.
        INPUTS:
            1. Numpy image array
            2. Network model                    <model object>
            3. GPU user selection:              <bool>
            4. Requested top K results:         <int>
        RETURNS:
            1. List of top K probabilities:     <list>
            2. List of top K classes:           <list>
        """
        # Predict the class from an image file
        
        # push model to correct processor
        device = self.gpu_status(gpu_arg)
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