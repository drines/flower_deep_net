# **PyTorch Convolutional Deep Net 1**

This project was developed as a deep learning image classifier built using the PyTorch framework. There are two python programs included, first to load and modify a pre-trained neural net (transfer learning), and second, another to generate an object class prediction. In other words, the first file, `train.py`, will train a new network on a dataset and save the model as a checkpoint. The second file, `predict.py`, uses a trained network to predict the class for an input image. If a GPU is available, both the training and prediction computational work can leverage this hardware by activating the `--gpu` flag during program execution, see below for usage details.


### **Training**
Train a new network on a data set with `train.py`. The code base was original developed and tested against the following dataset: [Univ. of Oxford - Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) which contains 102 different flower categories and was developed by [Maria-Elena Nilsback](http://www.robots.ox.ac.uk/~men/) and [Andrew Zisserman](http://www.robots.ox.ac.uk/~az/). Moreover, it is also built on the PyTorch framework and requires version 0.4 or later.

* Basic usage: `python train.py data_directory`
* Prints out training loss, validation loss, and validation accuracy as the network trains
* Options:
    * Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
    * Choose architecture: `python train.py data_dir --arch "vgg13"`
    * Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
    * Use GPU for training: `python train.py data_dir --gpu`


### **Predicting**
Predict the object's name from an image with `predict.py` along with the probability of that name. That is, you pass in a single image `/path/to/image` and the program returns the top **_K_** predicted object name(s) and class probabilities.

* Basic usage: `python predict.py /path/to/image checkpoint`
* Options:
    * Return top **_K_** most likely classes: `python predict.py input checkpoint --top_k 3`
    * Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
    * Use GPU for inference: `python predict.py input checkpoint --gpu`

### **Data Loading**
Both programs use PyTorch's `torchvision` library to load data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The dataset should be split into three parts, training (`/train`), validation (`/valid`), and testing (`/test`). 

For training, image transformations using random scaling, cropping, and flipping are applied to help the network generalize prediction ability and leads to better performance ultimately. Moreover, the image input data is resized to `224x224 pixels` as required by the pre-trained networks.

The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. No scaling or rotation transformations are applied during these steps; however, the images are still resized then cropped to the appropriate size as above.

The pre-trained networks available come from the `torchvision.models` library (`"VGG13", "VGG16", or "DenseNet121"`) and were trained on the ImageNet dataset where each color channel was normalized separately. Thus, all the input images are normalized with the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images. These values will shift each color channel to be centered at 0 and range from -1 to 1.

### **Truth Label Mapping**
During training and validation, a mapping from category label to category name is loaded from the `JSON` file. If you use the flower dataset referenced above, then the `cat_to_name.json` file included in the repository can be used to map the flower names with the labels. It's a `JSON` object which can be read in with the python `json` [module](https://docs.python.org/2/library/json.html).