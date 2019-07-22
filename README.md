# PyTorch Convolutional Deep Net 1

This project was developed as a deep learning image classifier built using the PyTorch framework. There are two python programs that can be used from the command line to first use transfer learning to modify a previously trained neural net and another program to generate a class prediction based on the network refined by the training program. In other words, the first file, `train.py`, will train a new network on a dataset and save the model as a checkpoint. The second file, `predict.py`, uses a trained network to predict the class for an input image. If you are working in a GPU computational environment, you can leverage the GPUs by activating the `--gpu` flag during program execution, see below for more details.


### Training
Train a new network on a data set with `train.py`. The code base was original developed and tested against the following data set: [Univ. of Oxford - Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) which contains 102 different flower categories and was developed by [Maria-Elena Nilsback](http://www.robots.ox.ac.uk/~men/) and [Andrew Zisserman](http://www.robots.ox.ac.uk/~az/). Moreover, it is also built on the PyTorch framework and requires version 0.4 or later.

* Basic usage: `python train.py data_directory`
* Prints out training loss, validation loss, and validation accuracy as the network trains
* Options:
    * Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
    * Choose architecture: `python train.py data_dir --arch "vgg13"`
    * Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
    * Use GPU for training: `python train.py data_dir --gpu`


### Predicting
Predict the object's name from an image with `predict.py` along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the object's name and class probability.

* Basic usage: `python predict.py /path/to/image checkpoint`
* Options:
    * Return top **_K_** most likely classes: `python predict.py input checkpoint --top_k 3`
    * Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
    * Use GPU for inference: `python predict.py input checkpoint --gpu`