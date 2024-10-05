# README
This project is a parametric dataset generator for making mismatches with digital twins. This project was done as the master thesis for Pim van den Akker in collaboration with TU Eindhoven with supervisor Elena Torta.

The purpose of this project is to parametrically generate synthetic datasets for training a computer vision model to detect mismatches in digital twins. The dataset generator is made using Blender and Python. This generator allows for the creation of endless variations of rooms with different parameters. The dataset generator can be used to create a dataset for training a model to detect mismatches in digital twins.

For example, by changing a few parameters in the script, we can generate different room layouts:

<img src="room_variation.gif" alt="gif" width="400" height="300" title="Room Variations">

<img src="prediction examples.png" alt="alt text" width="400" height="300">

## Installation
To install, please download [Blender 3.6](https://www.blender.org/download/lts/3-6/).

Required packages (tested on Python 3.9):
- numpy (1.23.5)
- pandas (2.2.1)
- matplotlib (3.6.3)
- opencv-python (4.7.0.68)
- pytorch (2.1.2)
- torchvision (0.16.2)
- [bpycv](https://github.com/DIYer22/bpycv)

Install packages in Blender through [this guide](https://stackoverflow.com/questions/11161901/how-to-install-python-modules-in-blender).

NOTE: Blender uses its own Python version, so make sure to install the packages in the Blender Python folder according to the guide above.

## Usage
To use the dataset generator, open the `room_generation_v4.blend` file in Blender (located in the `blender_files` folder). Then run the script in the text editor (top tab). The script will generate a dataset with the parameters set in the script. Make sure the main folder is called "Blender".

To train the computer vision model, run the `train_model_mask_rcnn.py` script. This does not have to be run in Blender.

## File Structure
The `blender_files` folder contains the Blender file to generate the dataset. The geometry is determined here by using Blender geometry nodes.

The Python code that generates the datasets is in the `blender_python_code` folder. In `main.py`, the parameters for the dataset can be set. In the `dataset_generation` folder, the code for generating the dataset is located. If you want to change the way the ground truth or the images are generated, you can change the code here.

In the `model_training` folder, the code for training the model is located. The `train_model_mask_rcnn.py` script trains the model. The `visualization_and_validation` folder contains code for visualizing the results of the model.