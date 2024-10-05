# README
This project is a paramateric dataset generator for making mismatches with digital twins. This project was done as the master thesis for Pim van den Akker in collaboration with TU Eindhoven with supervisor Elena Torta. 

The purpose of this project is the parametrically generate synthetic datasets for training of a computer vision model to detect mismatches in digital twins by simulating 2D LiDAR sensors. The dataset generator is made using blender and python. This generator allows for the creation of endless variations of rooms with different parameters. The dataset generator can be used to create a dataset for training a model to detect mismatches in digital twins.

For example by changing a few parameters in the script we can generate different room layouts:


<img src="room_variation.gif" alt="gif" width="500" height="400" title="Room Variations">

These rooms are use to generate mismatches. We train a computer vision model to detect these mismatches:

<img src="prediction examples.png" alt="Computer vision mismatch examples" width="500" height="400">



## Installation
To install please go download [blender 3.6](https://www.blender.org/download/lts/3-6/). 

required packages (tested on python 3.9):
- numpy (1.23.5)
- pandas (2.2.1)
- matplotlib (3.6.3)
- opencv-python (4.7.0.68)
- pytorch (2.1.2)
- torchvision (0.16.2)
- [bpycv](https://github.com/DIYer22/bpycv)

install packages in blender through [this](https://stackoverflow.com/questions/11161901/how-to-install-python-modules-in-blender) guide. 

NOTE: Blender uses it's own python version, so make sure to install the packages in the blender python folder according to the guide above. 




## Usage	
To use the dataset generator, open the room_generation_v4.blend file in blender (located in the blender_files folder). Then run the script in the text editor (top tab). The script will generate a dataset with the parameters set in the script. Make sure the main folder is called "Blender". 

To train the computer vision model run the train_model_mask_rcnn.py script. This does not have to be run in blender. 





