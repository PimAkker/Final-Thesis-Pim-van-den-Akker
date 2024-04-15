"""This file is used to calculate the Intersection over Union (IoU) of the model output and the ground truth masks."""
#%%
#%%
import torch
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
path = "\\".join(path.split("\\")[:-1])
os.chdir(path)
# ensure we are in the correct directory
root_dir_name = 'Blender'
current_directory = os.getcwd().split("\\")
assert root_dir_name in current_directory, f"Current directory is {current_directory} and does not contain root dir name:  {root_dir_name}"
if current_directory[-1] != root_dir_name:
    # go down in the directory tree until the root directory is found
    while current_directory[-1] != root_dir_name:
        os.chdir("..")
        current_directory = os.getcwd().split("\\")
        
sys.path.append(os.path.join(os.curdir, r"model_training\\utilities"))
sys.path.append(os.getcwd())
import torch
from PIL import Image
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import numpy as np

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from  model_training.utilities.coco_eval import *
from  model_training.utilities.engine import *
from  model_training.utilities.utils import *
from  model_training.utilities.transforms import *
from  model_training.utilities.dataloader import *
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import sys

import os
from model_training.utilities.engine import train_one_epoch, evaluate
import model_training.utilities.utils
from category_information import category_information, class_factor

# force reload the module
import importlib
importlib.reload(model_training.utilities.utils)
import time
#%%
if __name__ == '__main__':
    data_root = r'data'
    num_classes = len(category_information)
    
    continue_from_checkpoint = False
    save_model = True
    num_epochs = 3
    
    
    train_percentage = 0.8
    test_percentage = 0.2
    percentage_of_data_to_use = 1 #for debugging purposes only option to only use a percentage of the data
    
    batch_size = 4
    learning_rate = 0.005
    momentum=0.9
    weight_decay=0.0005
    
    weights_save_path = r"data\Models"
    weights_load_path = r"C:\Users\pimde\OneDrive\thesis\Blender\data\Models\model_2024-02-15_13-45-08.pth"
    
    save_info_path= r"data\Models\info"
    
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # device = torch.device('cpu')

    dataset = LoadDataset(data_root, get_transform(train=True))
    dataset_test = LoadDataset(data_root, get_transform(train=False))
   
    total_samples = len(dataset)
    train_samples = int(train_percentage * total_samples*percentage_of_data_to_use)
