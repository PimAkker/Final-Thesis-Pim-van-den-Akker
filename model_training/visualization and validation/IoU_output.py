"""This file is used to calculate the Intersection over Union (IoU) of the model output and the ground truth masks."""
#%%
#%%
import torch
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
path = "\\".join(path.split(os.sep)[:-1])
os.chdir(path)
# ensure we are in the correct directory
root_dir_name = 'Blender'
current_directory = os.getcwd().split(os.sep)
assert root_dir_name in current_directory, f"Current directory is {current_directory} and does not contain root dir name:  {root_dir_name}"
if current_directory[-1] != root_dir_name:
    # go down in the directory tree until the root directory is found
    while current_directory[-1] != root_dir_name:
        os.chdir("..")
        current_directory = os.getcwd().split(os.sep)
        
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
    
    data_root = r'real_world_data\Real_world_data_V1'
    num_classes = len(category_information)
    
    
    weights_save_path = r"data\Models"
    weights_load_path = r"data\Models\info\2024-04-23_13-57-08\weights.pth"
    
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset_test = LoadDataset(data_root, get_transform(train=False))

    data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    collate_fn=utils.collate_fn
    )


    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(weights_load_path))

    # move model to the right device
    model.to(device)
        
    eval_output = evaluate(model, data_loader_test, device=device)
    
# %%
    eval_output.eval_imgs
    bbox_IoU_array = eval_output.coco_eval['bbox'].stats 
    print(f"mean average precision (box)   {round(np.mean(bbox_IoU_array[:6]), 6)}")
    print(f"mean average recall    (box)   {round(np.mean(bbox_IoU_array[6:]), 6)}")
    
    segm_IoU_array = eval_output.coco_eval['segm'].stats
    print(f"mean average precision (segm)  {round(np.mean(segm_IoU_array[:6]), 6)}")
    print(f"mean average recall    (segm)  {round(np.mean(segm_IoU_array[6:]), 6)}")
    
    segm_IoU_array = eval_output.coco_eval['segm'].stats
# %%
