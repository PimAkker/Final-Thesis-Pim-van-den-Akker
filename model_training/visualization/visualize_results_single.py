#%%
import torch
import os
import sys

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "utilities"))
sys.path.append(utils_path)
root_dir_name = 'Blender'

current_directory = os.getcwd().split("\\")
assert root_dir_name in current_directory, f"Current directory is {current_directory} and does not contain {root_dir_name}"
if current_directory[-1] != root_dir_name:
    # go down in the directory tree until the root directory is found
    while current_directory[-1] != root_dir_name:
        os.chdir("..")
        current_directory = os.getcwd().split("\\")


# add all the subdirectories to the path
dirs  = os.listdir()
root = os.getcwd()
for dir in dirs:
    sys.path.append(os.path.join(root, dir))
sys.path.append(os.getcwd())

import torch
from PIL import Image
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import numpy as np
from category_information import category_information
from utilities.coco_eval import *
from utilities.engine import *
from utilities.utils import *
from utilities.transforms import *
from utilities.dataloader import *
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import sys
from torchvision.ops.boxes import masks_to_boxes
import numpy as np

#%%
import matplotlib.pyplot as plt

image_path = r"C:\Users\pimde\OneDrive\thesis\Blender\data\Images\input-0-.jpg"
mask_path = r"C:\Users\pimde\OneDrive\thesis\Blender\data\Masks\inst-mask-0-.npy"
if __name__ == '__main__':
    

    num_classes = len(category_information)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model_instance_segmentation(num_classes)
    # load the weights
    model.load_state_dict(torch.load(r"C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\2024-04-11_11-49-12\model_2024-04-11_11-49-12_epochs_3.pth"))
    model.to(device)
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))

    # pick a random files pair from the specified folder



    image_orig = read_image(image_path)
    mask_true = np.load(mask_path)
    eval_transform = get_transform(train=False)

    model.eval()
    with torch.no_grad():
        x = eval_transform(image_orig)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    confidence_threshold = 0.9
    

    image_orig = (255.0 * (image_orig - image_orig.min()) / (image_orig.max() - image_orig.min())).to(torch.uint8)
    image_orig = image_orig[:3, ...]
    # get the labels from category_information and link them to the pred_labels
    pred_labels = pred["labels"]
    pred_labels = pred_labels[pred["scores"] > confidence_threshold]
    pred_labels = [list(category_information.keys())[list(category_information.values()).index(x)] for x in pred_labels]
    # change chairs removed to REMCHAIR

    red_labels = [x.replace("chairs removed", " REMCHAIR") for x in pred_labels]
    pred_labels = [x.replace("chairs new", " NEWCHAIR") for x in pred_labels]
    pred_boxes = pred["boxes"].long()
    pred_boxes = pred_boxes[pred["scores"] > confidence_threshold]
    
    output_image = draw_bounding_boxes(image_orig, pred_boxes, pred_labels, colors="red")
    # output_image = image
    # output_image = torch.zeros_like(image)
    
    mask_true = torch.from_numpy(mask_true)
    # instances are encoded as different colors
    obj_ids = torch.unique(mask_true)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]
    num_objs = len(obj_ids)
    
    # split the color-encoded mask into a set
    # of binary masks
    masks_true = (mask_true == obj_ids[:, None, None]).to(dtype=torch.uint8)
    # convert masks too booleon
    masks_true = masks_true.bool()

    # masks = (pred["masks"] > 0.7).squeeze(1)
    # output_image = draw_segmentation_masks(output_image, masks_true, alpha=0.5, colors="purple")


    masks = (pred["masks"] > confidence_threshold).squeeze(1)
    output_image = draw_segmentation_masks(output_image, masks, alpha=.5, colors="blue")

    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Prediction")
    plt.imshow(output_image.permute(1, 2, 0))


    
    show_image = True
    show_mask = True
    draw_bounding = True
    
    image_orig = read_image(image_path_temp)
    mask = torch.from_numpy(np.load(mask_path_temp))

    obj_ids = torch.unique(mask)[1:]
    num_objs = len(obj_ids)
    labels = obj_ids // 1000
    labels = labels.long()
    labels = labels
    masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8).bool()
    masks = masks
    boxes = masks_to_boxes(masks)
    boxes[:, 2] += boxes[:, 0] == boxes[:, 2]
    boxes[:, 3] += boxes[:, 1] == boxes[:, 3]
    if show_image:
        image_orig = (255.0 * (image_orig - image_orig.min()) / (image_orig.max() - image_orig.min())).to(torch.uint8)
        image_orig = image_orig[:3, ...]
    else :
        image_orig = torch.zeros_like(image_orig)
    pred_labels = [list(category_information.keys())[list(category_information.values()).index(x)] for x in labels]
    pred_labels = [x.replace("chairs removed", " REMCHAIR") for x in pred_labels]
    pred_labels = [x.replace("chairs new", " NEWCHAIR") for x in pred_labels]

    if draw_bounding:
        output_image = draw_bounding_boxes(image_orig, boxes, pred_labels, colors="red")
    
    if show_mask:
        output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="purple")

    
    plt.subplot(1, 2, 2) 
    plt.title("ground truth")
    plt.imshow(output_image.permute(1, 2, 0))
    plt.show()


# %%
