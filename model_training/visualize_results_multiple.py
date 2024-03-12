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

    

#%%
image_path = r"data\Images"
mask_path = r"data\Masks"
show_image = True
show_mask = True
show_ground_truth = True
draw_bounding = True
render_num_images = 100

mask_confidence_threshold = 0.95
label_confidence_threshold = 0.5

image_path_list = [os.path.join(image_path, file) for file in os.listdir(image_path) if file.endswith(".jpg")]
if show_ground_truth:
    mask_path_list = [os.path.join(mask_path, file) for file in os.listdir(mask_path) if file.endswith(".npy")]

if __name__ == '__main__':
    for file_nr in range(render_num_images):
        print(f"Showing image {file_nr + 1} of {render_num_images} with name {os.path.split(image_path_list[file_nr])[-1]}")
        
        num_classes = len(category_information)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = get_model_instance_segmentation(num_classes)
        model.load_state_dict(torch.load(r"C:\Users\pimde\OneDrive\thesis\Blender\data\Models\model_2024-02-16_11-37-42_epochs_10.pth"))
        model.to(device)
        sys.path.append(os.path.abspath(os.path.dirname(__file__)))

        image_orig = read_image(image_path_list[file_nr])
        
        if show_ground_truth:
            mask_path_temp = mask_path_list[file_nr]
            mask_true = np.load(mask_path_temp)
            mask_true = torch.from_numpy(mask_true)
            obj_ids = torch.unique(mask_true)
            masks_true = (mask_true == obj_ids[:, None, None]).to(dtype=torch.uint8).bool()
            
        eval_transform = get_transform(train=False)

        model.eval()
        with torch.no_grad():
            x = eval_transform(image_orig)
            x = x[:3, ...].to(device)
            predictions = model([x, ])
            pred = predictions[0]

      

        image_orig = (255.0 * (image_orig - image_orig.min()) / (image_orig.max() - image_orig.min())).to(torch.uint8)
        image_orig = image_orig[:3, ...]

        pred_labels = pred["labels"]
        pred_labels = pred_labels[pred["scores"] > label_confidence_threshold]
        pred_labels = [list(category_information.keys())[list(category_information.values()).index(x)] for x in pred_labels]
        red_labels = [x.replace("chairs removed", " REMCHAIR") for x in pred_labels]
        pred_labels = [x.replace("chairs new", " NEWCHAIR") for x in pred_labels]
        pred_boxes = pred["boxes"].long()
        pred_boxes = pred_boxes[pred["scores"] > label_confidence_threshold]


        if draw_bounding:
            output_image = draw_bounding_boxes(image_orig, pred_boxes, pred_labels, colors="red")
        else:
            output_image = image_orig
        
        
        if show_ground_truth == False:
            obj_ids = torch.unique(pred["masks"])

        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        

        masks = (pred["masks"] > mask_confidence_threshold).squeeze(1)
        output_image = draw_segmentation_masks(output_image, masks, alpha=.5, colors="blue")

        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1)
        # plt.title("Prediction")
        # plt.imshow(output_image.permute(1, 2, 0))
        # plt.show()
        if show_ground_truth:
            mask_true = torch.from_numpy(np.load(mask_path_temp))

            obj_ids = torch.unique(mask_true)[1:]
            num_objs = len(obj_ids)
            labels = obj_ids // 1000
            labels = labels.long()
            labels = labels
            masks = (mask_true == obj_ids[:, None, None]).to(dtype=torch.uint8).bool()
            masks = masks
            boxes = masks_to_boxes(masks)
            boxes[:, 2] += boxes[:, 0] == boxes[:, 2]
            boxes[:, 3] += boxes[:, 1] == boxes[:, 3]
            if show_image:
                image_orig = (255.0 * (image_orig - image_orig.min()) / (image_orig.max() - image_orig.min())).to(torch.uint8)
                image_orig = image_orig[:3, ...]
            else:
                image_orig = torch.zeros_like(image_orig)
            pred_labels = [list(category_information.keys())[list(category_information.values()).index(x)] for x in labels]
            pred_labels = [x.replace("chairs removed", " REMCHAIR") for x in pred_labels]
            pred_labels = [x.replace("chairs new", " NEWCHAIR") for x in pred_labels]

            if draw_bounding:
                output_image = draw_bounding_boxes(image_orig, boxes, pred_labels, colors="red")
            else:
                output_image = image_orig
            
            if show_mask:
                output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="purple")

            plt.subplot(1, 2, 2)
            plt.title("ground truth")
            plt.imshow(output_image.permute(1, 2, 0))
            plt.show()
            
    print("done")

    # %%
