#%%
import torch
import os
import sys


# ensure we are in the correct directory
root_dir_name = 'Blender'
root_dir_path = os.path.abspath(__file__).split(root_dir_name)[0] + root_dir_name
os.chdir(root_dir_path)
sys.path.extend([os.path.join(root_dir_path, dir) for dir in os.listdir(root_dir_path)])
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
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from numpy import random
    

#%%

image_path = r"C:\Users\pimde\OneDrive\thesis\Blender\data\test\same_heights_v3\[]\Images"
mask_path = r"C:\Users\pimde\OneDrive\thesis\Blender\data\test\same_heights_v3\[]\Masks"

# image_path = r'C:\Users\pimde\OneDrive\thesis\Blender\real_world_data\Real_world_data_V2\Images'
# mask_path = r'C:\Users\pimde\OneDrive\thesis\Blender\real_world_data\Real_world_data_V2\Masks'

# image_path = r"C:\Users\pimde\OneDrive\thesis\Blender\data\test1\memoryleaksetc\[]\Images"
# mask_path = r"C:\Users\pimde\OneDrive\thesis\Blender\data\test1\memoryleaksetc\[]\Masks"

show_input_image = False
show_image = True
show_mask = True
show_ground_truth = True
draw_bounding = True
render_num_images = 5
randomize_images = False
 
model_weights_path = r'C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\same_height_no_walls_v3\weights.pth'

mask_confidence_threshold = 0.5

number_of_pred_mask_to_show = 20
label_confidence_threshold = 0.5


image_path_list = [os.path.join(image_path, file) for file in os.listdir(image_path) if file.endswith(".png")]
if show_ground_truth:
    mask_path_list = [os.path.join(mask_path, file) for file in os.listdir(mask_path) if file.endswith(".npy")]

def replace_label_name(list, from_name, to_name):
    [x.replace(from_name, to_name) for x in list]
    return list



if __name__ == '__main__':
    
    image_indices =  list(range(len(image_path_list)))
    if randomize_images:
        random_indices = random.choice(image_indices, render_num_images, replace=False)
    else:
        random_indices = image_indices[:render_num_images]
    for i, file_nr in enumerate(random_indices):
        print(f"Showing image {i + 1} of {render_num_images} with name {os.path.split(image_path_list[file_nr])[-1]}")
        
        
        num_classes = len(category_information)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = get_model_instance_segmentation(num_classes)
        model.load_state_dict(torch.load(model_weights_path))
        model.to(device)
    
    
        image_orig = read_image(image_path_list[file_nr])
        
        if show_input_image:
            plt.imshow(image_orig.permute(1, 2, 0))
            plt.axis('off')
            plt.show()
        plt.figure(figsize=(10,10))
        if show_ground_truth:
            mask_path_temp = mask_path_list[file_nr]
            mask_true = np.load(mask_path_temp)
            mask_true = torch.from_numpy(mask_true)
            obj_ids = torch.unique(mask_true)
            masks_true = (mask_true == obj_ids[:, None, None]).to(dtype=torch.uint8).bool()
            
        eval_transform = get_transform(train=False)

        model.eval()
        
        # Make the model predictions
        with torch.no_grad():
            x = eval_transform(image_orig)
            x = x[:3, ...].to(device)
            predictions = model([x, ])
            pred = predictions[0]

      

        # Normalize the image to the range [0, 255]
        # image_orig = (255.0 * (image_orig - image_orig.min()) / (image_orig.max() - image_orig.min())).to(torch.uint8)
        image_orig = image_orig[:3, ...]

        pred_labels = pred["labels"]
        pred_labels = pred_labels[pred["scores"] > label_confidence_threshold]
        pred_labels = [list(category_information.keys())[list(category_information.values()).index(x)] for x in pred_labels]
        
        # For easier reading change the label names
        pred_labels = replace_label_name(pred_labels, "chairs removed", " REMCHAIR")
        pred_labels = replace_label_name(pred_labels, "chairs new", " NEWCHAIR")
        pred_labels = replace_label_name(pred_labels, "pillars new", "NEWPIL")
        pred_labels = replace_label_name(pred_labels, "pillars removed", "REMPIL")
        pred_labels = replace_label_name(pred_labels, "walls tables removed", "REMTAB")
        pred_labels = replace_label_name(pred_labels, "walls tables new", "NEWTAB")
        
        pred_boxes = pred["boxes"].long()
        pred_boxes = pred_boxes[pred["scores"] > label_confidence_threshold]


        if draw_bounding:
            output_image = draw_bounding_boxes(image_orig, pred_boxes, pred_labels, colors="red")
        else:
            output_image = image_orig
        
        
        if show_ground_truth == False:
            obj_ids = torch.unique(pred["masks"])

        obj_ids = obj_ids[2:]
        num_objs = len(obj_ids)

        masks = (pred["masks"] > mask_confidence_threshold).squeeze(1)
        mask_confidence_scores = torch.sum(masks, dim=(1, 2)) # this is the confidence score for each mask between 0 and masks.size(1) * masks.size(2)
        
        _, indices = torch.topk(mask_confidence_scores, k=min(number_of_pred_mask_to_show, len(mask_confidence_scores)))
        masks = masks[indices, ...]
        output_image = draw_segmentation_masks(output_image, masks, alpha=.5)
        plt.imshow(output_image.permute(1, 2, 0))
        if show_ground_truth:
            mask_true = torch.from_numpy(np.load(mask_path_temp))

            obj_ids = torch.unique(mask_true)[1:]
            num_objs = len(obj_ids)
            labels = obj_ids // 1000
            labels = labels.long()
            labels = labels
            masks = (mask_true == obj_ids[:, None, None]).to(dtype=torch.uint8).bool()

            boxes = masks_to_boxes(masks)
            boxes[:, 2] += boxes[:, 0] == boxes[:, 2]
            boxes[:, 3] += boxes[:, 1] == boxes[:, 3]
            if show_image:
                image_orig = (255.0 * (image_orig - image_orig.min()) / (image_orig.max() - image_orig.min())).to(torch.uint8)
                image_orig = image_orig[:3, ...]
            else:
                image_orig = torch.zeros_like(image_orig)
            pred_labels = [list(category_information.keys())[list(category_information.values()).index(x)] for x in labels]
            pred_labels = replace_label_name(pred_labels, "chairs removed", " REMCHAIR")
            pred_labels = replace_label_name(pred_labels, "chairs new", " NEWCHAIR")

            if draw_bounding:
                output_image = draw_bounding_boxes(image_orig, boxes, pred_labels, colors="red")
            else:
                output_image = image_orig
            
            if show_mask:
                output_image = draw_segmentation_masks(output_image, masks, alpha=0.5)

            plt.figure(figsize=(10,10))
            plt.title(" ground truth")


            plt.imshow(output_image.permute(1, 2, 0))
            plt.axis('off')
            plt.show()
            
    print("done")

    # %%
