#%%
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
from category_information import category_information
import torch
from  model_training.utilities.dataloader import get_transform, LoadDataset, get_model_instance_segmentation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import scipy.cluster.hierarchy as fclusterdata

def load_data(data_path, percentage=1):
    
    dataset = LoadDataset(data_path, get_transform(train=False), ignore_indexes=2)    
    percentage_of_dataset_to_use = 1
    dataset = torch.utils.data.Subset(dataset, range(int(len(dataset) * percentage_of_dataset_to_use)))
    return dataset

def load_model(weights_load_path, num_classes, device):
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(weights_load_path))
    model.to(device)
    return model

def run_model(model, data, image_number, device, mask_threshold, box_threshold):
    """runs the model and returns number_of_images images"""
    boxes = []
    labels = []
    masks = []
    scores = []
    
    img, _ = data[image_number]
    model.eval()
    
    with torch.no_grad():
        prediction = model([img.to(device)])
        pass_threshold_indices_labels = torch.where(prediction[0]['scores'] > box_threshold)
        boxes_thresholded = prediction[0]['boxes'][pass_threshold_indices_labels]
        labels_thresholded = prediction[0]['labels'][pass_threshold_indices_labels]
        masks_thresholded = prediction[0]['masks'][pass_threshold_indices_labels]
        
        boxes.append(boxes_thresholded)
        labels.append(labels_thresholded)
        masks.append(masks_thresholded)
        scores.append(prediction[0]['scores'])
        
    return prediction, boxes, labels, masks, img, scores

def show_boxes(img, boxes):
    fig, ax = plt.subplots(1)
    ax.imshow(img.mul(255).permute(1, 2, 0).byte().numpy())
    for box in boxes[0]:
        box = box.cpu().numpy()
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()
    
def show_masks(img, masks):
    fig, ax = plt.subplots(1)
    ax.imshow(img.mul(255).permute(1, 2, 0).byte().numpy())
    for mask in masks[0]:
        mask = mask[0].cpu().numpy()
        ax.imshow(mask, alpha=0.5)
        # show some info 
        plt.title(f"mask shape: {mask.shape}")
    plt.show()
    
def fill_bounding_boxes(boxes, masks):
    
    filled_bb = np.zeros_like(masks[0].sum(1).cpu().numpy())
    for i, box in enumerate(boxes[0]):
        x_min, y_min, x_max, y_max = box.cpu().numpy()
        filled_bb[i][int(y_min):int(y_max), int(x_min):int(x_max)] = 1
    return filled_bb
    
def plot_overlap(img, boxes, masks, threshold=0.0):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    masks_copy = masks[0].clone()
    masks_copy[masks_copy > threshold] = 1
    masks_copy[masks_copy < threshold] = 0
    mask = masks_copy.sum((0, 1))

    ax1.set_title("mask uncertainty regions")
    ax1.imshow(mask.cpu().numpy(), alpha=0.5)

    filled_bb = fill_bounding_boxes(boxes, masks)
    
    ax2.imshow(np.sum(filled_bb, axis=0), alpha=0.5)


    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ax1.images[0], cax=cbar_ax)
    cbar.set_label('Heatmap Intensity')

    plt.show()
    
def find_area_of_uncertainty(boxes,masks,threshold, show_overlap=True):
    """ 
    Using scipy clustering, clusters all boxes that overlap
    """
    # use hcluster
    # prepare the data
    filled_bb = fill_bounding_boxes(boxes, masks)
    filled_bb_summed = filled_bb.sum(0)
    filled_bb_single_val = filled_bb.sum(0)
    filled_bb_single_val[filled_bb_single_val != 0] = 1 
    filled_bb_single_val= filled_bb_single_val.astype('uint8')

    # calculate which pixels are connected
    num_clusters, cluster_id = cv2.connectedComponents(filled_bb_single_val)
    uncertain_area = np.zeros_like(filled_bb_summed)
    for i in range(1,num_clusters-1):
        cluster_mask = cluster_id == i
        cluster_max = np.max(filled_bb_summed[cluster_mask])
        if cluster_max > threshold:
            uncertain_area[cluster_mask] = 1
            
    if show_overlap:
        fig, ax = plt.subplots(1)
        ax.imshow(uncertain_area, alpha=0.5)
        plt.show()        
    
    return uncertain_area
        
    
    
    
    
#%%

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_to_test_on = r'real_world_data\Real_world_data_V2'
    # data_to_test_on = r"C:\Users\pimde\OneDrive\thesis\Blender\data\test\varying_heights\[]"
    num_classes = len(category_information)  
    
    
    box_threshold = 0.5
    mask_threshold = 0.5
    nr_images_to_show = 5
    image_start_number = 1
    overlap_bb_threshold = 2
    
    
    for i in range(nr_images_to_show):
        
        image_nr = i + image_start_number
        
        weights_load_path = r"C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\same_height_v3\weights.pth"
        model = load_model(weights_load_path, num_classes, device)
        data = load_data(data_to_test_on)
        
        prediction_dict, boxes,labels,masks,img, scores  = run_model(model, data, image_nr, device, mask_threshold, box_threshold)
        # print(f"number of boxes: {len(boxes)}")
        # print(f"number of labels: {len(labels)}")
        # print(f"number of masks: {len(masks)}")
        
        
        show_boxes(img, boxes)
        # show_masks(img, masks)
        plot_overlap(img, boxes, masks,0.1)
    
        find_area_of_uncertainty(boxes,masks,overlap_bb_threshold, show_overlap=True)
    
# %%
