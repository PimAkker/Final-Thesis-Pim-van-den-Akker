#%%
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(path)
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
import matplotlib.patheffects as path_effects
from itertools import combinations


category_information_flipped = {v: k for k, v in category_information.items()}

def load_data(data_path, percentage=1):
    
    dataset = LoadDataset(data_path, get_transform(train=False), ignore_classes=np.array([0,1]))    

    dataset = torch.utils.data.Subset(dataset, range(int(len(dataset) * percentage)))
    return dataset

def load_model(weights_load_path, num_classes, device):
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(weights_load_path))
    model.to(device)
    return model

def run_model(model, data, image_number, device, box_threshold):
    """Runs the model and returns the prediction, boxes, labels, masks, image, and scores.

    Args:
        model (torch.nn.Module): The model to be run.
        data (torch.utils.data.Dataset): The dataset containing the images.
        image_number (int): The index of the image to be processed.
        device (torch.device): The device to run the model on.
        box_threshold (float): The threshold value for filtering bounding boxes(0-1).

    Returns:
        tuple: A tuple containing the prediction, boxes, labels, masks, image, and scores.
    """
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
def switch_black_and_white(image):
    """
    Switches the black and white colors in an image.

    Parameters:
    - image: numpy.ndarray
        The input image.

    Returns:
    - numpy.ndarray
        The image with black and white colors switched.
    """
    maskwhite = cv2.inRange(image, (180, 180, 180), (255, 255, 255))
    maskblack = cv2.inRange(image, (0, 0, 0), (180, 180, 180))
    image[maskwhite > 0] = [0, 0, 0]
    image[maskblack > 0] = [255, 255, 255]
    return image 
def show_pred_bounding_boxes(img, boxes, scores, labels, show_confidence_values=True, show_labels=True):
    """
    Display the image with bounding boxes overlaid.

    Args:
        img (Tensor): The input image tensor.
        boxes (Tensor): The bounding box coordinates.
        scores (Tensor): The confidence scores for each bounding box.
        labels (Tensor): The labels for each bounding box.
        show_confidence_values (bool, optional): Whether to show the confidence values inside the boxes. Defaults to True.
        show_labels (bool, optional): Whether to show the labels below the boxes. Defaults to True.
    """
    fig, ax = plt.subplots(1, figsize=(12, 6))
    image = img.mul(255).permute(1, 2, 0).byte().numpy()
    
    image = switch_black_and_white(image)

    
    ax.imshow(image)
    ax.axis('off')
    for i, box in enumerate(boxes[0]):
        if show_confidence_values:
            # show the confidence value inside the box
            score = scores[0][i].cpu().numpy()
            score_txt = plt.text(box[0], box[1], f"Confidence: {score:.2f}", color='black', fontsize=13, alpha=1, fontname='Computer Modern', fontfamily='serif')
            score_txt.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()]) 
        if show_labels:
            label = labels[0][i].cpu().numpy()
            label = category_information_flipped[int(label)].capitalize()
            # set the first letter to uppercase
            
            label_txt = plt.text(box[0], box[3],label , color='black', fontsize=13, alpha=1, ha='left', va='bottom', fontname='Computer Modern', fontfamily='serif')
            label_txt.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])

        box = box.cpu().numpy()
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=3, edgecolor='Blue', facecolor='None', alpha=0.5)
        ax.add_patch(rect)

    plt.show()

    
def show_masks(img, masks,mask_threshold=0.2):
    """
    Display the image with overlaid masks.

    Args:
        img (torch.Tensor): The input image tensor.
        masks (torch.Tensor): The masks tensor.

    Returns:
        None
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(img.mul(255).permute(1, 2, 0).byte().numpy())
    # show all the masks as different colors
    
    masks = masks[0].cpu().numpy()
    masks = masks > mask_threshold
    masks = np.sum(masks, axis=(0,1))
    plt.imshow(masks)
    plt.axis('off')
    plt.show()
        # show some info 
    # plt.title(f"mask shape: {mask.shape}")

    
def fill_bounding_boxes(boxes, shape= (1,256,256)):
    """
    Fills the bounding boxes with ones in the given masks.

    Args:
        boxes (torch.Tensor): A tensor containing the bounding box coordinates in the format [x_min, y_min, x_max, y_max].
        masks (torch.Tensor): A tensor containing the masks, only used to determine the size.

    Returns:
        numpy.ndarray: A numpy array with the filled bounding boxes. Consisting of ones inside the boxes and zeros outside.
    """
    
    filled_bb = np.zeros(shape)
    
    # make sure the boxes are numpy arrays
    if shape[0] == 0: 
        return np.zeros((1, shape[1], shape[2]))
    
    if type(boxes) == list and not isinstance(boxes[0], np.ndarray):
        boxes = boxes[0].cpu().numpy()
    
        

    for i, box in enumerate(boxes):
        if not isinstance(boxes, np.ndarray):
            x_min, y_min, x_max, y_max = box
        else:
            x_min, y_min, x_max, y_max = box
            
        filled_bb[i][int(y_min):int(y_max), int(x_min):int(x_max)] = 1
    return filled_bb
    
def plot_overlap(img, boxes, masks, threshold=0.0):
    """
    Plots the overlap between masks and bounding boxes to visualize uncertainty areas.

    Args:
        img (Tensor): The input image.
        boxes (Tensor): The bounding boxes.
        masks (Tensor): The masks.
        threshold (float, optional): The threshold value for the masks. Default is 0.0.

    Returns:
        None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    masks_copy = masks[0].clone()
    masks_copy[masks_copy > threshold] = 1
    masks_copy[masks_copy < threshold] = 0
    mask = masks_copy.sum((0, 1))

    ax1.set_title("mask uncertainty regions")
    ax1.imshow(mask.cpu().numpy(), alpha=0.5)

    shape = np.array(masks[0].sum(1).shape)
    filled_bb = fill_bounding_boxes(boxes, shape)
    
    ax2.imshow(np.sum(filled_bb, axis=0), alpha=0.5)


    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(ax1.images[0], cax=cbar_ax)
    cbar.set_label('Heatmap Intensity')

    plt.show()

def quantify_area_of_overlap(clustered_boxes, filled_bb):
    """
    Quantify the area of overlap between the boxes in the cluster. This is done by calculating the intersection over union metric, modified to work with more than two boxes.
    
    Parameters:
        clustered_boxes (numpy.ndarray): Array of clustered boxes. 
        filled_bb (numpy.ndarray): Array of filled bounding boxes.
        
    Returns:
        dict: A dictionary containing the overlap scores for each cluster.
    """
    
    clusters = np.unique(clustered_boxes)[1:]

    assigned_clusters = {}
    # Assign the boxes to the clusters
    for cluster in clusters:
        temp_cluster_list = []
        for i, filled_box in enumerate(filled_bb):
            if np.any(filled_box.astype(bool) & (clustered_boxes==cluster)):
                temp_cluster_list.append(i)        
        assigned_clusters[cluster] = temp_cluster_list

    intersection = {}
    union = {}
    boxes_overlap_scores = {i+1: 0 for i in range(len(clusters))}
    
    # calculate the "intersection OVER union" for each cluster
    for cluster, assigned_boxes in assigned_clusters.items():
        if len(assigned_boxes) > 1:
            # create a list of possible combinations of boxes in the cluster
            overlap_interaction_array =  combinations(assigned_boxes, 2)
            
            intersection[cluster] = 0  
            union[cluster] = 0 
            
            # calculate the intersection and union for each combination of boxes in the cluster,
            # this is not very efficient, but there aren't that many boxes generally so it should be fine.
             
            for i,j in overlap_interaction_array:
                intersection[cluster] += np.sum(filled_bb[i].astype(bool) & filled_bb[j].astype(bool))
                union[cluster] += np.sum(filled_bb[i].astype(bool) | filled_bb[j].astype(bool))
            boxes_overlap_scores[cluster] = intersection[cluster] / union[cluster]    


    return boxes_overlap_scores
    
    
def find_area_of_uncertainty(boxes, masks, labels_list, threshold, show_overlap=True):
    """
    Finds the area of uncertainty based on the given bounding boxes, masks, labels, and threshold.

    Parameters:
    - boxes (numpy.ndarray): The bounding boxes of the objects.
    - masks (numpy.ndarray): The masks of the objects.
    - labels_list (list): The list of labels for the objects.
    - threshold (float): The threshold value for determining the area of uncertainty.
    - show_overlap (bool): Whether to show the overlap between the uncertain area masks and boxes. Default is True.

    Returns:
    - uncertain_area_boxes (numpy.ndarray): The binary image representing the uncertain area.
    - labels_in_cluster (dict): A dictionary containing the labels in each cluster, that has uncertainty abovce the threshold. 
    - filled_bb (numpy.ndarray): The filled bounding boxes, of shape (num_boxes, height, width). The num_boxes index is the same as the index in the labels_list.

    """
    # prepare the data
    
    shape = np.array(masks[0].sum(1).shape)
    filled_bb = fill_bounding_boxes(boxes, shape)
    filled_bb_single_val = filled_bb.sum(0)
    filled_bb_single_val[filled_bb_single_val != 0] = 1 
    filled_bb_single_val= filled_bb_single_val.astype('uint8')

    # calculate which pixels are connected
    num_clusters_boxes, clustered_image_boxes = cv2.connectedComponents(filled_bb_single_val)
    uncertain_area_boxes = np.zeros_like(filled_bb_single_val)

    # check if labels_list and boxes are numpy arrays
    if not isinstance(labels_list, np.ndarray):
        labels_list = labels_list[0].cpu().numpy()
        boxes = boxes[0].cpu().numpy()
    
    labels_in_cluster = {}      
    labels_in_cluster_pos = {}
    
    boxes_overlap_score = quantify_area_of_overlap(clustered_image_boxes,filled_bb)
    
    
    # TODO: this is very slow, implement in a numpy way
    for i in range(1,num_clusters_boxes):
        cluster_mask = clustered_image_boxes == i
        
                
        
        if boxes_overlap_score[i] >= threshold:
            uncertain_area_boxes[cluster_mask] = 1
            
            # find which labels are in the cluster   for display purposes
            temp_labels_in_cluster = []
            temp_labels_in_cluster_pos = []
            for j, box in enumerate(filled_bb):
                
                # find the locations of the boxes in the cluster for display purposes
                x_min, y_min, _, _ = boxes[j]
                if np.any(box.astype(bool) & cluster_mask):
                    temp_labels_in_cluster_pos.append((int(x_min), int(y_min)) )
                    temp_labels_in_cluster.append(labels_list[j])    
                
            labels_in_cluster_pos[i]  = temp_labels_in_cluster_pos
            labels_in_cluster[i] = temp_labels_in_cluster
                    
    uncertain_area_indexes =  np.array(list(labels_in_cluster.keys())).astype(int)
    masks_merged = masks[0].cpu().numpy()[uncertain_area_indexes-1].squeeze(1).sum(0) 
    masks_merged[masks_merged > 0.05] = 1

    
            
    if show_overlap:
        labels_in_cluster_names = {k: [category_information_flipped[int(label)] for label in v] for k,v in labels_in_cluster.items()}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.set_title("Uncertain area masks")
        ax1.imshow(masks_merged, alpha=0.5)
        ax2.set_title(f"Uncertain area boxes")
        ax2.imshow(uncertain_area_boxes, alpha=0.5)
        # plot the labels in the cluster
        
        for i, (cluster, cluster_name) in enumerate(labels_in_cluster_names.items()):
            labels_list = f"Score:{round(boxes_overlap_score[cluster],2)}"
            for label in cluster_name:
                    labels_list = f"{labels_list} \n {label}"  
               
            ax2.text(labels_in_cluster_pos[cluster][0][0],labels_in_cluster_pos[cluster][0][1], labels_list, color='red', fontsize=12, alpha=0.8)
            
        plt.show()
        
    return uncertain_area_boxes, labels_in_cluster, filled_bb
def draw_ground_truth_bounding_boxes(img, boxes, labels):
    """
    Display the image with ground truth bounding boxes overlaid.

    Args:
        img (Tensor): The input image tensor.
        boxes (Tensor): The bounding box coordinates.
        labels (Tensor): The labels for each bounding box.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, figsize=(12, 6))
    image = img.mul(255).permute(1, 2, 0).byte().numpy()
    image = switch_black_and_white(image)
    ax.imshow(image)
    ax.axis('off')
    for i, box in enumerate(boxes):
        label = labels[i]
        label = category_information_flipped[int(label)].capitalize()
        # set the first letter to uppercase
        label_txt = plt.text(box[0], box[3], label, color='black', fontsize=13, alpha=1, ha='left', va='bottom', fontname='Computer Modern', fontfamily='serif')
        label_txt.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
        box = box.cpu().numpy()
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=3, edgecolor='Blue', facecolor='None', alpha=0.5)
        ax.add_patch(rect)
    plt.show()

def show_ground_truth_masks(img, masks, labels):
    """
    Display the image with ground truth masks overlaid.

    Args:
        img (Tensor): The input image tensor.
        masks (Tensor): The masks tensor.
        labels (Tensor): The labels for each mask.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, figsize=(10, 10))
    image = img.mul(255).permute(1, 2, 0).byte().numpy()
    ax.imshow(image)
    ax.axis('off')
    masks= torch.sum(masks, dim=0)
    masks = masks.cpu().numpy()

    plt.imshow(masks)
    plt.show()


 #%%

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # data_to_test_on = r'C:\Users\pimde\OneDrive\thesis\Blender\data\test\same_height_no_walls_no_tables_no_object_shift_big'
    data_to_test_on = r"real_world_data\Real_world_data_V3"
    # data_to_test_on = r"C:\Users\pimde\OneDrive\thesis\Blender\data\test\varying_heights\[]"
    num_classes = len(category_information)  
    
    
    box_threshold = 0.5
    mask_threshold = 0.5
    image_start_number = 7
    nr_images_to_show = 120
    
    overlap_bb_threshold = .1
    
    
    for i in range(nr_images_to_show):
        
        image_nr = i + image_start_number
        
        weights_load_path =r'C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\same_height_no_walls_WITH_shift_big_v4_model\weights.pth'
        model = load_model(weights_load_path, num_classes, device)
        data = load_data(data_to_test_on)
        
        ground_truth_boxes = data[image_nr][1]['boxes']
        ground_truth_labels = data[image_nr][1]['labels']
        
        prediction_dict, boxes,labels,pred_masks,img, scores  = run_model(model, data, image_nr, device,  box_threshold)
        # print(f"number of boxes: {len(boxes)}")
        # print(f"number of labels: {len(labels)}")
        # print(f"number of masks: {len(masks)}")
        plt.figure(figsize=(10,10)) 
        plt.axis('off')
        img_swapped = switch_black_and_white(img.mul(255).cpu().numpy().transpose(1,2,0).copy())
        plt.imshow(img_swapped)
        plt.show()
        
        show_pred_bounding_boxes(img, boxes, scores,labels)
        draw_ground_truth_bounding_boxes(img, ground_truth_boxes, ground_truth_labels)

        show_masks(img, pred_masks,mask_threshold=mask_threshold)
        show_ground_truth_masks(img, data[image_nr][1]['masks'], ground_truth_labels)
        # plot_overlap(img, boxes, pred_masks,0.2)
    # 
        # find_area_of_uncertainty(boxes,pred_masks,labels, overlap_bb_threshold, show_overlap=True)
    
# %%
