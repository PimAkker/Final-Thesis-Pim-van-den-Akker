#%%
import torch
import os
import sys
import numpy as np
root_dir_name = 'Blender'
root_dir_path = os.path.abspath(__file__).split(root_dir_name)[0] + root_dir_name
os.chdir(root_dir_path)
sys.path.extend([os.path.join(root_dir_path, dir) for dir in os.listdir(root_dir_path)])
sys.path.append(os.getcwd())        
sys.path.append(os.path.join(os.curdir, r"model_training\\utilities"))
sys.path.append(os.getcwd())
from category_information import category_information

from visualization_and_validation import detect_uncertainty_areas
import matplotlib.pyplot as plt
from importlib import reload
reload(detect_uncertainty_areas)

def find_overlap_indexes(filled_predict_boxes, ground_truth_boxes):
    
    # make the shape for the ground truth boxes 
    ground_truth_shape = (len(ground_truth_boxes), filled_predict_boxes.shape[1], filled_predict_boxes.shape[2])
    
    filled_ground_truth_boxes = detect_uncertainty_areas.fill_bounding_boxes(ground_truth_boxes, shape = ground_truth_shape)  
    
    gt_boxes_reshaped = filled_ground_truth_boxes[:, np.newaxis, :, :]
    pred_boxes_reshaped = filled_predict_boxes[np.newaxis, :, :, :]
    
    overlap_matrix = np.any(np.logical_and(gt_boxes_reshaped, pred_boxes_reshaped), axis=(2, 3))
    
    overlap_indexes_dict = {i: list(np.nonzero(overlap_matrix[i])[0]) for i in range(overlap_matrix.shape[0])}
            
    return overlap_indexes_dict

def get_ground_truth(data, i):
    ground_truth_boxes = data.dataset[i][1]['boxes'].cpu().numpy()
    ground_truth_masks = data.dataset[i][1]['masks'].cpu().numpy()
    ground_truth_labels = data.dataset[i][1]['labels'].cpu().numpy()
    
    
    return ground_truth_boxes,ground_truth_masks, ground_truth_labels

def calculate_correct_hypothesis_rate(overlap_indexes_dict,ground_truth_labels, predict_labels):
    correct_hypothesis = 0
    for ground_truth_cluster, predict_cluster in overlap_indexes_dict.items():
        
        # check if the array is empty if so continue
        if len(predict_cluster) == 0:
            continue
        
        if ground_truth_labels[ground_truth_cluster] in predict_labels[predict_cluster]:
            correct_hypothesis += 1
    if len(ground_truth_labels) == 0:
        return 1, 0, 1        
        
    num_predictions = len(predict_labels)
    true_positives = correct_hypothesis
    
    false_positives = num_predictions - true_positives
    false_negatives = len(ground_truth_labels) - true_positives
    f1 = true_positives / (true_positives + 0.5 * (false_positives + false_negatives))
    true_positive_rate = true_positives / (true_positives + false_negatives)
    false_positive_rate = 0 if false_positives == 0 else false_positives / (false_positives + true_positives)

    
    return  true_positive_rate, false_positive_rate, f1

def calculate_mask_only_IoU(pred_masks, ground_truth_mask, pred_scores, box_threshold, mask_threshold = 0.1):
    """
    Calculate the intersection over union metric for two masks.

    Args:
        pred_masks (numpy.ndarray): The prediction mask.
        ground_truth_mask (numpy.ndarray): The ground truth mask.
        box_threshold (float): The threshold for the bounding box. (NOT FOR THE MASK)

    Returns:
        float: The intersection over union metric.
"""

    pass_threshold_indices_labels = np.where(pred_scores[0].cpu().numpy() > box_threshold)

    pred_masks = pred_masks[pass_threshold_indices_labels]
    
    # set the masks to boolean values so they can be used for the intersection over union
    pred_masks = np.sum(pred_masks, axis=(0,1)) > mask_threshold
    pred_masks[pred_masks > 0] = 1
    ground_truth_mask = np.sum(ground_truth_mask, axis=0) > 0
    ground_truth_mask[ground_truth_mask > 0] = True  
    
    intersection = np.sum(pred_masks & ground_truth_mask)
    union = np.sum(pred_masks | ground_truth_mask)
    return intersection / union, pred_masks
        



if __name__ == "__main__":
    # data_path = r'C:\Users\pimde\OneDrive\thesis\Blender\real_world_data\Real_world_data_V3'
    data_path = r'C:\Users\pimde\OneDrive\thesis\Blender\real_world_data\Real_world_data_V3'
    model_path = r'C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\same_height_no_walls_no_tables_no_object_shift_big_model\weights.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = detect_uncertainty_areas.load_data(data_path)
    model = detect_uncertainty_areas.load_model(model_path, num_classes= len(category_information), device=device)

    uncertainty_threshold = 0.2
    mask_threshold = 0.3
    nr_of_images = len(data)
    # nr_of_images =  30
    box_threshold = 0.5
    visualize_images = False

    true_positive_rate_array = np.zeros(nr_of_images)
    false_positive_rate_array = np.zeros(nr_of_images)
    f1_array = np.zeros(nr_of_images)
    mask_IoU = np.zeros(nr_of_images)
    
    for i in range(nr_of_images):
        if i % 10 == 0:
            print(f"Image {i} of {nr_of_images}", end='\r')
        prediction, pred_boxes, pred_labels, pred_masks, pred_img, pred_scores = detect_uncertainty_areas.run_model(model, data, i, device, box_threshold)
        
        uncertain_area_boxes, labels_incluster, filled_bb = detect_uncertainty_areas.find_area_of_uncertainty(pred_boxes, pred_masks, pred_labels, uncertainty_threshold, show_overlap=False)
        ground_truth_boxes,ground_truth_masks, ground_truth_labels = get_ground_truth(data, i)
        overlap_indexes = find_overlap_indexes(filled_bb, ground_truth_boxes)
        true_positive_rate_array[i],false_positive_rate_array[i], f1_array[i]  = calculate_correct_hypothesis_rate(overlap_indexes,ground_truth_labels,pred_labels[0].cpu().numpy())

        mask_IoU[i], scored_pred_mask = calculate_mask_only_IoU(pred_masks[0].cpu().numpy(), ground_truth_masks, pred_scores, box_threshold,mask_threshold=mask_threshold)
        if visualize_images:            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.title("Input")
            plt.imshow(pred_img.permute(1, 2, 0))
            
            plt.subplot(1, 3, 2)
            plt.title("Ground truth")
            plt.imshow(np.sum(ground_truth_masks,axis=0))
            
            
            plt.subplot(1, 3, 3)
            plt.title(f"Prediction")
            plt.imshow(scored_pred_mask)
            plt.show()

        
    print(f"for model {os.path.split(model_path)[-2]}")
    print(f"True positive rate: {np.sum(true_positive_rate_array) / nr_of_images}")
    print(f"False positive rate: {np.sum(false_positive_rate_array) / nr_of_images}")
    print(f"F1 score: {np.sum(f1_array) / nr_of_images}")
    print(f"Mask IoU: {np.nansum(mask_IoU)/nr_of_images}")
    
# %%
