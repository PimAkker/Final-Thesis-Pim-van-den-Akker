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
    ground_truth_labels = data.dataset[i][1]['labels'].cpu().numpy()
    
    return ground_truth_boxes, ground_truth_labels

def calculate_correct_hypothesis_rate(overlap_indexes_dict,ground_truth_labels, predict_labels):
    correct_hypothesis = 0
    for ground_truth_cluster, predict_cluster in overlap_indexes_dict.items():
        
        # check if the array is empty if so continue
        if len(predict_cluster) == 0:
            continue
        
        if ground_truth_labels[ground_truth_cluster] in predict_labels[predict_cluster]:
            correct_hypothesis += 1
    if len(ground_truth_labels) == 0:
        return 1
    else:
        correct_hypothesis_rate = correct_hypothesis / len(ground_truth_labels)
    return correct_hypothesis_rate
if __name__ == "__main__":
    data_path = r'C:\Users\pimde\OneDrive\thesis\Blender\real_world_data\Real_world_data_V2'
    model_path = r'C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\same_height_no_walls_v4\weights.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = detect_uncertainty_areas.load_data(data_path)
    model = detect_uncertainty_areas.load_model(model_path, num_classes= len(category_information), device=device)
    threshold = 0.2
    nr_of_images = len(data)
    
    correct_hypothesis_rate_array = np.zeros(nr_of_images)
    for i in range(nr_of_images):
        
        prediction, pred_boxes, pred_labels, pred_masks, pred_img, pred_scores = detect_uncertainty_areas.run_model(model, data, i, device, 0.5)
        
        uncertain_area_boxes, labels_incluster, filled_bb = detect_uncertainty_areas.find_area_of_uncertainty(pred_boxes, pred_masks, pred_labels, threshold, show_overlap=False)
        ground_truth_boxes, ground_truth_labels = get_ground_truth(data, i)
        overlap_indexes = find_overlap_indexes(filled_bb, ground_truth_boxes)
        correct_hypothesis_rate_array[i] = calculate_correct_hypothesis_rate(overlap_indexes,ground_truth_labels,pred_labels[0].cpu().numpy())
    print(f"Correct hypothosis rate: {np.sum(correct_hypothesis_rate_array) / nr_of_images}")
# %%
