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

def find_overlap_indexes(filled_bb,labels_incluster, ground_truth_boxes, ground_truth_labels):
    
    
        
    
    return overlap_indexes
def get_ground_truth(data, i):
    ground_truth_boxes = data.dataset[i][1]['boxes'].cpu().numpy()
    ground_truth_labels = data.dataset[i][1]['labels'].cpu().numpy()
    
    return ground_truth_boxes, ground_truth_labels
def calculate_correct_hypothesis_rate(uncertain_area_boxes, labels_incluster, ground_truth_boxes, ground_truth_labels):
    filled_ground_truth_boxes = detect_uncertainty_areas.fill_bounding_boxes(ground_truth_boxes, ground_truth_labels, category_information)
    
    
    pass
if __name__ == "__main__":
    data_path = r'C:\Users\pimde\OneDrive\thesis\Blender\real_world_data\Real_world_data_V3'
    model_path = r'C:\Users\pimde\OneDrive\thesis\Blender\data\Models\no_tables_model\weights.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = detect_uncertainty_areas.load_data(data_path)
    model = detect_uncertainty_areas.load_model(model_path, num_classes= len(category_information), device=device)
    threshold = 0.2
    
    # for i in range(len(data)):
    for i in range(3):
        
        prediction, boxes, labels, masks, img, scores = detect_uncertainty_areas.run_model(model, data, i, device, 0.5)
        
        uncertain_area_boxes, labels_incluster, filled_bb = detect_uncertainty_areas.find_area_of_uncertainty(boxes, masks, labels, threshold, show_overlap=True)
        ground_truth_boxes, ground_truth_labels = get_ground_truth(data, i)
        overlap_indexes = find_overlap_indexes(filled_bb,labels_incluster, ground_truth_boxes, ground_truth_labels)
# %%
