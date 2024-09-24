#%%
import torch
import os
import sys
import numpy as np
root_dir_name = 'Blender'
root_dir_path = os.path.abspath(__file__).split(root_dir_name)[0] + os.sep + root_dir_name
os.chdir(root_dir_path)
sys.path.extend([os.path.join(root_dir_path, dir) for dir in os.listdir(root_dir_path)])
sys.path.append(os.getcwd())        
sys.path.append(os.path.join(os.curdir, r"model_training\\utilities"))
from category_information import category_information

import visualization_and_validation.detect_uncertainty_areas as detect_uncertainty_areas
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
    correct_labels = 0
    for ground_truth_cluster, predict_cluster in overlap_indexes_dict.items():
        
        # check if the array is empty if so continue
        if len(predict_cluster) == 0:
            continue
        
        if ground_truth_labels[ground_truth_cluster] in predict_labels[predict_cluster]:
            correct_hypothesis += 1
        if ground_truth_labels[ground_truth_cluster] == predict_labels[predict_cluster][0]:
            correct_labels += 1
            
    if len(ground_truth_labels) == 0:
        return 1, 0, 1, 1
        
    num_predictions = len(predict_labels)
    hyp_true_positives = correct_hypothesis
    
    hyp_false_positives = num_predictions - hyp_true_positives
    hyp_false_negatives = len(ground_truth_labels) - hyp_true_positives

    hyp_f1 = hyp_true_positives / (hyp_true_positives + 0.5 * (hyp_false_positives + hyp_false_negatives))
    hyp_TPR = hyp_true_positives / (hyp_true_positives + hyp_false_negatives)
    hyp_false_positive_rate = 0 if hyp_false_positives == 0 else hyp_false_positives / (hyp_false_positives + hyp_true_positives)
 
    
    true_positive_rate = correct_labels / len(predict_labels) if len(predict_labels) > 0 else 0
    
    
    
    
    return  hyp_TPR, hyp_false_positive_rate ,hyp_f1,  true_positive_rate

def calculate_mask_only_IoU(pred_masks, ground_truth_mask, pred_scores, box_threshold, mask_threshold = 0.1, ground_truth_labels=None, pred_labels=None, ignore_classes = None):
    """
    Calculate the intersection over union metric for two masks.

    Args:
        pred_masks (numpy.ndarray): The prediction mask.
        ground_truth_mask (numpy.ndarray): The ground truth mask.
        box_threshold (float): The threshold for the bounding box. (NOT FOR THE MASK)

    Returns:
        float: The intersection over union metric.
"""
    if ignore_classes is not None:
        remove_indexes_pred = np.where(np.isin(pred_labels, ignore_classes))
        remove_indexes_ground_truth = np.where(np.isin(ground_truth_labels, ignore_classes))
        pred_masks = np.delete(pred_masks, remove_indexes_pred, axis=0)
        pred_scores = np.delete(pred_scores, remove_indexes_pred, axis=0)
        ground_truth_mask = np.delete(ground_truth_mask, remove_indexes_ground_truth, axis=0)
        

    pass_threshold_indices_labels = np.where(pred_scores > box_threshold)

    pred_masks = pred_masks[pass_threshold_indices_labels]
    
    # set the masks to boolean values so they can be used for the intersection over union
    pred_masks = np.sum(pred_masks, axis=(0,1)) > mask_threshold
    pred_masks[pred_masks > 0] = 1
    ground_truth_mask = np.sum(ground_truth_mask, axis=0) > 0
    ground_truth_mask[ground_truth_mask > 0] = True  
    
    intersection = np.sum(pred_masks & ground_truth_mask)
    union = np.sum(pred_masks | ground_truth_mask)
    if intersection == 0 and union == 0:
        return 1, pred_masks
    # plt.imshow(pred_masks)
    # plt.show()
    # plt.imshow(ground_truth_mask)
    # plt.show()
    return intersection / union, pred_masks
        



# if __name__ == "__main__":
def run_correct_hypothesis_rate(data_path=None, model_path=None, device=None, uncertainty_threshold=0.2, mask_threshold=0.5, nr_of_images=None, box_threshold=0.5, visualize_images=False, ignore_classes_in_mask = None):
    data = detect_uncertainty_areas.load_data(data_path)
    model = detect_uncertainty_areas.load_model(model_path, num_classes= len(category_information), device=device)

    if nr_of_images is None:
        nr_of_images = len(data)
    hyp_TPR_array = np.zeros(nr_of_images)
    hyp_false_positive_rate_array = np.zeros(nr_of_images)
    hyp_f1_array = np.zeros(nr_of_images)
    mask_IoU = np.zeros(nr_of_images)
    true_positive_rate_array = np.zeros(nr_of_images)
    
    
    for i in range(nr_of_images):
        if i % 10 == 0:
            print(f"Image {i} of {nr_of_images}", end='\r')
        prediction, pred_boxes, pred_labels, pred_masks, pred_img, pred_scores = detect_uncertainty_areas.run_model(model, data, i, device, box_threshold)
        
        uncertain_area_boxes, labels_incluster, filled_bb = detect_uncertainty_areas.find_area_of_uncertainty(pred_boxes, 
            pred_masks, pred_labels, uncertainty_threshold, show_overlap=False)

        ground_truth_boxes,ground_truth_masks, ground_truth_labels = get_ground_truth(data, i)
        overlap_indexes = find_overlap_indexes(filled_bb, ground_truth_boxes)
        
        hyp_TPR_array[i],hyp_false_positive_rate_array[i], hyp_f1_array[i], true_positive_rate_array[i]  = calculate_correct_hypothesis_rate(overlap_indexes,
            ground_truth_labels,pred_labels[0].cpu().numpy())

        mask_IoU[i], scored_pred_mask = calculate_mask_only_IoU(pred_masks[0].cpu().numpy(), ground_truth_masks, 
            pred_scores[0].cpu().numpy(), box_threshold,mask_threshold=mask_threshold, ground_truth_labels=ground_truth_labels, pred_labels=pred_labels[0].cpu().numpy(),
            ignore_classes = ignore_classes_in_mask)
        
        if visualize_images:           
            import cv2 
            plt.figure(figsize=(15, 5))
            #  swap black and white to make the image more visible
            pred_img = (255.0 * (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min())).to(torch.uint8)
            pred_img = pred_img.cpu().numpy()
            pred_img = np.transpose(pred_img, (1, 2, 0))
            maskwhite = cv2.inRange(pred_img, (180, 180, 180), (255, 255, 255))
            maskblack = cv2.inRange(pred_img, (0, 0, 0), (180, 180, 180))
            pred_img[maskwhite > 0] = [0, 0, 0]
            pred_img[maskblack > 0] = [255, 255, 255]

            
            plt.figure(figsize=(15, 5))
            plt.imshow(pred_img)
            plt.axis('off')

            plt.figure(figsize=(15, 5))
            plt.imshow(np.sum(ground_truth_masks,axis=0))
            plt.axis('off')

            plt.figure(figsize=(15, 5))
            plt.imshow(scored_pred_mask)
            plt.axis('off')

            plt.show()
            # plt.figure(figsize=(15, 5))
            
            # plt.subplot(1, 3, 1)
            # plt.title("Input")
            # plt.imshow(pred_img.permute(1, 2, 0))

            # plt.subplot(1, 3, 2)
            # plt.title("Ground truth")
            # plt.imshow(np.sum(ground_truth_masks,axis=0))
            
            # plt.subplot(1, 3, 3)
            # plt.title(f"Prediction")
            # plt.imshow(scored_pred_mask)
            # plt.show()

    model_name = os.path.split(model_path)[-2]
    hyp_miscro_avg_true_positive_rate = np.sum(hyp_TPR_array) / nr_of_images
    hyp_miscro_avg_false_positive_rate = np.sum(hyp_false_positive_rate_array) / nr_of_images
    hyp_miscro_avg_f1 = np.sum(hyp_f1_array) / nr_of_images
    mask_IoU = np.nansum(mask_IoU)/nr_of_images
    accuracy = np.nansum(hyp_TPR_array) / nr_of_images
    True_positive_rate = np.nansum(true_positive_rate_array) / nr_of_images

    
    print(f"for model {model_name}")
    print(f"for data {data_path.split(os.sep)[-1]}")
    print(f"Hyp True Positive Rate: {hyp_miscro_avg_true_positive_rate}")
    print(f"Hyp False positive rate: {hyp_miscro_avg_false_positive_rate}")
    print(f"F1 score: {hyp_miscro_avg_f1}")
    print(f"Average hyp Mask IoU: {mask_IoU}")
    print(f"True positive rate: {True_positive_rate}")

    return hyp_miscro_avg_true_positive_rate, hyp_miscro_avg_false_positive_rate, hyp_miscro_avg_f1, mask_IoU

def randomized_paramater_search(data_path, model_path, device, uncertainty_thresholds, mask_thresholds, box_thresholds, visualize_images=False, nr_of_iterations=10, optimize_for='f1',nr_of_images_to_evaluate_on = None):
    # best_f1 = -1
    best_uncertainty_threshold = -1
    best_mask_threshold = -1
    best_box_threshold = -1
    best_metric = -1
    for i in range(nr_of_iterations):
        uncertainty_threshold = np.random.uniform(uncertainty_thresholds[0], uncertainty_thresholds[1])
        mask_threshold = np.random.uniform(mask_thresholds[0], mask_thresholds[1])
        box_threshold = np.random.uniform(box_thresholds[0], box_thresholds[1]) 

        print(f"Iteration {i+1} of {nr_of_iterations}")
        true_positive_rate,false_positive,f1,mask_IoU = run_correct_hypothesis_rate(data_path=data_path, 
        model_path=model_path, device=device, uncertainty_threshold=uncertainty_threshold, mask_threshold=mask_threshold, 
        box_threshold=box_threshold, visualize_images=visualize_images, nr_of_images=nr_of_images_to_evaluate_on)
        if optimize_for == 'true_positive':
            metric = true_positive_rate
        elif optimize_for == 'false_positive':
            metric = false_positive
        elif optimize_for == 'f1':
            metric = f1
        elif optimize_for == 'mask_IoU':
            metric = mask_IoU
        else:
            raise ValueError("Invalid optimize_for parameter")
        if metric > best_metric:
            best_metric = metric
            final_IoU_score= mask_IoU
            final_f1_score = f1
            final_true_positive_rate = true_positive_rate
            final_false_positive_rate = false_positive
            
            best_uncertainty_threshold = uncertainty_threshold
            best_mask_threshold = mask_threshold
            best_box_threshold = box_threshold
    print (f"Best {optimize_for}: {best_metric}")
    print (f"Best uncertainty threshold: {best_uncertainty_threshold}")
    print (f"Best mask threshold: {best_mask_threshold}")
    print (f"Best box threshold: {best_box_threshold}")
    print(f"Final f1 score: {final_f1_score}")
    print(f"Final true positive rate: {final_true_positive_rate}")
    print(f"Final false positive rate: {final_false_positive_rate}")
    print(f"Final mask IoU: {final_IoU_score}")
    
    return best_uncertainty_threshold, best_mask_threshold, best_box_threshold, best_metric, final_f1_score, final_true_positive_rate, final_false_positive_rate, final_IoU_score
    
# %%
if __name__ == "__main__":
    uncertainty_threshold = 0.5
    mask_threshold = 0.5
    box_threshold = 0.5
    
    visualize_images = True
    nr_of_images = None
    # data_path = r'C:\Users\pimde\OneDrive\thesis\Blender\data\test\same_height_no_walls_no_tables_no_object_shift_big\[]'
    # data_path = r'real_world_data\Real_world_data_V3'
    # data_path = r'C:\Users\pimde\OneDrive\thesis\Blender\data\test\Customized_dataset\[]'
    data_path = r'C:\Users\pimde\OneDrive\thesis\Blender\data\test\finaltestset\[]'
    # model_path = r'C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\varying_height_no_walls_no_object_shift_big_varying_model_v3\weights.pth'
    model_path = r"C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\!Customized_model\weights.pth"
    # model_path = r"C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\ablation_v3_models\[low freq noise variance]\weights.pth"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ignore_classes_in_mask = [9]
    ignore_classes_in_mask = None

#%%
if __name__ == "__main__":
    # run once:
    _,_,f1, _ = run_correct_hypothesis_rate(data_path=data_path, model_path=model_path, device=device, uncertainty_threshold=uncertainty_threshold, mask_threshold=mask_threshold, box_threshold=box_threshold, visualize_images=visualize_images, nr_of_images= nr_of_images, ignore_classes_in_mask=ignore_classes_in_mask)
#%%
    
if __name__ == "__main__":
    # hyperparameters
    uncertainty_thresholds = [0.2, 0.8]
    mask_thresholds = [0.2, 0.8]
    box_thresholds = [0.2, 0.8]
    visualize_images = False
    nr_of_images = 200
    
    nr_of_iterations = 100
    optimize_for = 'mask_IoU'
    # run the randomized search
    best_uncertainty_threshold, best_mask_threshold, best_box_threshold, best_metric, final_f1_score, final_true_positive_rate, final_false_positive_rate,final_IoU_score = randomized_paramater_search(
        data_path, model_path, device, uncertainty_thresholds, mask_thresholds, box_thresholds, 
        visualize_images, nr_of_iterations, optimize_for= optimize_for, nr_of_images_to_evaluate_on = nr_of_images)
# %%
