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
import numpy as np

from  model_training.utilities.coco_eval import *
from  model_training.utilities.engine import *
from  model_training.utilities.utils import *
from  model_training.utilities.transforms import *
from  model_training.utilities.dataloader import get_transform, LoadDataset, get_model_instance_segmentation
import matplotlib.pyplot as plt
import sys

import os
from model_training.utilities.engine import  evaluate
import model_training.utilities.utils
from category_information import category_information
# force reload the module
import importlib
importlib.reload(model_training.utilities.utils)

#%%
if __name__ == '__main__':
    
    """NOTE strangely,sometimes, this function doesn't work for some reason until you run it in the debugger one time  ¯\_(ツ)_/¯"""
    
    data_to_test_on = r'real_world_data\Real_world_data_V3'
    # data_to_test_on = r"C:\Users\pimde\OneDrive\thesis\Blender\data\test\varying_heights\[]"
    num_classes = len(category_information)
    
    

    weights_load_path = r"C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\same_height_model\weights.pth"
    
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    dataset_test = LoadDataset(data_to_test_on, get_transform(train=False), ignore_indexes=2)

    
    percentage_of_dataset_to_use = 1
    dataset_test = torch.utils.data.Subset(dataset_test, range(int(len(dataset_test) * percentage_of_dataset_to_use)))
    
    precision_recall_dict = {}
    
    
    # num_samples = int(len(dataset_test) * percentage_of_dataset_to_use)
    # indices = torch.randperm(len(dataset_test))[:num_samples]
    # dataset_test = torch.utils.data.Subset(dataset_test, indices)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn
    )


    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(weights_load_path))

    # move model to the right device
    model.to(device)
        
    eval_output = evaluate(model, data_loader_test, device=device)
    
    eval_output.eval_imgs
    bbox_IoU_array = eval_output.coco_eval['bbox'].stats 
    segm_IoU_array = eval_output.coco_eval['segm'].stats

    bbox_IoU_array[bbox_IoU_array == -1] = np.nan
    segm_IoU_array[segm_IoU_array == -1] = np.nan
    
    mean_avg_precision_box = round(np.nanmean(bbox_IoU_array[:6]), 6)
    mean_avg_recall_box = round(np.nanmean(bbox_IoU_array[6:]), 6)
    mean_avg_precision_segm = round(np.nanmean(segm_IoU_array[:6]), 6)
    mean_avg_recall_segm = round(np.nanmean(segm_IoU_array[6:]), 6)

    precision_recall_dict["Combined"] = {
        'mean_avg_precision_box': mean_avg_precision_box,
        'mean_avg_recall_box': mean_avg_recall_box,
        'mean_avg_precision_segm': mean_avg_precision_segm,
        'mean_avg_recall_segm': mean_avg_recall_segm
    }

    print(f"mean average precision (box)   {mean_avg_precision_box}")
    print(f"mean average recall    (box)   {mean_avg_recall_box}")
    print(f"mean average precision (segm)  {mean_avg_precision_segm}")
    print(f"mean average recall    (segm)  {mean_avg_recall_segm}")
    
    segm_IoU_array = eval_output.coco_eval['segm'].stats

#%%

#NOTE is very slow, has to run entire thing for each mode

if __name__ == '__main__': 
    
    cat_inv = {v: k for k, v in category_information.items()}
    for iou_type in eval_output.coco_eval.keys():
        print(f"\n iou type: {iou_type}\n" )
        
        cat_Ids = [1, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        # cat_Ids = [9]

        for id in cat_Ids:
            # flip value and key
            
            
            name = cat_inv[id]
            print("-----------------------------------")
            IoU_list_bold = []
            for id_temp in cat_Ids:
                if id_temp == id:
                    IoU_list_bold.append(f"\033[1m\033[91m{id_temp}\033[0m")
                else:
                    IoU_list_bold.append(f"{id_temp}")
            print(f"Eval for {name}, id: {', '.join(IoU_list_bold)} for iou_type {iou_type} out of {', '.join(eval_output.coco_eval.keys())}")
            print("-----------------------------------")
            
            eval_output  = evaluate(model, data_loader_test, device=device, catIDs=[id])
        
            # eval_output.coco_eval[iou_type].summarize()
            bbox_IoU_array = eval_output.coco_eval['bbox'].stats 
            segm_IoU_array = eval_output.coco_eval['segm'].stats
            # when there are no detections, the array is filled with nan
            bbox_IoU_array[bbox_IoU_array == -1] = np.nan
            segm_IoU_array[segm_IoU_array == -1] = np.nan
            
            
            mean_avg_precision_box = round(np.nanmean(bbox_IoU_array[:6]), 6)
            mean_avg_recall_box = round(np.nanmean(bbox_IoU_array[6:]), 6)
            mean_avg_precision_segm = round(np.nanmean(segm_IoU_array[:6]), 6)
            mean_avg_recall_segm = round(np.nanmean(segm_IoU_array[6:]), 6)

            precision_recall_dict[name] = {
                'mean_avg_precision_box': mean_avg_precision_box,
                'mean_avg_recall_box': mean_avg_recall_box,
                'mean_avg_precision_segm': mean_avg_precision_segm,
                'mean_avg_recall_segm': mean_avg_recall_segm
            }

            print(f"mean average precision (box)   {mean_avg_precision_box}")
            print(f"mean average recall    (box)   {mean_avg_recall_box}")
            print(f"mean average precision (segm)  {mean_avg_precision_segm}")
            print(f"mean average recall    (segm)  {mean_avg_recall_segm}")
    
            

# %%
if __name__ == '__main__': 
    plt.figure(figsize=(10, 10))

    # Subplot 1: mean average precision (box)
    plt.subplot(2, 2, 1)
    plt.bar(range(len(precision_recall_dict)), [v['mean_avg_precision_box'] for v in precision_recall_dict.values()])
    plt.title('Mean Average Precision (Box)')
    plt.xticks(range(len(precision_recall_dict)), precision_recall_dict.keys(), rotation=45, ha='right')

    # Subplot 2: mean average recall (box)
    plt.subplot(2, 2, 2)
    plt.bar(range(len(precision_recall_dict)), [v['mean_avg_recall_box'] for v in precision_recall_dict.values()])
    plt.title('Mean Average Recall (Box)')
    plt.xticks(range(len(precision_recall_dict)), precision_recall_dict.keys(), rotation=45, ha='right')

    # Subplot 3: mean average precision (segm)
    plt.subplot(2, 2, 3)
    plt.bar(range(len(precision_recall_dict)), [v['mean_avg_precision_segm'] for v in precision_recall_dict.values()])
    plt.title('Mean Average Precision (Segm)')
    plt.xticks(range(len(precision_recall_dict)), precision_recall_dict.keys(), rotation=45, ha='right')

    # Subplot 4: mean average recall (segm)
    plt.subplot(2, 2, 4)
    plt.bar(range(len(precision_recall_dict)), [v['mean_avg_recall_segm'] for v in precision_recall_dict.values()])
    plt.title('Mean Average Recall (Segm)')
    plt.xticks(range(len(precision_recall_dict)), precision_recall_dict.keys(), rotation=45, ha='right')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

# %%
