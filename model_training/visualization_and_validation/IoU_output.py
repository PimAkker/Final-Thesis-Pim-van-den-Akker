#%%

import torch
import os
import sys

root_dir_name = 'Blender'
root_dir_path = os.path.abspath(__file__).split(root_dir_name)[0] + root_dir_name
os.chdir(root_dir_path)
sys.path.extend([os.path.join(root_dir_path, dir) for dir in os.listdir(root_dir_path)])
sys.path.append(os.getcwd())

        
sys.path.append(os.path.join(os.curdir, r"model_training\\utilities"))
sys.path.append(os.getcwd())
import torch
from PIL import Image
import numpy as np
from  model_training.utilities.dataloader import get_transform, LoadDataset, get_model_instance_segmentation
from  model_training.utilities.coco_eval import *
from  model_training.utilities.engine import *
from  model_training.utilities.utils import *
from  model_training.utilities.transforms import *

import matplotlib.pyplot as plt
import sys

import os
from model_training.utilities.engine import  evaluate
import model_training.utilities.utils
from category_information import category_information
# force reload the module
import importlib
importlib.reload(model_training.utilities.utils)
import pandas as pd
#%%
if __name__ == '__main__':
    

    
    
        
    def get_category_ids(data_to_test_on, weights_load_path, cat_Ids, percentage_of_dataset_to_use=1):

        
    



        num_classes = len(category_information)
        # train on the GPU or on the CPU, if a GPU is not available
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # device = torch.device('cpu')
        dataset_test = LoadDataset(data_to_test_on, get_transform(train=False))

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
            'model_name': os.path.basename(os.path.dirname(weights_load_path)),
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


    #NOTE is very slow, has to run entire thing for each mode

        if __name__ == '__main__': 
            
            cat_inv = {v: k for k, v in category_information.items()}
            for iou_type in eval_output.coco_eval.keys():
                print(f"\n iou type: {iou_type}\n" )
                


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
                        'model_name': os.path.basename(os.path.dirname(weights_load_path)),
                        'mean_avg_precision_box': mean_avg_precision_box,
                        'mean_avg_recall_box': mean_avg_recall_box,
                        'mean_avg_precision_segm': mean_avg_precision_segm,
                        'mean_avg_recall_segm': mean_avg_recall_segm
                    }
                    
                        
            # set missing to Nan
            # print(f"Saved results to {os.path.join(result_folder, f'precision_recall_{name}.csv')}")
                

            print(f"mean average precision (box)   {mean_avg_precision_box}")
            print(f"mean average recall    (box)   {mean_avg_recall_box}")
            print(f"mean average precision (segm)  {mean_avg_precision_segm}")
            print(f"mean average recall    (segm)  {mean_avg_recall_segm}")
            
            return precision_recall_dict


            
            
           
            

                    
    def plot_precision_recall(precision_recall_dict, weights_load_path, result_folder=None ):           



        if __name__ == '__main__': 
            plt.figure(figsize=(10, 10))
            plt.title
            # Subplot 1: mean average precision (box)
            plt.subplot(2, 2, 1)
            plt.bar(range(len(precision_recall_dict)), [v['mean_avg_precision_box'] for v in precision_recall_dict.values()])
            plt.title('Mean Average Precision (Box)')
            plt.xticks(range(len(precision_recall_dict)), precision_recall_dict.keys(), rotation=45, ha='right')
            plt.yticks(np.arange(0, 1.1, 0.1))  # Set y-axis labels from 0 to 1 with a step of 0.1
            plt.grid(color='gray', linestyle='--', linewidth=0.5)

            # Subplot 2: mean average recall (box)
            plt.subplot(2, 2, 2)
            plt.bar(range(len(precision_recall_dict)), [v['mean_avg_recall_box'] for v in precision_recall_dict.values()])
            plt.title('Mean Average Recall (Box)')
            plt.xticks(range(len(precision_recall_dict)), precision_recall_dict.keys(), rotation=45, ha='right')
            plt.yticks(np.arange(0, 1.1, 0.1))  # Set y-axis labels from 0 to 1 with a step of 0.1
            plt.grid(color='gray', linestyle='--', linewidth=0.5)

            # Subplot 3: mean average precision (segm)
            plt.subplot(2, 2, 3)
            plt.bar(range(len(precision_recall_dict)), [v['mean_avg_precision_segm'] for v in precision_recall_dict.values()])
            plt.title('Mean Average Precision (Segm)')
            plt.xticks(range(len(precision_recall_dict)), precision_recall_dict.keys(), rotation=45, ha='right')
            plt.yticks(np.arange(0, 1.1, 0.1))  # Set y-axis labels from 0 to 1 with a step of 0.1
            plt.grid(color='gray', linestyle='--', linewidth=0.5)

            # Subplot 4: mean average recall (segm)
            plt.subplot(2, 2, 4)
            plt.bar(range(len(precision_recall_dict)), [v['mean_avg_recall_segm'] for v in precision_recall_dict.values()])
            plt.title('Mean Average Recall (Segm)')
            plt.xticks(range(len(precision_recall_dict)), precision_recall_dict.keys(), rotation=45, ha='right')
            plt.yticks(np.arange(0, 1.1, 0.1))  # Set y-axis labels from 0 to 1 with a step of 0.1
            plt.grid(color='gray', linestyle='--', linewidth=0.5)
            
            
            # Adjust layout to prevent overlap
            plt.tight_layout()
            
            if result_folder is not None:
                if not os.path.exists(result_folder):
                    os.makedirs(result_folder)
                plt.savefig(os.path.join(result_folder, f"{os.path.basename(os.path.dirname(weights_load_path))}.png"))
            plt.show()
    def save_results(precision_recall_list, result_folder):
        # if result_folder does not exist, create it
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
            
        dataframe = pd.DataFrame()
        for item in precision_recall_list:
            dataframe = pd.concat([dataframe, pd.DataFrame(item)])
            
        dataframe.to_csv(os.path.join(result_folder, "IoU_results.csv"))
        print(f"Saved results to {os.path.join(result_folder, 'IoU_results.csv')}")


# %%

if __name__ == '__main__':
    
    
    """NOTE strangely,sometimes, this function doesn't work for some reason until you run it in the debugger one time  ¯\_(ツ)_/¯"""
    
    cat_Ids = [4, 7, 8, 9, 13, 14, 15]
    # main_directory = r"C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info2"
    # weight_parent_folders = [f for f in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, f))]
    # weights_paths_list = []
    # for folder in weight_parent_folders:
    #     folder_path = os.path.join(main_directory, folder)
    #     filepaths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pth')]
    #     weights_paths_list.extend(filepaths)
    # precision_recall_list = []
    
    # for weights_path in weights_paths_list:
    #     precision_recall_dict = get_category_ids(r"C:\Users\pimde\OneDrive\thesis\Blender\real_world_data\Real_world_data_V3", weights_path, r"C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info2\results", list(category_information.keys()))
    #     precision_recall_list.append(precision_recall_dict)
    # plot_precision_recall(precision_recall_list, r"C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info2\results", weights_path)
    # save_results(precision_recall_list, r"C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info2\results")
    
    data_to_test_on = r"C:\Users\pimde\OneDrive\thesis\Blender\real_world_data\Real_world_data_V3"
    weights_path = r"C:\Users\pimde\OneDrive\thesis\Blender\data\Models\info\same_height_no_walls_no_tables_no_object_shift_model2\weights.pth"
    results_folder = r"/home/student/Pim/code/Blender/data/test/same_height_no_walls_no_tables_no_object_shift_model/IoU_info"
    results_dict = get_category_ids(data_to_test_on, weights_path,cat_Ids, percentage_of_dataset_to_use=1)
    plot_precision_recall(results_dict,  weights_path, result_folder = results_folder)
    
    

# %%

# %%
