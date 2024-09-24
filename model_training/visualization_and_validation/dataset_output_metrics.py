"""
This file import the data from the dataset and outputs the metrics of the dataset such as number of items, number of classes, number of items per class, etc. and some statistics about averages 

"""
#%%
# ensure we are in the correct directory
import os
import sys


# ensure we are in the correct directory
root_dir_name = 'Blender'
root_dir_path = os.path.abspath(__file__).split(root_dir_name)[0] + root_dir_name
os.chdir(root_dir_path)
sys.path.extend([os.path.join(root_dir_path, dir) for dir in os.listdir(root_dir_path)])

# add all the subdirectories to the path
dirs  = os.listdir()
root = os.getcwd()
for dir in dirs:
    sys.path.append(os.path.join(root, dir))
sys.path.append(os.getcwd())
from category_information import category_information
import matplotlib.pyplot as plt

import numpy as np
import matplotlib
def import_dataset(dataset_path):
    """
    Import all the masks from the dataset
    
    Args:
        dataset_path (str): The path to the dataset
        
    Returns:
        dataset (dict): The dataset
    """
    mask_paths = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.endswith(".npy")]
    masks = [np.load(mask_path) for mask_path in mask_paths]
    
    return masks

def count_instances(masks):
    """
    Count the number of instances per class
    
    Args:
        masks (list): The list of masks
        
    Returns:
        instance_counts (dict): The number of instances per class
    """
    instance_counts = {}
    for mask in masks:
        obj_ids = np.unique(mask)//1000
        for obj_id in obj_ids:
            if obj_id not in instance_counts:
                instance_counts[obj_id] = 0
            instance_counts[obj_id] += 1
            
    return instance_counts
def keys_to_names(instance_counts):
    """
    Convert the class keys to class names
    
    Args:
        instance_counts (dict): The number of instances per class
    """
    swapped_dict = {value: key for key, value in category_information.items()}
    swapped_dict = {value: key.capitalize() for key, value in category_information.items()}
    instance_counts = {swapped_dict[int(key)]: value for key, value in instance_counts.items()}
    return instance_counts

#%%
dataset_path = r"C:\Users\pimde\OneDrive\thesis\Blender\real_world_data\Real_world_data_V3\Masks"
masks = import_dataset(dataset_path)
instance_counts = count_instances(masks)
instance_counts = keys_to_names(instance_counts)
print(f"Number of items in dataset: {len(masks)}")
print(f"Number of classes in dataset: {len(instance_counts)}")
print(f"Number of items per class: {instance_counts}")
print(f"Average number of items per class: {np.mean(list(instance_counts.values()))}")
print(f"Standard deviation of number of items per class: {np.std(list(instance_counts.values()))}")
print(f"Median number of items per class: {np.median(list(instance_counts.values()))}")
print(f"Minimum number of items per class: {np.min(list(instance_counts.values()))}")

# make some bar charts 

matplotlib.rcParams['font.family'] = 'serif'

sorted_instance_counts = dict(sorted(instance_counts.items(), key=lambda x: x[1], reverse=True))

# remove the Background class
del sorted_instance_counts['Background']

plt.bar(sorted_instance_counts.keys(), sorted_instance_counts.values())
plt.xlabel("Class")
plt.ylabel("Number of items")
plt.xticks(rotation=45, ha='right')

plt.grid(axis='y', linestyle=':', alpha=0.5)  # Add subtle gridlines

plt.tight_layout()
plt.rcParams['font.family'] = 'serif'  # Set font to computer modern
plt.savefig('real_world_dataset_values.pdf')
plt.show()




# %%
