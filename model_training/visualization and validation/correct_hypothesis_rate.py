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

from visualization and validation 