# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#%%
# import os
# import sys


# # ensure we are in the correct directory
# root_dir_name = 'Blender'
# current_directory = os.getcwd().split("\\")
# assert root_dir_name in current_directory, f"Current directory is {current_directory} and does not contain {root_dir_name}"
# if current_directory[-1] != root_dir_name:
#     # go down in the directory tree until the root directory is found
#     while current_directory[-1] != root_dir_name:
#         os.chdir("..")
#         current_directory = os.getcwd().split("\\")


# # add all the subdirectories to the path
# dirs  = os.listdir()
# root = os.getcwd()
# for dir in dirs:
#     sys.path.append(os.path.join(root, dir))
# sys.path.append(os.getcwd())
# import bpy
# import bpycv
# import random
# import numpy as np
# import time
# import custom_render_utils
# import importlib
# import blender_python_code.data_gen_utils as data_gen_utils
# from category_information import category_information
# total_start_time = time.time()


from PIL import Image


#%%
image_real = Image.open(r"C:\Users\pimde\OneDrive\thesis\Blender\real_world_data\Real_world_data_V2\Images\room1_7_1.png")
image_syn = Image.open(r"C:\Users\pimde\OneDrive\thesis\Blender\data\test\same_height_no_walls_v2\[]\Images\input-2-.png")
#%%
# show the two images side by side
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2)

axes[0].imshow(image_real)
axes[0].set_title('Real Image')

axes[1].imshow(image_syn)
axes[1].set_title('Synthetic Image')

plt.show()


# %%
