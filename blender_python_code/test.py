# #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys


# ensure we are in the correct directory
root_dir_name = 'Blender'
current_directory = os.getcwd().split("\\")
assert root_dir_name in current_directory, f"Current directory is {current_directory} and does not contain {root_dir_name}"
if current_directory[-1] != root_dir_name:
    # go down in the directory tree until the root directory is found
    while current_directory[-1] != root_dir_name:
        os.chdir("..")
        current_directory = os.getcwd().split("\\")


# add all the subdirectories to the path
dirs  = os.listdir()
root = os.getcwd()
for dir in dirs:
    sys.path.append(os.path.join(root, dir))
sys.path.append(os.getcwd())
import bpy
import bpycv
import random
import numpy as np
import time
import custom_render_utils
import importlib
import blender_python_code.data_gen_utils as data_gen_utils
from category_information import category_information
total_start_time = time.time()

timings = []
print('starting')
for i in range(100):
    tic = time.time()
    # save to temp directory
    bpy.context.scene.render.filepath= r"C:/tmp/"
    bpy.ops.render.render(animation=False, write_still=True, use_viewport=False, layer='', scene='')
    toc = time.time()
    timings.append(toc-tic)
print(f"Average time to render: {np.mean(timings)}")