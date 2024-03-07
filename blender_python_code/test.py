#!/usr/bin/env python3
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

# masks_folder = r"data\Masks"
# images_folder = r"data\Images"
# nr_of_images = 1
# overwrite_data = False
# empty_folders = True
# obj_ids = category_information

# place_class = data_gen_utils.blender_object_placement(delete_duplicates=False)
   
# bbox_raytrace, _, _, _ = place_class.get_object_dims("raytrace.001")

# objects_to_move = place_class.select_subset_of_objects(object_type_name="chairs display", selection_percentage=1, bbox=bbox_raytrace)
# place_class.move_objects_relative(objects_to_move, [0,0,-10])

obj = bpy.data.objects["walls.001"]

obj.to_mesh()


# path  = os.getcwd()
# bpy.context.scene.render.filepath= path
# path = os.path.join(path, "test-1.png")
# bpy.context.scene.render.filepath= path
# bpy.ops.render.render(animation=False, write_still=True, use_viewport=False, layer='', scene='')

# print(f"Total time: {time.time()-total_start_time}")