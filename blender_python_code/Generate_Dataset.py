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
import object_placement_utils
from category_information import category_information
total_start_time = time.time()

masks_folder = r"data\Masks"
images_folder = r"data\Images"
nr_of_images = 1
overwrite_data = False
empty_folders = True
obj_ids = category_information
walls_modifiers = {"Wall width":(0.05,0.2), 
                    "Wall Amount X": (2,5),
                    "Wall Amount Y": (2,5),
                    "Wall Density": (0.3,0.9),
                    "Seed": (0,10000),
                    "Min door width": 0.3,
                    "Max door width": 1.5,
                    "Max wall randomness": (0,0.1),
                    "Max door rotation": (0,np.pi),
                    "Door density": (0.1,1),                    
                   }

# force a reload of object_placement_utils to help during development
importlib.reload(object_placement_utils)
importlib.reload(custom_render_utils)
importlib.reload(bpycv)

# if not overwrite data then get the highest file number and continue from there
file_number = 0
if not overwrite_data:
        
        file_names = os.listdir(images_folder)
    # split the files names by  "-"
        file_names = [file.split("-") for file in file_names]
    # get the file numbers
        if file_names != []:
            file_numbers = [int(file[-2]) for file in file_names]
            file_number = max(file_numbers)+1
if empty_folders:
    for folder in [masks_folder, images_folder]:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))


for i in np.arange(file_number,nr_of_images+file_number):
    print(f"Creating image {i}/{nr_of_images+file_number}")
    start_time = time.time()
    
    
    place_class = object_placement_utils.object_placement(delete_duplicates=False)
    cru_class = custom_render_utils.custom_render_utils(image_id = str(i))


    for modifier in list(walls_modifiers.keys()):
        place_class.set_modifier("walls", modifier, walls_modifiers[modifier])
    # Generate the room
  
    _, height, width, depth = place_class.get_object_dims(object_name="walls")
    place_class.place_walls(inst_id=obj_ids["walls"])
    place_class.place_doors(inst_id=obj_ids["doors"])
    place_class.place_objects(object_name="chairs display", inst_id=obj_ids["chairs"])
    place_class.place_objects(object_name="tables display", inst_id=obj_ids["tables"])
    place_class.place_objects(object_name="pillars display", inst_id=obj_ids["pillars"])
    
    # cru_class.render_data(folder =masks_folder,  path_affix=f"True{i}", save_rgb=False, save_combined=False, save_inst=True)   



    # Generate pointcloud image
    place_class.place_raytrace()
    bbox_raytrace, _, _, _ = place_class.get_object_dims("raytrace.001")
   
    # move a percentage of objects down, this will move them out of the raytrace image
    # and will therefore not be seen in the pointcloud but will be seen in the mask and map
    # simulating that they are removed in real life but present on the map
    objects_to_move = place_class.select_subset_of_objects(object_type_name="chairs display", selection_percentage=1, bbox=bbox_raytrace)
    place_class.move_objects_relative(objects_to_move, [0,0,-10])
    place_class.set_object_id(obj_ids["chairs removed"],selection=objects_to_move)
    
    place_class.isolate_object("raytrace")
    place_class.configure_camera(position=(0,0,height/2))
    cru_class.simple_render(folder= images_folder,file_prefix ="pointcloud", file_affix="")
    place_class.unisolate()

    # place_class.delete_single_object("raytrace.001")
    
    objects_to_delete = place_class.select_subset_of_objects(object_type_name="chairs display", selection_percentage=0.3,bbox=bbox_raytrace)
    place_class.set_object_id(obj_ids["chairs new"],selection=objects_to_delete)

    cru_class.render_data(folder =masks_folder,  path_affix=f"Mask{i}", save_combined=False,save_rgb=False, save_inst=True)   
    place_class.delete_objects(objects_to_delete)
    
    cru_class.simple_render(folder =r"data",file_prefix ="Map", file_affix="")

    place_class.finalize()
    
    cru_class.combine_simple_renders(path= images_folder, remove_originals = False, file_nr=f"{i}")
    print(f"Total time: {time.time() - start_time}")

    
print(f"Done! Created {nr_of_images} images in {time.time() - total_start_time} seconds.")