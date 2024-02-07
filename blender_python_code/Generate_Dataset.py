#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(r'C:\Users\pimde\OneDrive\thesis\Blender\blender_python_code')
os.chdir(r'C:\Users\pimde\OneDrive\thesis\Blender')
import bpy
import bpycv
import random
import numpy as np
import time

# print current directory
print(os.getcwd())


import custom_render_utils
import importlib
import object_placement_utils

import category_information

# open the blender file

total_start_time = time.time()
# change path to C:\Users\pimde\OneDrive\thesis\Blender



masks_folder = r"data\Masks"
images_folder = r"data\Images"
nr_of_images = 1000
overwrite_data = False
empty_folders = False
obj_ids = category_information.catogory_information
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
        place_class.set_modifier("Walls", modifier, walls_modifiers[modifier])
    # Generate the room
  
    _, height, width, depth = place_class.get_object_dims(object_name="Walls")
    place_class.place_walls(inst_id=obj_ids["Walls"])
    place_class.place_doors(inst_id=obj_ids["Doors"])
    place_class.place_objects(object_name="Chairs display", inst_id=obj_ids["Chairs"])
    place_class.place_objects(object_name="Tables display", inst_id=obj_ids["Tables"])
    place_class.place_objects(object_name="Pillars display", inst_id=obj_ids["Pillars"])
    
    # cru_class.render_data(folder =masks_folder,  path_affix=f"True{i}", save_rgb=False, save_combined=False, save_inst=True)   

    # Generate pointcloud image
    place_class.place_raytrace()
    bbox_raytrace, _, _, _ = place_class.get_object_dims("raytrace.001")
    place_class.isolate_object("raytrace")
    place_class.configure_camera(position=(0,0,height/2))
    cru_class.simple_render(folder= images_folder,file_prefix ="pointcloud", file_affix="")
    place_class.unisolate()

    place_class.delete_single_object("raytrace.001")
    
    objects_to_delete = place_class.select_subset_of_objects(object_type_name="Chairs display", delete_percentage=1,bbox=bbox_raytrace)
    place_class.set_object_id(obj_ids["Chairs removed"],selection=objects_to_delete)

    cru_class.render_data(folder =masks_folder,  path_affix=f"Mask{i}", save_combined=False,save_rgb=False, save_inst=True)   
    place_class.delete_objects(objects_to_delete)
    cru_class.simple_render(folder =r"data",file_prefix ="Map", file_affix="")

    place_class.finalize()
    
    cru_class.combine_simple_renders(path= images_folder, remove_originals = True, file_nr=f"{i}")
    print(f"Total time: {time.time() - start_time}")

    
print(f"Done! Created {nr_of_images} images in {time.time() - total_start_time} seconds.")