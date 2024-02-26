#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

# set file as curdir
path = os.path.dirname(os.path.abspath(__file__))
path = "\\".join(path.split("\\")[:-1])

os.chdir(path)


# ensure we are in the correct directory
root_dir_name = 'Blender'
current_directory = os.getcwd().split("\\")
assert root_dir_name in current_directory, f"Current directory is {current_directory} and does not contain root dir name:  {root_dir_name}"
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
from category_information import category_information, class_factor
total_start_time = time.time()

masks_folder = r"data\Masks"
images_folder = r"data\Images"
nr_of_images = 1
overwrite_data = False
empty_folders = True
render_only_visible_parts_of_map= True



obj_ids = category_information
walls_modifiers = {
    "Wall width": (0.05, 0.2),
    "Wall Amount X": (0, 5),
    "Wall Amount Y": (0, 5),
    "Wall Density": (0.5, 0.95),
    "Seed": (0, 10000),
    "Min door width": 0.7,
    "Max door width": 1.3,
    "Max wall randomness": (0, 0.3),
    "Max door rotation": (0, np.pi),
    "Door density": (0.1, 1),
}

chair_size = (0.4, 0.6)
chairs_modifiers = {
    "chair width": chair_size,
    "chair length": chair_size,
    "leg width": (0.05, 0.1),
    "circular legs": np.random.choice([True, False]),
    "leg type": False,
}

round_table_modifiers = {
    "table legs": (3, 5),
    "table x width": (0.5, 1.5),
    "table y width": (0.5, 1.5),
    "leg radius": (0.05, 0.1),
}

pillar_table_modifiers = {
    "width": (0.3, 1),
    "round/square": np.random.choice([True, False]),
}

# these colors are used for the map not for the annotations
set_colors = {
            "walls": (255, 0, 0, 255),  # Red
            "chairs display": (0, 255, 0, 255),  # Green
            "tables display": (0, 0, 255, 255),  # Blue
            "pillars display": (255, 255, 0, 255),  # Yellow
            "doors": (255, 0, 255, 255),  # Magenta
            "raytrace": (0, 255, 255, 255),  # Cyan
        }


# force a reload of object_placement_utils to help during development
importlib.reload(data_gen_utils)
importlib.reload(custom_render_utils)
importlib.reload(bpycv)


data_gen_utils.create_folders([masks_folder,images_folder])
data_gen_utils.delete_folder_contents(masks_folder,images_folder,empty_folders=empty_folders)
file_number = data_gen_utils.overwrite_data(images_folder,overwrite_data= overwrite_data)


place_class = data_gen_utils.blender_object_placement(delete_duplicates=False, class_multiplier=class_factor)

for object_name, color in set_colors.items():
    place_class.set_object_color(object_name, color)


for i in np.arange(file_number, nr_of_images + file_number):
    print(f"Creating image {i}/{nr_of_images + file_number}")
    
    place_class.delete_duplicates_func() #delete duplicates at the start to refresh the scene
    
    start_time = time.time()

    
    cru_class = custom_render_utils.custom_render_utils(image_id=str(i),render_only_visible=render_only_visible_parts_of_map)
    
    for modifier in list(walls_modifiers.keys()):
        place_class.set_modifier("walls", modifier, walls_modifiers[modifier])
    for modifier in list(chairs_modifiers.keys()):
        place_class.set_modifier("chair", modifier, chairs_modifiers[modifier])
    for modifier in list(round_table_modifiers.keys()):
        place_class.set_modifier("round table", modifier, round_table_modifiers[modifier])
    for modifier in list(pillar_table_modifiers.keys()):   
        place_class.set_modifier("pillar", modifier, pillar_table_modifiers[modifier])    

    _, height, width, depth = place_class.get_object_dims(object_name="walls")
    place_class.place_walls(inst_id=obj_ids["walls"])
    place_class.place_objects(object_name="doors", inst_id=obj_ids["doors"])
    place_class.place_objects(object_name="chairs display", inst_id=obj_ids["chairs"])
    place_class.place_objects(object_name="tables display", inst_id=obj_ids["tables"])
    place_class.place_objects(object_name="pillars display", inst_id=obj_ids["pillars"])
    
    
    cru_class.render_data(folder=masks_folder, path_affix=f"True{i}", save_rgb=False, save_combined=False, save_inst=True)   

    # Generate pointcloud image
    rand_pos_in_room = [random.gauss(0, width/6), random.gauss(0, depth/6), 0]
    place_class.place_raytrace()
    bbox_raytrace, _, _, _ = place_class.get_object_dims("raytrace.001")
   
    # move a percentage of objects down, this will move them out of the raytrace image
    # and will therefore not be seen in the pointcloud but will be seen in the mask and map
    # simulating that they are1
    # removed in real life but present on the map
    chairs_to_remove = place_class.select_subset_of_objects(object_type_name="chairs display", selection_percentage=0.3)
    tables_to_remove = place_class.select_subset_of_objects(object_type_name="tables display", selection_percentage=0.3)
    pillars_to_remove = place_class.select_subset_of_objects(object_type_name="pillars display", selection_percentage=0.3)
    
    place_class.move_objects_relative(chairs_to_remove, [0, 0, -10])
    place_class.move_objects_relative(tables_to_remove, [0, 0, -10])
    place_class.move_objects_relative(pillars_to_remove, [0, 0, -10])
    
    
    place_class.set_object_id(obj_ids["chairs removed"], selection=chairs_to_remove)
    place_class.set_object_id(obj_ids["tables removed"], selection=tables_to_remove)
    place_class.set_object_id(obj_ids["pillars removed"], selection=pillars_to_remove)
    

    
    place_class.isolate_object("raytrace")
    place_class.configure_camera(position=(0, 0, height/2))
    place_class.set_modifier("raytrace.001", "visible surface switch", False)
    cru_class.simple_render(folder=images_folder, file_prefix="pointcloud", file_affix="")
    place_class.set_modifier("raytrace.001", "visible surface switch", True)
    cru_class.simple_render(folder=images_folder, file_prefix="visible_region_mask", file_affix="")
    place_class.unisolate()

    place_class.delete_single_object("raytrace.001")
    
    # Here we remove a percentage of the objects and add a percentage of the objects from the map, they will still be 
    # visible in the pointcloud but not on the map, additionanly we will change the object id of the objects that are added
    chairs_to_remove = place_class.select_subset_of_objects(object_type_name="chairs display", selection_percentage=0.3)
    tables_to_remove = place_class.select_subset_of_objects(object_type_name="tables display", selection_percentage=0.3)
    pillars_to_remove = place_class.select_subset_of_objects(object_type_name="pillars display", selection_percentage=0.3)
    
    place_class.set_object_id(obj_ids["chairs new"], selection=chairs_to_remove)
    place_class.set_object_id(obj_ids["tables new"], selection=tables_to_remove)
    place_class.set_object_id(obj_ids["pillars new"], selection=pillars_to_remove)
    
    # render the instance segmentation mask
    cru_class.render_data(folder=masks_folder, path_affix=f"mask", save_combined=False, save_rgb=False, save_inst=True)   
    clean_up_start_time = time.time()
    place_class.clean_up_materials()
    print(f"Time for clean up: {time.time() - clean_up_start_time}")
    
    place_class.delete_objects(chairs_to_remove)
    place_class.delete_objects(tables_to_remove)
    place_class.delete_objects(pillars_to_remove)
    
    # create the map and combine the pointcloud and map to a single image, creating the input for the model
    cru_class.simple_render(folder=r"data", file_prefix="map", file_affix="")
    cru_class.combine_simple_renders(path=images_folder, file_nr=f"{i}", make_black_and_white=False)
    
    place_class.finalize()
    
    print(f"Time for this image: {time.time() - start_time}")

print(f"Done! Created {nr_of_images} images in {time.time() - total_start_time} seconds.")
