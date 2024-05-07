#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

# set file as curdir
path = os.path.dirname(os.path.abspath(__file__))


# ensure we are in the correct directory
os.chdir(os.path.dirname(path))


# ensure we are in the correct directory
root_dir_name = 'Blender'
current_directory = os.getcwd().split(os.sep)
assert root_dir_name in current_directory, f"Current directory is {current_directory} and does not contain root dir name:  {root_dir_name}"
if current_directory[-1] != root_dir_name:
    # go down in the directory tree until the root directory is found
    while current_directory[-1] != root_dir_name:
        os.chdir("..")
        current_directory = os.getcwd().split(os.sep)


# add all the subdirectories to the path
dirs  = os.listdir()
root = os.getcwd()
for dir in dirs:
    sys.path.append(os.path.join(root, dir))
sys.path.append(os.getcwd())

import bpycv
import random
import numpy as np
import time
import custom_render_utils
import importlib
import blender_python_code.data_gen_utils as data_gen_utils
from category_information import category_information, class_factor
total_start_time = time.time()

import pandas as pd

masks_folder = r"data\Masks"
images_folder = r"data\Images"
metadata_folder = r"data\Metadata"
nr_of_images = 10
overwrite_data = False
empty_folders = True

# what percentage of the an instance should be visible to the lidar to be included in the input_image/mask
minimum_overlap_percentage_for_visible = 0.1



objects_to_add_percentage = 0.6666
objects_to_remove_percentage = 0.333
object_to_move_percentage = 0.5 # true object to move percentage = object_to_move_percentage * objects_to_add_percentage

force_object_visibility = ['walls'] # categorie(s) that should always be visible in the map

max_shift_distance =.5

obj_ids = category_information
walls_modifiers = {
    "wall width": (0.1, 0.3),
    "wall nr x": (0, 2),
    "wall nr y": (0, 2),
    "wall density": (0.7, 1),
    "seed": (0, 10000),
    "min door width": 0.7,
    "max door width": 1.3,
    "max wall randomness": (0, 0.3),
    "max door rotation": (np.pi/4, np.pi),
    "door density": (0.5, 1),
}

chair_size = (0.8, 1)
chairs_modifiers = {
    "chair width": chair_size,
    "chair length": chair_size,
    "leg width": (0.05, 0.1),
    "circular legs": np.random.choice([True, False]),
    "leg type": False,
}

table_size = (1, 1.5)
round_table_modifiers = {
    "table legs": (4, 6),
    "table x width": table_size,
    "table y width": table_size,
    "leg radius": (0.05, 0.12),
}

pillar_table_modifiers = {
    "width": (0.5, 1),
    "round/square": np.random.choice([True, False]),
}
raytrace_modifiers = {"high freq noise variance": (0, 0.03), 
                      "low freq noise variance": (0, 0.1),
                      "lidar block size":(0.08,0.12),
                      }



# these colors are used for the map not for the annotations
set_colors = {
            "walls": (255, 255, 255, 255),  # White
            "chairs display": (0, 255, 0, 255),  # Green
            "tables display": (0, 0, 255, 255),  # Blue
            "pillars display": (255, 255, 0, 255),  # Yellow
            "doors": (255, 0, 255, 255),  # Magenta
            "raytrace": (255, 0, 0, 255),  # Red
        }


# force a reload of object_placement_utils to help during development
importlib.reload(data_gen_utils)
importlib.reload(custom_render_utils)
importlib.reload(bpycv)


data_gen_utils.create_folders([masks_folder,images_folder, metadata_folder])
data_gen_utils.delete_folder_contents([masks_folder,images_folder, metadata_folder],empty_folders=empty_folders)
file_number = data_gen_utils.overwrite_data(images_folder, overwrite_data = overwrite_data)

pc = data_gen_utils.blender_object_placement(delete_duplicates=False)

# Create an empty pandas dataframe
instance_nr_df = pd.DataFrame(index=range(nr_of_images), columns=category_information.keys())


for object_name, color in set_colors.items():
    pc.set_object_color(object_name, color)

for i in np.arange(file_number, nr_of_images + file_number):
    print(f"Creating image {i}/{nr_of_images + file_number}")
    
    pc.delete_duplicates_func() #delete duplicates at the start to refresh the scene
    
    
    start_time = time.time()
    
    cru_class = custom_render_utils.custom_render_utils(image_id=str(i),
                                                        remove_intermediary_images=True,
                                                        minimum_render_overlap_percentage=minimum_overlap_percentage_for_visible, 
                                                        exclude_from_render=pc.original_objects,
                                                        force_map_visibility=force_object_visibility)
                                                        
    
    for modifier in list(walls_modifiers.keys()):
        pc.set_modifier("walls", modifier, walls_modifiers[modifier])
    for modifier in list(chairs_modifiers.keys()):
        pc.set_modifier("chair", modifier, chairs_modifiers[modifier])
    for modifier in list(round_table_modifiers.keys()):
        pc.set_modifier("round table", modifier, round_table_modifiers[modifier])
    for modifier in list(pillar_table_modifiers.keys()):   
        pc.set_modifier("pillar", modifier, pillar_table_modifiers[modifier])  
    for modifier in list(raytrace_modifiers.keys()):   
        pc.set_modifier("raytrace", modifier, raytrace_modifiers[modifier])        
          

    _, height, width, depth = pc.get_object_dims(object_name="walls")
    pc.place_objects(object_name="walls", inst_id=obj_ids["walls"], seperate_loose=False)
    pc.place_objects(object_name="doors", inst_id=obj_ids["doors"])
    pc.place_objects(object_name="chairs display", inst_id=obj_ids["chairs"])
    pc.place_objects(object_name="tables display", inst_id=obj_ids["tables"])
    pc.place_objects(object_name="pillars display", inst_id=obj_ids["pillars"])
    
    # Generate pointcloud image
    rand_pos_in_room = [random.gauss(0, width/6), random.gauss(0, depth/6), 0]
    pc.place_LiDAR(position=rand_pos_in_room)
   
    # move a percentage of objects down, this will move them out of the raytrace image
    # and will therefore not be seen in the pointcloud but will be seen in the mask and map
    # simulating that they are removed in real life but present on the map
    chairs_to_remove = pc.select_subset_of_objects(object_type_name="chairs display", selection_percentage=  objects_to_remove_percentage)
    tables_to_remove = pc.select_subset_of_objects(object_type_name="tables display", selection_percentage=  objects_to_remove_percentage)
    pillars_to_remove = pc.select_subset_of_objects(object_type_name="pillars display", selection_percentage=objects_to_remove_percentage)
    
    pc.set_object_id(obj_ids["chairs removed"], selection=chairs_to_remove)
    pc.set_object_id(obj_ids["tables removed"], selection=tables_to_remove)
    pc.set_object_id(obj_ids["pillars removed"], selection=pillars_to_remove)
    
    pc.move_objects_relative(chairs_to_remove,  [0, 0, -10])
    pc.move_objects_relative(tables_to_remove,  [0, 0, -10])
    pc.move_objects_relative(pillars_to_remove, [0, 0, -10])
    
    

    pc.hide_objects(chairs_to_remove+tables_to_remove+pillars_to_remove)
    
    
    pc.isolate_object("raytrace")
    # place_class.configure_camera(position=(0, 0, height/2))
    pc.set_modifier("raytrace.001", "visible surface switch", False)
    cru_class.simple_render(folder=images_folder, file_prefix="pointcloud", file_affix="")
    pc.set_modifier("raytrace.001", "visible surface switch", True)
    cru_class.simple_render(folder=images_folder, file_prefix="visible_region_mask", file_affix="")
    
    pc.unisolate()

    pc.delete_single_object("raytrace.001")
    
    # Here we remove a percentage of the objects and add a percentage of the objects from the map, they will still be 
    # visible in the pointcloud but not on the map, additionanly we will change the object id of the objects that are added
    chairs_to_add = pc.select_subset_of_objects(object_type_name="chairs display", selection_percentage=  objects_to_add_percentage)
    tables_to_add = pc.select_subset_of_objects(object_type_name="tables display", selection_percentage=  objects_to_add_percentage)
    pillars_to_add = pc.select_subset_of_objects(object_type_name="pillars display", selection_percentage=objects_to_add_percentage)
    
    pc.set_object_id(obj_ids["chairs new"],  selection=chairs_to_add)
    pc.set_object_id(obj_ids["tables new"],  selection=tables_to_add)
    pc.set_object_id(obj_ids["pillars new"], selection=pillars_to_add)
    

    chairs_to_move = list(np.random.choice(chairs_to_add, int(len(chairs_to_add) *    object_to_move_percentage)))
    tables_to_move = list(np.random.choice(tables_to_add, int(len(tables_to_add) *    object_to_move_percentage)))
    pillars_to_move = list(np.random.choice(pillars_to_add, int(len(pillars_to_add) * object_to_move_percentage)))

    # this move the objects in a random direction and shifts them down, this will make them invisible in the pointcloud
    relative_move = lambda : [random.uniform(-max_shift_distance, max_shift_distance) , random.uniform(-max_shift_distance, max_shift_distance), -10]

    moved_chairs =  pc.duplicate_move(objects_list=chairs_to_move, relative_position=relative_move())
    moved_tables =  pc.duplicate_move(objects_list=tables_to_move, relative_position=relative_move())
    moved_pillars = pc.duplicate_move(objects_list=pillars_to_move, relative_position=relative_move())

    pc.set_object_id(obj_ids["chairs removed"], selection=moved_chairs)
    pc.set_object_id(obj_ids["tables removed"], selection=moved_tables)
    pc.set_object_id(obj_ids["pillars removed"], selection=moved_pillars)
    

    # render the instance segmentation mask1
    pc.unhide_objects(chairs_to_move+tables_to_move+pillars_to_move)
    

    
    cru_class.render_data_semantic_map(folder=masks_folder, path_affix=f"mask", save_combined=False, save_rgb=False, save_inst=True)   
    pc.clean_up_materials()
    
    pc.delete_objects(object_list = chairs_to_add + tables_to_add + pillars_to_add)
    
    # # create the map and combine the poi
    # tcloud and map to a single image, creating the input for the model
    cru_class.simple_render(folder=images_folder, file_prefix="map", file_affix="")
    cru_class.combine_simple_renders(path=images_folder, file_nr=f"{i}", make_black_and_white=False)
    
    instance_nr_df = cru_class.update_dataframe_with_metadata(instance_nr_df)
    
    # pc.finalize()
    
    print(f"Time for this image: {time.time() - start_time}")

print(f"Done! Created {nr_of_images} images in {time.time() - total_start_time} seconds.")


instance_nr_df.to_csv(os.path.join(metadata_folder, "object_count_metadata.csv"), index=False)
data_gen_utils.save_metadata(metadata_path=metadata_folder,nr_of_images=nr_of_images, modifiers_list= [walls_modifiers, chairs_modifiers, round_table_modifiers, pillar_table_modifiers, raytrace_modifiers, set_colors],time_taken= time.time() - total_start_time)
