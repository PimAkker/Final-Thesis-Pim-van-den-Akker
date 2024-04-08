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
from random import uniform
import pandas as pd

masks_folder = r"data\Masks"
images_folder = r"data\Images"
metadata_folder = r"data\Metadata"
nr_of_images = 10
overwrite_data = False
empty_folders = True
render_only_visible_parts_of_map= True

objects_to_add_percentage = 0.6666
objects_to_remove_percentage = 0.333
object_to_move_percentage = 0.5 # true object to move percentage = object_to_move_percentage * objects_to_add_percentage


max_shift_distance =.5

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
raytrace_modifiers = {"high freq noise variance": (0, 0.03), 
                      "low freq noise variance": (0, 0.2),
                      "lidar block size":(0.05,0.07),
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
file_number = data_gen_utils.overwrite_data(images_folder,overwrite_data= overwrite_data)

place_class = data_gen_utils.blender_object_placement(delete_duplicates=False)

# Create an empty pandas dataframe
instance_nr_df = pd.DataFrame(index=range(nr_of_images), columns=category_information.keys())


for object_name, color in set_colors.items():
    place_class.set_object_color(object_name, color)

for i in np.arange(file_number, nr_of_images + file_number):
    print(f"Creating image {i}/{nr_of_images + file_number}")
    
    place_class.delete_duplicates_func() #delete duplicates at the start to refresh the scene
    
    start_time = time.time()
    
    cru_class = custom_render_utils.custom_render_utils(image_id=str(i),render_only_visible=render_only_visible_parts_of_map, exclude_from_render=place_class.original_objects)
    
    for modifier in list(walls_modifiers.keys()):
        place_class.set_modifier("walls", modifier, walls_modifiers[modifier])
    for modifier in list(chairs_modifiers.keys()):
        place_class.set_modifier("chair", modifier, chairs_modifiers[modifier])
    for modifier in list(round_table_modifiers.keys()):
        place_class.set_modifier("round table", modifier, round_table_modifiers[modifier])
    for modifier in list(pillar_table_modifiers.keys()):   
        place_class.set_modifier("pillar", modifier, pillar_table_modifiers[modifier])  
    for modifier in list(raytrace_modifiers.keys()):   
        place_class.set_modifier("raytrace", modifier, raytrace_modifiers[modifier])        
          

    _, height, width, depth = place_class.get_object_dims(object_name="walls")
    place_class.place_objects(object_name="walls", inst_id=obj_ids["walls"], seperate_loose=False)
    place_class.place_objects(object_name="doors", inst_id=obj_ids["doors"])
    place_class.place_objects(object_name="chairs display", inst_id=obj_ids["chairs"])
    place_class.place_objects(object_name="tables display", inst_id=obj_ids["tables"])
    place_class.place_objects(object_name="pillars display", inst_id=obj_ids["pillars"])
    
    # Generate pointcloud image
    # rand_pos_in_room = [random.gauss(0, width/6), random.gauss(0, depth/6), 0]
    place_class.place_raytrace(position=(-0.3,-3,0))
   
    # move a percentage of objects down, this will move them out of the raytrace image
    # and will therefore not be seen in the pointcloud but will be seen in the mask and map
    # simulating that they are1
    # removed in real life but present on the map
    chairs_to_remove = place_class.select_subset_of_objects(object_type_name="chairs display", selection_percentage=  objects_to_remove_percentage)
    tables_to_remove = place_class.select_subset_of_objects(object_type_name="tables display", selection_percentage=  objects_to_remove_percentage)
    pillars_to_remove = place_class.select_subset_of_objects(object_type_name="pillars display", selection_percentage=objects_to_remove_percentage)
    
    place_class.set_object_id(obj_ids["chairs removed"], selection=chairs_to_remove)
    place_class.set_object_id(obj_ids["tables removed"], selection=tables_to_remove)
    place_class.set_object_id(obj_ids["pillars removed"], selection=pillars_to_remove)
    
    place_class.move_objects_relative(chairs_to_remove,  [0, 0, -10])
    place_class.move_objects_relative(tables_to_remove,  [0, 0, -10])
    place_class.move_objects_relative(pillars_to_remove, [0, 0, -10])
    
    

    place_class.hide_objects(chairs_to_remove+tables_to_remove+pillars_to_remove)
    
    
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
    chairs_to_add = place_class.select_subset_of_objects(object_type_name="chairs display", selection_percentage=  objects_to_add_percentage)
    tables_to_add = place_class.select_subset_of_objects(object_type_name="tables display", selection_percentage=  objects_to_add_percentage)
    pillars_to_add = place_class.select_subset_of_objects(object_type_name="pillars display", selection_percentage=objects_to_add_percentage)
    
    place_class.set_object_id(obj_ids["chairs new"], selection=chairs_to_add)
    place_class.set_object_id(obj_ids["tables new"], selection=tables_to_add)
    place_class.set_object_id(obj_ids["pillars new"], selection=pillars_to_add)
    

    chairs_to_move = list(np.random.choice(chairs_to_add, int(len(chairs_to_add) *    object_to_move_percentage)))
    tables_to_move = list(np.random.choice(tables_to_add, int(len(tables_to_add) *    object_to_move_percentage)))
    pillars_to_move = list(np.random.choice(pillars_to_add, int(len(pillars_to_add) * object_to_move_percentage)))

    relative_move = lambda : [random.uniform(-max_shift_distance, max_shift_distance) , random.uniform(-max_shift_distance, max_shift_distance), -1]

    moved_chairs =  place_class.duplicate_move(objects_list=chairs_to_move, relative_position=relative_move())
    moved_tables =  place_class.duplicate_move(objects_list=tables_to_move, relative_position=relative_move())
    moved_pillars = place_class.duplicate_move(objects_list=pillars_to_move, relative_position=relative_move())

    place_class.set_object_id(obj_ids["chairs removed"], selection=moved_chairs)
    place_class.set_object_id(obj_ids["tables removed"], selection=moved_tables)
    place_class.set_object_id(obj_ids["pillars removed"], selection=moved_pillars)
    

    # render the instance segmentation mask1
    place_class.unhide_objects(chairs_to_move+tables_to_move+pillars_to_move)
    
    cru_class.render_data_semantic_map(folder=masks_folder, path_affix=f"mask", save_combined=False, save_rgb=False, save_inst=True)   
    place_class.clean_up_materials()
    
    place_class.delete_objects(chairs_to_add)
    place_class.delete_objects(tables_to_add)
    place_class.delete_objects(pillars_to_add)
    
    # # create the map and combine the pointcloud and map to a single image, creating the input for the model
    cru_class.simple_render(folder=r"data", file_prefix="map", file_affix="")
    cru_class.combine_simple_renders(path=images_folder, file_nr=f"{i}", make_black_and_white=False)
    
    instance_nr_df = cru_class.update_dataframe_with_metadata(instance_nr_df)
    
    place_class.finalize()
    
    print(f"Time for this image: {time.time() - start_time}")

print(f"Done! Created {nr_of_images} images in {time.time() - total_start_time} seconds.")
instance_nr_df.to_csv(os.path.join(metadata_folder, "object_count_metadata.csv"), index=False)

metadata_file = os.path.join(metadata_folder, "metadata.txt")

# Open the metadata file in write mode
with open(metadata_file, "w") as f:
    f.write(f"This file contains the metadata for the generated dataset\n\n")
    f.write(f"This dataset was created on {time.ctime()}\n\n")
    f.write(f"Total number of images: {nr_of_images}\n\n")
    
    
    # Write the values of walls_modifiers
    f.write("Walls Modifiers:\n")
    for modifier, value in walls_modifiers.items():
        f.write(f"{modifier}: {value}\n")
    
    # Write the values of chairs_modifiers
    f.write("\nChairs Modifiers:\n")
    for modifier, value in chairs_modifiers.items():
        f.write(f"{modifier}: {value}\n")
    
    # Write the values of round_table_modifiers
    f.write("\nRound Table Modifiers:\n")
    for modifier, value in round_table_modifiers.items():
        f.write(f"{modifier}: {value}\n")
    
    # Write the values of pillar_table_modifiers
    f.write("\nPillar Table Modifiers:\n")
    for modifier, value in pillar_table_modifiers.items():
        f.write(f"{modifier}: {value}\n")
    
    # Write the values of raytrace_modifiers
    f.write("\nRaytrace Modifiers:\n")
    for modifier, value in raytrace_modifiers.items():
        f.write(f"{modifier}: {value}\n")
    
    # Write the values of set_colors
    f.write("\nSet Colors:\n")
    for object_name, color in set_colors.items():
        f.write(f"{object_name}: {color}\n")

# Print a message to indicate that the metadata file has been created
print(f"Metadata file created: {metadata_file}")
