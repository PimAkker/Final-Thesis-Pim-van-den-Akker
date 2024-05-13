import os
import sys


# ensure we are in the correct directory
root_dir_name = 'Blender'
root_dir_path = os.path.abspath(__file__).split(root_dir_name)[0] + root_dir_name
os.chdir(root_dir_path)
sys.path.extend([os.path.join(root_dir_path, dir) for dir in os.listdir(root_dir_path)])
from generate_dataset import generate_dataset
from category_information import category_information
import numpy as np

nr_of_images = 10
overwrite_data = False
empty_folders = True
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
raytrace_modifiers = {"high freq noise variance": (0.04, 0.1), 
                      "low freq noise variance": (0, 0.22),
                      "lidar block size":(0.15,0.25),
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

generate_dataset(nr_of_images=nr_of_images, 
                 overwrite_data=overwrite_data,
                 empty_folders=empty_folders, 
                 minimum_overlap_percentage_for_visible=minimum_overlap_percentage_for_visible, 
                 objects_to_add_percentage=objects_to_add_percentage, 
                 objects_to_remove_percentage=objects_to_remove_percentage, 
                 object_to_move_percentage=object_to_move_percentage,
                 force_object_visibility=force_object_visibility, 
                 max_shift_distance=max_shift_distance, 
                 walls_modifiers=walls_modifiers, 
                 chairs_modifiers=chairs_modifiers, 
                 round_table_modifiers=round_table_modifiers,
                 raytrace_modifiers=raytrace_modifiers, 
                 set_colors=set_colors)