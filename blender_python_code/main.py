import os
import sys


# ensure we are in the correct directory
import bpycv

# ensure we are in the correct directory
root_dir_name = 'Blender'
root_dir_path = os.path.abspath(__file__).split(root_dir_name)[0] + root_dir_name
os.chdir(root_dir_path)
sys.path.extend([os.path.join(root_dir_path, dir) for dir in os.listdir(root_dir_path)])

# add all the subdirectories to the path
dirs  = os.listdir()
root = os.getcwd()
for dir in dirs:
    sys.path.append(os.path.join(root, dir))
sys.path.append(os.getcwd())
import generate_dataset
from category_information import category_information
import numpy as np
from importlib import reload
import custom_render_utils
import blender_python_code.data_gen_utils as data_gen_utils
from category_information import category_information

# force a reload of object_placement_utils to help during development
reload(data_gen_utils)
reload(custom_render_utils)
reload(bpycv)
reload(generate_dataset)


# these are the per ablation run parameters
nr_of_images = 10
overwrite_data = False
empty_folders = True
minimum_overlap_percentage_for_visible = 0.1
objects_to_add_percentage = 0.6666
objects_to_remove_percentage = 0.333
object_to_move_percentage = 0.5 # true object to move percentage = object_to_move_percentage * objects_to_add_percentage
force_object_visibility = ['walls'] # categorie(s) that should always be visible in the map
max_shift_distance =.5
output_parent_folder = r"data/ablation/"


ablate_over_parameters = [{"wall width":'mean'}, 
                        #   {"wall nr x":0, "wall nr y":0}, 
                        #   {"low freq noise variance":'mean'},
                        #   {"wall density":'mean'},
                        #   {"min door width":1.0, "max door width":1.0},
                        #   {'max door rotation':0},
                        #   {'max wall randomness':0},
                        #   {'door density':'mean'},
                        #   {'chair width':'mean', 'chair length':'mean'},
                        #   {"table_legs":4},
                        #   {"table x width":'mean', "table y width":'mean'},
                        #   {"leg radius":'mean'},
                        #   {"high freq noise variance":0},
                        #   {"low freq noise variance":0}                  
                          ] # the parameters that will be ablated over one by one 

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
raytrace_modifiers = {"high freq noise variance": (0.08, 0.2), 
                      "low freq noise variance": (0, 0.44),
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

output_parent_folder = os.path.join(os.getcwd(), output_parent_folder)


for fixed_modifier in ablate_over_parameters:

    try: 
        gen = generate_dataset.generate_dataset(nr_of_images=nr_of_images, 
                        folder_name=os.path.join(output_parent_folder,f"{list(fixed_modifier)}"),
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
                        set_colors=set_colors, 
                        ablation_parameter=fixed_modifier)
        
    
    except Exception as e:
        # when an error occurs write it to the error log but continue with the next 
        # ablation parameter
        
        import datetime
        errorlog_location = os.path.join(os.getcwd(), "error_log.txt")
        with open(os.path.join(output_parent_folder,("error_log.txt")), "a") as f:
            f.write(f"Error in fixed modifier {fixed_modifier} \n")
            f.write(f"time: {datetime.datetime.now()} \n")
            f.write("\n")
            f.write(f"{e} \n")
            f.write("\n")
    