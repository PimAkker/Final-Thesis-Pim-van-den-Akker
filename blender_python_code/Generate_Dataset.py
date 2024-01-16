#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import bpy
import bpycv
import random
import numpy as np
import time

import custom_render_utils
import importlib
import object_placement_utils

walls_modifiers = {"Wall width":(0.05,0.2), 
                    "Wall Amount X": (0,10),
                    "Wall Amount Y": (0,10),
                    "Wall Density": (0,1),
                    "Seed": (0,10000000),
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
for _ in range(1):
     
    start_time = time.time()

    place_class = object_placement_utils.object_placement(delete_duplicates=False)
    
    for modifier in list(walls_modifiers.keys()):
        place_class.set_modifier("Walls", modifier, walls_modifiers[modifier])
    # Generate the room
    height, width, depth = place_class.get_object_dims(object_name="Walls")
    place_class.place_walls(inst_id=1)
    place_class.place_doors(inst_id=2)
    place_class.place_objects(object_name="Chairs display", inst_id=3)
    place_class.place_objects(object_name="Tables display", inst_id=4)
    place_class.place_objects(object_name="Pillars display", inst_id=5)

    custom_render_utils.render_data(folder ="blender_python_code\\data",  path_affix="True", save_rgb=True, save_inst=True)   


    # Generate pointcloud image
    place_class.place_raytrace()
    place_class.isolate_object("raytrace")
    place_class.configure_camera(position=(0,0,height/2))
    custom_render_utils.simple_render(folder ="blender_python_code\\data",file_prefix ="pointcloud", file_affix="1")
    place_class.unisolate()


    # get map image
    place_class.delete_single_object("raytrace.001")
    place_class.delete_random(object_type_name="Chairs display", delete_percentage=0.5)
    custom_render_utils.simple_render(folder ="blender_python_code\\data",file_prefix ="Map", file_affix="1")





    custom_render_utils.render_data(folder ="blender_python_code\\data",  path_affix="Prior", save_rgb=True, save_inst=True)   

    place_class.finalize()


    print(f"Total time: {time.time() - start_time}")

# try:
#     bpy.ops.object.convert(target='MESH')
# except:
#     pass


# for vert in bpy.data.objects['Table Placement'].data.vertices:
#     vert_loc = tuple(vert.co)
#     place_class = object_placement_utils.object_placement(delete_duplicates=False)
#     place_class.place_tables(inst_id=100,leg_nr_range=(3,10),size_range=(0.3,0.7),location=vert_loc)

