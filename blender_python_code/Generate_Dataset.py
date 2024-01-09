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

# force a reload of object_placement_utils to help during development
importlib.reload(object_placement_utils)
importlib.reload(custom_render_utils)

start_time = time.time()

place_class = object_placement_utils.object_placement(delete_duplicates=True)


height, width, depth = place_class.get_object_dims(object_name="Walls")
place_class.place_walls(inst_id=255)
place_class.place_doors(inst_id=150)
place_class.place_objects(object_name="Chairs display", inst_id=100)
place_class.place_objects(object_name="Tables display", inst_id=50)
place_class.place_objects(object_name="Pillars display", inst_id=10)




# Generate pointcloud image
place_class.place_raytrace()
place_class.isolate_object("raytrace")
place_class.configure_camera(position=(0,0,height/2))
custom_render_utils.simple_render(folder ="blender_python_code\\data",file_prefix ="pointcloud", file_affix="1")
place_class.unisolate()


# get map image
place_class.delete_object("raytrace.001")
custom_render_utils.simple_render(folder ="blender_python_code\\data",file_prefix ="map", file_affix="1")



custom_render_utils.render_data(folder ="blender_python_code\\data",  path_affix="1", save_rgb=True, save_inst=True, save_depth=True)   

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
    