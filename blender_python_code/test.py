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
import cv2
import custom_render_utils
import importlib
import object_placement_utils


# force a reload of object_placement_utils to help during development
importlib.reload(object_placement_utils)
importlib.reload(custom_render_utils)

# start_time = time.time()

try:
    for ob in bpy.context.selected_objects:
        ob.select = False
except:
    pass
try:
    bpy.ops.outliner.item_activate(deselect_all=True)
except:
    pass
try:
    bpy.ops.object.select_all(action='DESELECT')
except:
    pass   

# modifier = bpy.data.objects['Walls'].modifiers['GeometryNodes']
# for input in modifier.node_group.inputs:
#     print(f"Input {input.identifier} is named {input.type}")
place_class = object_placement_utils.object_placement(delete_duplicates=False)

geometry_nodes = bpy.data.objects["Walls"].modifiers['GeometryNodes']

modifier_identifier_list = [input.identifier for input in geometry_nodes.node_group.inputs]

modifier_name_list = [input.name for input in geometry_nodes.node_group.inputs]
modifier_name_list = [name.lower() for name in modifier_name_list][1:]

# for modifier in modifier_name_list:

place_class.set_modifier("Walls", "seed", 3.11)



# place_class.place_walls(inst_id=1)
# place_class.finalize()


#%%
import PIL
import numpy as np
# open image
img = PIL.Image.open(r"C:\Users\pimde\OneDrive\thesis\Blender\blender_python_code\data\Images\Map1.png")
numpy_img = np.array(img)
print(numpy_img[0])


# %%
