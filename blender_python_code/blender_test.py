#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Notice: When update demo.py: 
    1. Update README.md
    2. @diyer22 update the answer on stackexchange:
            https://blender.stackexchange.com/a/162746/86396 
"""

import cv2
import bpy
import bpycv
import random
import numpy as np

# remove all MESH objects
# [bpy.data.objects.remove(obj) for obj in bpy.data.objects if obj.type == "MESH"]

# for index in range(1, 20):
#     # create cube and sphere as instance at random location
#     location = [random.uniform(-2, 2) for _ in range(3)]
#     if index % 2:
#         bpy.ops.mesh.primitive_cube_add(size=0.5, location=location)
#         categories_id = 1
#     else:
#         bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=location)
#         categories_id = 2
#     obj = bpy.context.active_object
#     # set each instance a unique inst_id, which is used to generate instance annotation.
#     obj["inst_id"] = categories_id * 1000 + index


#select room object
bpy.data.objects['room'].select_set(True)
# set object as active
bpy.context.view_layer.objects.active = bpy.data.objects['room']

bpy.context.object.modifiers["GeometryNodes"]["Socket_6"] = np.random.randint(0, 100000)
obj = bpy.context.active_object

bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked":False, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(45.2641, 3.86619, -3.15708), "orient_type":'GLOBAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'FACE_NEAREST'}, "use_snap_project":True, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
bpy.ops.object.convert(target='MESH')
obj = bpy.context.active_object
obj["inst_id"] = 255

table_dict  = {}

for i in range(5):
    # deselect all
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['table square'].select_set(True)
    # set object as active
    bpy.context.view_layer.objects.active = bpy.data.objects['table square']
    rand_transform = (np.random.uniform(17, 40), np.random.uniform(7, -30), 0)
    print(rand_transform)
    bpy.ops.object.duplicate_move_linked(OBJECT_OT_duplicate={"linked":True, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":rand_transform, "orient_type":'GLOBAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'FACE_NEAREST'}, "use_snap_project":True, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
    bpy.ops.object.convert(target='MESH')
    obj = bpy.context.active_object
    obj["inst_id"] = 100
    
# do bpy.data.objects["table square"].data.copy() to each instance of table square
for obj in bpy.data.objects:
    if obj.name.startswith("table square"):
        obj.data = bpy.data.objects["table square"].data.copy()
    
    
    

# render image, instance annoatation and depth in one line code
# result["ycb_meta"] is 6d pose GT
result = bpycv.render_data()
# save result
cv2.imwrite(
    "demo-rgb.jpg", result["image"][..., ::-1]
)  # transfer RGB image to opencv's BGR

# save instance map as 16 bit png
# the value of each pixel represents the inst_id of the object to which the pixel belongs
cv2.imwrite("demo-inst.png", np.uint16(result["inst"]))

# convert depth units from meters to millimeters
depth_in_mm = result["depth"] * 1000
cv2.imwrite("demo-depth.png", np.uint16(depth_in_mm))  # save as 16bit png

# visualization inst_rgb_depth for human
cv2.imwrite("demo-vis(inst_rgb_depth).jpg", result.vis()[..., ::-1])

# delete all objects which are copied
[bpy.data.objects.remove(obj) for obj in bpy.data.objects if "." in obj.name]