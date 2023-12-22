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

from custom_render_utils import render_data
import importlib
import object_placement_utils

# force a reload of object_placement_utils to help during development
importlib.reload(object_placement_utils)



place_class.blend_deselect_all()
bpy.data.objects['Table Placement'].select_set(True)
# place tables at the location of the vertices of the table placement object

try:
    bpy.ops.object.convert(target='MESH')
except:
    pass


for vert in bpy.data.objects['Table Placement'].data.vertices:
    vert_loc = tuple(vert.co)
    place_class = object_placement_utils.object_placement(delete_duplicates=False)
    place_class.place_tables(inst_id=100,leg_nr_range=(3,10),size_range=(0.3,0.7),location=vert_loc)
    