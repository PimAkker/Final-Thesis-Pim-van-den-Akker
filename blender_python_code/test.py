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

place_class.finalize()