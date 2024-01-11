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

# place_class = object_placement_utils.object_placement(delete_duplicates=False)

# load image
cv2.imread("blender_python_code\\data\\-inst-Prior_inst.png")

pass

