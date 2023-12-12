#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import cv2
import bpy
import bpycv
import random
import numpy as np

from object_placement_utils import object_placement
from custom_render_utils import render_data

nr_of_renders = 4

place_class = object_placement(delete_duplicates=True)
place_class.place_room()
place_class.place_tables(num_tables=5,inst_id=100,leg_nr_range=(4,4))

render_data(folder =r"data/", path_name="demo", path_affix="", save_rgb=True, save_inst=True, save_depth=True)    

# run this at the end of the script
place_class.finalize()