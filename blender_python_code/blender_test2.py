import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import bpy
import bpycv
import random
import numpy as np
import time
from object_placement_utils import object_placement
from custom_render_utils import render_data

place_class = object_placement(delete_duplicates=False)
place_class.place_room()
# place_class.place_tables(num_tables=5,inst_id=100)

place_class.place_raytrace(position=(1,2,0))
place_class.finalize()