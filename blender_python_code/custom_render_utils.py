import cv2
import bpy
import bpycv
import random
import numpy as np
import os
import sys
import time
# render image, instance annoatation and depth in one line code
# result["ycb_meta"] is 6d pose GT
def render_data(folder = r"data", path_affix="", save_rgb=True, save_inst=True, save_combined=True):
    
    # render image, instance annoatation and depth in one line code   
    result = bpycv.render_data()

    rgb_pathname = f"{folder}\\-rgb-{path_affix}.png"
    depth_pathname = f"{folder}\\-depth-{path_affix}_depth.png"
    inst_pathname = f"{folder}\\-inst-{path_affix}_inst.png"
    combined_pathname = f"{folder}\\-combined-{path_affix}.png" 
    
    if save_rgb:
        cv2.imwrite(rgb_pathname, result["image"][..., ::-1])
    
    if save_inst:
        # cv2.imwrite(inst_pathname, np.uint16(result["inst"]))
        # save numpy 
        np.save(f"{folder}\\-inst-{path_affix}_inst.npy", result["inst"])
        nr_of_inst= len(np.unique(result["inst"]))
        if nr_of_inst > 3:
            print(f"instance image has {nr_of_inst} unique values")
        elif nr_of_inst < 3:
            print(f"instance image is empty with {nr_of_inst} unique values")
    if save_combined:
        cv2.imwrite(combined_pathname, result.vis()[..., ::-1])
        

def simple_render(folder = r"data", file_prefix = "", file_affix=""):

    sys.path.append(os.path.dirname(os.path.realpath(__file__)))

    path= os.path.join(os.getcwd(), folder)
    path = os.path.join(path, file_prefix + file_affix+".png")
    bpy.context.scene.render.filepath= path
    output = bpy.ops.render.render(animation=False, write_still=True, use_viewport=False, layer='', scene='')
