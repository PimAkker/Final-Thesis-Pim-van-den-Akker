import cv2
import bpy
import bpycv
import random
import numpy as np
import os
import sys
import time


class custom_render_utils:
    def __init__(self, image_id = "0"):
        self.image_id = image_id
        self.file_type = ".jpg" # only change this when also changed in blender renderer
        self.image_path_list = []
        
    def render_data(self,folder = r"data", path_affix="", save_rgb=True, save_inst=True, save_combined=True):
        
        # render image, instance annoatation and depth
        result = bpycv.render_data()

        rgb_pathname =      f"{folder}\\rgb-{path_affix}-{self.image_id}-{self.file_type}"
        depth_pathname =    f"{folder}\\depth-{path_affix}_depth-{self.image_id}-{self.file_type}"
        inst_pathname =     f"{folder}\\inst-{path_affix}-{self.image_id}-{self.file_type}"
        combined_pathname = f"{folder}\\combined-{path_affix}-{self.image_id}-{self.file_type}" 
        
        if save_rgb:
            cv2.imwrite(rgb_pathname, result["image"][..., ::-1])
        
        if save_inst:
            
            # cv2.imwrite(inst_pathname, np.uint16(result["inst"]))
            # save numpy 
            
            np.save(f"{folder}\\inst-{path_affix}.npy", result["inst"])
            nr_of_inst= len(np.unique(result["inst"]))
            
            if nr_of_inst > 3:
                print(f"instance image has {nr_of_inst} unique values")
            elif nr_of_inst < 3:
                print(f"instance image is empty with {nr_of_inst} unique values")
        if save_combined:
            cv2.imwrite(combined_pathname, result.vis()[..., ::-1])

    def simple_render(self, folder = r"data", file_prefix = "", file_affix=""):

        sys.path.append(os.path.dirname(os.path.realpath(__file__)))
        
        path= os.path.join(os.getcwd(), folder)
        path = os.path.join(path, file_prefix + file_affix+ "-" + self.image_id + "-" + self.file_type)
        self.image_path_list.append(path)
        bpy.context.scene.render.filepath= path
        bpy.ops.render.render(animation=False, write_still=True, use_viewport=False, layer='', scene='')
    def combine_simple_renders(self, path= "data"):
        """ combine the simple renders into a single image. The first image is the pointcloud image and the second image is the map image.
        NOTE, we need to filter out the red tinted pixels from the pointcloud image and then combine the images, to overlay the images properly."""
        start_time  = time.time()
        pointcloud_image = cv2.imread(self.image_path_list[0])
        map_image = cv2.imread(self.image_path_list[1])
        
        
        # Extract the red channel
        red_channel = pointcloud_image[:, :, 0]

        # Set a threshold for red tinted pixels
        threshold = 40  # Adjust this value based on your specific case

        # Create a boolean mask for red tinted pixels
        red_tinted_mask = red_channel > threshold

        # Use the mask to filter out non-red tinted pixels
        combined_image = map_image.copy()
        combined_image[red_tinted_mask] = pointcloud_image[red_tinted_mask]
         
        cv2.imwrite(f"combined{self.file_type}", combined_image)
        print(f"Combining images took: {time.time() - start_time}")
        return combined_image
        
