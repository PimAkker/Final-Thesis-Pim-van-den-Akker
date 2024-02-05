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
        self.simple_render_image_path_list = []
        self.render_data_mask_path_list = []
        
    def render_data(self,folder = r"data", path_affix="", save_rgb=True, save_inst=True, save_combined=True):
        
        # render image, instance annoatation and depth
        result = bpycv.render_data()

        rgb_pathname =      os.path.join(folder, f"rgb-{path_affix}-{self.image_id}-{self.file_type}")
        depth_pathname =    os.path.join(folder, f"depth-{path_affix}_depth-{self.image_id}-{self.file_type}")
        inst_pathname =     os.path.join(folder, f"inst-{path_affix}-{self.image_id}-{self.file_type}")
        combined_pathname = os.path.join(folder, f"combined-{path_affix}-{self.image_id}-{self.file_type}" )
        
        if save_rgb:
            cv2.imwrite(rgb_pathname, result["image"][..., ::-1])
        
        if save_inst:
            path_name = os.path.join(folder, f"inst-{path_affix}-{self.image_id}-.npy") 
            np.save(path_name, result["inst"])
            self.render_data_mask_path_list.append(path_name)
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
        self.simple_render_image_path_list.append(path)
        bpy.context.scene.render.filepath= path
        bpy.ops.render.render(animation=False, write_still=True, use_viewport=False, layer='', scene='')
    def combine_simple_renders(self, path= "data", remove_originals = True, file_nr=""):
        """ combine the simple renders into a single image. The first image is the pointcloud image and the second image is the map image.
        NOTE, we need to filter out the red tinted pixels from the pointcloud image and then combine the images, to overlay the images properly."""

        pointcloud_image = cv2.imread(self.simple_render_image_path_list[0])
        map_image = cv2.imread(self.simple_render_image_path_list[1])
        
        red_channel_map = map_image[:, :, 0]
        blue_channel_map = map_image[:, :, 2]
        # if either the red or blue channel is above  then it is not background
        mask_map = (red_channel_map > 10) & (blue_channel_map > 10)
        map_image[mask_map] = [0, 0, 0]
        map_image[~mask_map] = [255, 255, 255]
        
              
        
        
        
        # Extract the red channel
        red_channel = pointcloud_image[:, :, 0]
        threshold = 20 
        red_tinted_mask = red_channel > threshold
        # Use the mask to filter out non-red tinted pixels
        combined_image = map_image.copy()
        combined_image[red_tinted_mask] = pointcloud_image[red_tinted_mask]
         
        cv2.imwrite(os.path.join(path, f"input-{file_nr}-{self.file_type}"), combined_image)
        if remove_originals:
            # delete the pointcloud and map images
            os.remove(self.simple_render_image_path_list[0])
            os.remove(self.simple_render_image_path_list[1])
    def combine_masks(self, path="data",remove_originals=True):
        """
        NOTE: will be removed, not used in the final version
        """
        
        
        mask_prior = np.load(self.render_data_mask_path_list[0])
        mask_true = np.load(self.render_data_mask_path_list[1])
        
        # where mask_prior is 0 where mask_true is not an object has been removed. 
        # If an object has been removed then +100 to the mask prior value
        mask_prior_nonzero = mask_prior != 0
        mask_true_nonzero = mask_true != 0
        non_present_mask = mask_prior_nonzero != mask_true_nonzero
        
        combined_mask = mask_prior.copy()
        combined_mask[non_present_mask] += 100
        
        np.save(f"combined_mask.npy", combined_mask)
        
        if remove_originals:
            # delete the prior and true masks
            os.remove(self.render_data_mask_path_list[0])
            os.remove(self.render_data_mask_path_list[1])

        
        
        
        
