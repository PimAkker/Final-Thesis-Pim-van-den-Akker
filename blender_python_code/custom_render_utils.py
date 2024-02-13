import cv2
import bpy
import bpycv
import random
import numpy as np
import os
import sys
import time
from PIL import Image

class custom_render_utils:
    def __init__(self, image_id = "0"):
        self.image_id = image_id
        self.input_file_type = ".png" # only change this when also changed in blender renderer
        self.output_file_type = ".jpg"
        self.simple_render_image_path_dict = {}
        self.render_masks = {}
        self.masks_path_dict = {}

    def render_data(self,folder = r"data", path_affix="", save_rgb=True, save_inst=True, save_combined=True):
        
        # render image, instance annoatation and depth
        result = bpycv.render_data()

        rgb_pathname =      os.path.join(folder, f"rgb-{path_affix}-{self.image_id}-{self.input_file_type}")
        depth_pathname =    os.path.join(folder, f"depth-{path_affix}_depth-{self.image_id}-{self.input_file_type}")
        inst_pathname =     os.path.join(folder, f"inst-{path_affix}-{self.image_id}-{self.input_file_type}")
        combined_pathname = os.path.join(folder, f"combined-{path_affix}-{self.image_id}-{self.input_file_type}" )
        
        if save_rgb:
            cv2.imwrite(rgb_pathname, result["image"][..., ::-1])
        
        if save_inst:
            path_name = os.path.join(folder, f"inst-{path_affix}-{self.image_id}-.npy") 
            self.masks_path_dict[path_affix]= path_name
            np.save(path_name, result["inst"])
            self.render_masks[path_affix]= result["inst"]
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
        path = os.path.join(path, file_prefix + file_affix+ "-" + self.image_id + "-" + self.input_file_type)
        self.simple_render_image_path_dict[file_prefix]= path
        bpy.context.scene.render.filepath= path
        bpy.ops.render.render(animation=False, write_still=True, use_viewport=False, layer='', scene='')
        
    def combine_simple_renders(self, path= "data", remove_originals = True, file_nr="",render_visible_only=False):
        """ combine the simple renders into a single image. The first image is the pointcloud image and the second image is the map image."""

        pointcloud_image = np.array(Image.open(self.simple_render_image_path_dict['pointcloud']))
        map_image = np.array(Image.open(self.simple_render_image_path_dict['map']))
        visible_region_mask = cv2.imread(self.simple_render_image_path_dict['visible_region_mask'], cv2.IMREAD_UNCHANGED)
        
        # cut out the visible region from the images

        
        # everywhere where map image opacity is  0 set it to white
        map_image[map_image[:,:,3] != 0] = [255,255,255,1]
        
        # everywhere where pointcloud image opacity is 0 set it to red
        pointcloud_image[pointcloud_image[:,:,3] != 0] = [0,0,255,1]
        
        #  add pointcloud image to map image where the pointcloud image does not have 0 opacity 
        combined_image = map_image.copy()
        combined_image[pointcloud_image[:,:,3] == 1] = pointcloud_image[pointcloud_image[:,:,3] == 1]
        
        # only show the visible region in the combined image
        if render_visible_only:
            combined_image[visible_region_mask[:,:,3] == 0] = [0,0,0,1]
           
         
        cv2.imwrite(os.path.join(path, f"input-{file_nr}-{self.output_file_type}"), combined_image)
        
        if remove_originals:
            # delete the pointcloud and map images
            for key in self.simple_render_image_path_dict:
                os.remove(self.simple_render_image_path_dict[key])
    def process_masks(self, path="data",remove_originals=False, output_only_visible_region=False):
        """
        
        """
        mask = self.render_masks['mask']
        visible_region_mask = cv2.imread(self.simple_render_image_path_dict['visible_region_mask'], cv2.IMREAD_UNCHANGED)
        visible_region_mask = visible_region_mask.astype(np.int64)
        obj_ids = np.unique(mask)
        
        
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None])
        
        vis_mask_mask = visible_region_mask[:,:,3] != 0
        vis_mask_mask = np.tile(vis_mask_mask,(len(obj_ids),1,1))
        overlap_areas = ((vis_mask_mask==1) & (masks==1))
        # select the masks that have an overlap with the visible region mask
        masks_containing_overlap_mask = overlap_areas.any(axis=(1,2))
        masks_containing_overlap = masks[masks_containing_overlap_mask]
        masks_containing_overlap = (np.sum(masks_containing_overlap,axis=0)).astype(bool)
        output_mask = np.zeros_like(mask)
        output_mask[masks_containing_overlap] = mask[masks_containing_overlap]
        np.save(self.masks_path_dict['mask'], output_mask)
    
        
