import cv2
import bpy # type: ignore
import bpycv
import os
import sys
import numpy as np

from category_information import category_information
import time
# from category_information import category_information

class custom_render_utils:
    def __init__(self, image_id = "0",remove_intermediary_images = True, minimum_render_overlap_percentage=False,exclude_from_render= None, force_map_visibility = []):
        """
        inputs: image_id (str): the id of the image will be used for naming all the files in this class
        remove_intermediary_images (bool): if True the original images in mask_path_dict and simple_render_image_path_dict 
        will be removed after the combined image is created.
        render_only_visible (bool): if True the mask will only contain the visible region of the mask, this functions
        only gets triggered when combine_simple_renders is called.
        exclude_from_render (list of bpy.types.Object): list of objects that will be excluded from the render useful for speeding up the rendering process and debugging
        
        """
        
        
        self.image_id = image_id
        self.input_file_type = ".png" # only change this when also changed in blender renderer
        self.output_file_type = ".png"
        self.exclude_from_render = exclude_from_render
        self.simple_render_image_path_dict = {}
        self.render_masks = {}
        self.masks_path_dict = {}
        self.input_file_path = ""
        self.remove_originals = remove_intermediary_images
        self.minimum_visible_overlap_percentage = minimum_render_overlap_percentage
        self.unique_classes = []
        self.nr_of_instances_per_class = []
        self.force_map_visibility = force_map_visibility
        
    def render_data_semantic_map(self,folder = r"data", path_affix="", save_rgb=True, save_inst=True, save_combined=True):
        
        tic = time.time()
        # to speed up the rendering process, we can exclude some objects from the render
        if self.exclude_from_render is not None:
            for obj in self.exclude_from_render:
                obj.hide_render = True
        
        # render image, instance annoatation and depth
        result = bpycv.render_data(render_image=False)

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
            
            unique_inst = np.unique(result["inst"])
            nr_of_inst= len(unique_inst)
            
            #save some metadata about the classes in the image
            self.unique_classes = np.unique(unique_inst//1000)
            self.nr_of_instances_per_class = [np.sum(unique_inst//1000 == i) for i in self.unique_classes]
            
            
            if nr_of_inst > 3:
                print(f"instance image has {nr_of_inst} unique values")
            elif nr_of_inst < 3:
                print(f"instance image is empty with {nr_of_inst} unique values")
                
        if save_combined:
            cv2.imwrite(combined_pathname, result.vis()[..., ::-1])
        
        print(f'time for rendering {path_affix}: {time.time()-tic}')
        
            

    def simple_render(self, folder = r"data", file_prefix = "", file_affix=""):
        
        # to speed up the rendering process, we can exclude some objects from the render
        if self.exclude_from_render is not None:
            for obj in self.exclude_from_render:
                obj.hide_render = True

        sys.path.append(os.path.dirname(os.path.realpath(__file__)))
        
        path= os.path.join(os.getcwd(), folder)
        path = os.path.join(path, file_prefix + file_affix+ "-" + self.image_id + "-" + self.input_file_type)
        self.simple_render_image_path_dict[file_prefix]= path
        tic = time.time()
        bpy.context.scene.render.filepath= path
        bpy.ops.render.render(animation=False, write_still=True, use_viewport=False, layer='', scene='')
        print(f'time for rendering {file_prefix}: {time.time()-tic}')
        
    def combine_simple_renders(self, path= "data", file_nr="", make_black_and_white=False):
        """ combine the simple renders into a single image. The first image is the pointcloud image and the second image is the map image."""


        pointcloud_image = (cv2.imread(self.simple_render_image_path_dict['pointcloud'],cv2.IMREAD_UNCHANGED)).astype(np.uint8)
        map_image = (cv2.imread(self.simple_render_image_path_dict['map'], cv2.IMREAD_UNCHANGED)).astype(np.uint8)

        # show unique rgba values
        if make_black_and_white:
            # everywhere where map image opacity is not  0 set it to white
            map_image[map_image[:,:,3] != 0] = [255,255,255,255]
            
        # everywhere where pointcloud image opacity is 0 set it to red 
        pointcloud_image[pointcloud_image[:,:,3] != 0] = [0,0,255,255]
                           
        #  Add pointcloud image to map image where the pointcloud image does not have 0 opacity 
        combined_image = map_image.copy()
        combined_image[pointcloud_image[:,:,3] == 255] = pointcloud_image[pointcloud_image[:,:,3] == 255]

        self.input_file_path = os.path.join(path, f"input-{file_nr}-{self.output_file_type}")
        if self.minimum_visible_overlap_percentage != 0:
            self.render_only_visible(combined_image)
        else:
            cv2.imwrite(self.input_file_path, combined_image)
            
        if self.remove_originals:
            # delete the pointcloud and map images that are used to create the combined image
            for key in self.simple_render_image_path_dict:
                os.remove(self.simple_render_image_path_dict[key])
        
    
    def render_only_visible(self,combined_image):
        """
        This function will only render the visible region of the mask	
        NOTE: has to be triggered after the creation of visible_region_mask
        """
        mask = np.load(self.masks_path_dict['mask'])
        visible_region_mask = cv2.imread(self.simple_render_image_path_dict['visible_region_mask'], cv2.IMREAD_UNCHANGED).astype(np.uint8)
        
        obj_ids = np.unique(mask)
        
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set of binary masks
        masks = (mask == obj_ids[:, None, None])
        vis_mask_mask = visible_region_mask[:,:,3] != 0
        vis_mask_mask = np.tile(vis_mask_mask,(len(obj_ids),1,1))
        overlap_areas = ((vis_mask_mask==1) & (masks==1))
        
        # select the masks that have an overlap with the visible region mask
        overlap_percentage_per_mask = np.sum(overlap_areas, axis=(1,2)) / np.sum(masks, axis=(1,2))
        
        # only render that have a significant overlap with the visible region mask
        masks_containing_overlap_selection = overlap_percentage_per_mask > self.minimum_visible_overlap_percentage
        
        obj_id_classes= obj_ids//1000
        force_visibility_categories = [category_information[object_name] for object_name in self.force_map_visibility]
        index_to_force_visibility_obj_ids = [i for i, obj_id_class in enumerate(obj_id_classes) if obj_id_class in force_visibility_categories]

        # force the visibility of the objects that are in the force_map_visibility list
        for index in index_to_force_visibility_obj_ids:
            masks_containing_overlap_selection[index] = True
            
        masks_containing_overlap = masks[masks_containing_overlap_selection]
        
        # combine the masks that have an overlap with the visible region mask
        masks_containing_overlap = (np.sum(masks_containing_overlap,axis=0)).astype(bool)
               
        output_mask = np.zeros_like(mask)
        output_mask[masks_containing_overlap] = mask[masks_containing_overlap]
        
        # modify the input file so that it only contains the visible region
        input_file_only_visible = np.zeros_like(combined_image)
        input_file_only_visible[masks_containing_overlap] = combined_image[masks_containing_overlap]
        
        np.save(self.masks_path_dict['mask'], output_mask)
        cv2.imwrite(self.input_file_path, input_file_only_visible,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
    def update_dataframe_with_metadata(self,df):
        """
        This function will update the dataframe with the metadata of the classes in the image
        """
        class_names = list(category_information.keys())

        self.unique_classes = [class_names[class_num] for class_num in self.unique_classes]
        
        for i,class_name in enumerate(self.unique_classes):
            df.at[int(self.image_id), class_name] = self.nr_of_instances_per_class[i]
        return df
    
        
        
        
        