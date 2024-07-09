#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import bpycv
import random
import numpy as np
import time
import custom_render_utils
import importlib
import blender_python_code.data_gen_utils as data_gen_utils
from category_information import category_information
total_start_time = time.time()

import pandas as pd


def modify_values_for_ablation(modifier_list, fix_values_dicts):
    """
    change a value for during ablation studies, this will change the value to the mean of the range of the value
    
    
    args: 
        modifier_list: list of dictionaries with the modifiers
        fix_values_dict: dictionary of values which can either have a fixed value or a 'mean'
            value, if the value is 'mean' the value will be changed to the mean of the range of the value
        
    returns:
        modifier_list: list of dictionaries with the modifiers with the fixed values
    """
    for modifier_dict in modifier_list:
        for fix_value_name in fix_values_dicts:
            if fix_value_name in modifier_dict:
                if fix_values_dicts[fix_value_name] == 'mean':
                    modifier_dict[fix_value_name] = np.mean(modifier_dict[fix_value_name])
                else:
                    modifier_dict[fix_value_name] = fix_values_dicts[fix_value_name]
    

    return modifier_list



def generate_dataset(nr_of_images=1,
                     folder_name="",  
                     overwrite_data=False, 
                     empty_folders=False, 
                     objects_to_add_percentage=0.6666,
                     objects_to_remove_percentage=0.333, 
                     object_to_move_percentage=0.5, 
                     force_object_visibility=['walls'], 
                     max_shift_distance=.5, 
                     set_colors={}, 
                     walls_modifiers={}, 
                     chairs_modifiers={}, 
                     round_table_modifiers={},
                     raytrace_modifiers={}, 
                     minimum_overlap_percentage_for_visible=0.1,
                     ablation_parameter={},
                     map_resolution=[270,270],
                     LiDAR_height=1.5
                     ):
    """
    Function to generate a dataset of images with corresponding masks and metadata
    """

    total_start_time = time.time()
    
    masks_folder = os.path.join(folder_name, r"Masks")
    images_folder =   os.path.join(folder_name, r"Images")
    metadata_folder = os.path.join(folder_name, r"Metadata")

    # for ablation studies we can fix the values of the modifiers
    modify_values_for_ablation([walls_modifiers, chairs_modifiers, round_table_modifiers, raytrace_modifiers], ablation_parameter)

    data_gen_utils.create_folders([masks_folder,images_folder, metadata_folder])
    data_gen_utils.delete_folder_contents([masks_folder,images_folder, metadata_folder],empty_folders=empty_folders)
    file_number = data_gen_utils.overwrite_data(images_folder, overwrite_data = overwrite_data)

    #  The object placement class, keeps track of object placement and can place objects in the scene
    pc = data_gen_utils.blender_object_placement(delete_duplicates=False)

    # Create an empty pandas dataframe
    instance_nr_df = pd.DataFrame(index=range(nr_of_images), columns=category_information.keys())

    for object_name, color in set_colors.items():
        pc.set_object_color(object_name, color)

    for i in np.arange(file_number, nr_of_images + file_number):
     
        pc.delete_duplicates_func() #delete duplicates at the start to refresh the scene
        
        start_time = time.time()
        
        cru_class = custom_render_utils.custom_render_utils(image_id=str(i),
                                                            remove_intermediary_images=True,
                                                            minimum_render_overlap_percentage=minimum_overlap_percentage_for_visible, 
                                                            exclude_from_render=pc.original_objects,
                                                            force_map_visibility=force_object_visibility,
                                                            output_resolution=map_resolution)
                                                            
      
        for modifier in list(walls_modifiers.keys()):
            pc.set_modifier("walls", modifier, walls_modifiers[modifier])
        for modifier in list(chairs_modifiers.keys()):
            pc.set_modifier("chair", modifier, chairs_modifiers[modifier])
        for modifier in list(round_table_modifiers.keys()):
            pc.set_modifier("round table", modifier, round_table_modifiers[modifier])
 
        for modifier in list(raytrace_modifiers.keys()):   
            pc.set_modifier("raytrace", modifier, raytrace_modifiers[modifier])        


        _, height, width, depth = pc.get_object_dims(object_name="walls")
        pc.place_objects(object_name="walls", inst_id=category_information["walls"], seperate_loose=False)
        pc.place_objects(object_name="doors", inst_id=category_information["doors"])
        pc.place_objects(object_name="chairs display", inst_id=category_information["chairs"])
        pc.place_objects(object_name="tables display", inst_id=category_information["tables"])
        pc.place_objects(object_name="pillars display", inst_id=category_information["pillars"])
 
        print("\033[91m" + f"Time for placing objects: {time.time() - start_time}" + "\033[0m")
        rand_pos_in_room = [random.gauss(0, width/6), random.gauss(0, depth/6), np.random.uniform(LiDAR_height[0],LiDAR_height[1])-height/2]
        pc.place_LiDAR(position=rand_pos_in_room)

        
        # move a percentage of objects down, this will move them out of the raytrace image
        # and will therefore not be seen in the pointcloud but will be seen in the mask and map
        # simulating that they are removed in real life but present on the map
        chairs_to_remove = pc.select_subset_of_objects(object_type_name="chairs display", selection_percentage=  objects_to_remove_percentage)
        tables_to_remove = pc.select_subset_of_objects(object_type_name="tables display", selection_percentage=  objects_to_remove_percentage)
        pillars_to_remove = pc.select_subset_of_objects(object_type_name="pillars display", selection_percentage=objects_to_remove_percentage)
        
        pc.set_object_id(category_information["chairs removed"], selection=chairs_to_remove)
        pc.set_object_id(category_information["tables removed"], selection=tables_to_remove)
        pc.set_object_id(category_information["pillars removed"], selection=pillars_to_remove)
        
        pc.move_objects_relative(chairs_to_remove,  [0, 0, -height])
        pc.move_objects_relative(tables_to_remove,  [0, 0, -height])
        pc.move_objects_relative(pillars_to_remove, [0, 0, -height])
        


        pc.hide_objects(chairs_to_remove+tables_to_remove+pillars_to_remove)
        
        
        pc.isolate_object("raytrace")
        # place_class.configure_camera(position=(0, 0, height/2))
        pc.set_modifier("raytrace.001", "visible surface switch", False)
        cru_class.simple_render(folder=images_folder, file_prefix="pointcloud", file_affix="")
        pc.set_modifier("raytrace.001", "visible surface switch", True)
        cru_class.simple_render(folder=images_folder, file_prefix="visible_region_mask", file_affix="")
        
        pc.unisolate()


        
        pc.delete_single_object("raytrace.001")
        
        # Here we remove a percentage of the objects and add a percentage of the objects from the map, they will still be 
        # visible in the pointcloud but not on the map, additionanly we will change the object id of the objects that are added
        chairs_to_add = pc.select_subset_of_objects(object_type_name="chairs display", selection_percentage=  objects_to_add_percentage)
        tables_to_add = pc.select_subset_of_objects(object_type_name="tables display", selection_percentage=  objects_to_add_percentage)
        pillars_to_add = pc.select_subset_of_objects(object_type_name="pillars display", selection_percentage=objects_to_add_percentage)
        
        pc.set_object_id(category_information["chairs new"],  selection=chairs_to_add)
        pc.set_object_id(category_information["tables new"],  selection=tables_to_add)
        pc.set_object_id(category_information["pillars new"], selection=pillars_to_add)
        

        chairs_to_move = list(np.random.choice(chairs_to_add, int(len(chairs_to_add) *    object_to_move_percentage)))
        tables_to_move = list(np.random.choice(tables_to_add, int(len(tables_to_add) *    object_to_move_percentage)))
        pillars_to_move = list(np.random.choice(pillars_to_add, int(len(pillars_to_add) * object_to_move_percentage)))

        # this move the objects in a random direction and shifts them down, this will make them invisible in the pointcloud
        relative_move = lambda : [random.uniform(-max_shift_distance, max_shift_distance) , random.uniform(-max_shift_distance, max_shift_distance), -height]

        moved_chairs =  pc.duplicate_move(objects_list=chairs_to_move, relative_position=relative_move())
        moved_tables =  pc.duplicate_move(objects_list=tables_to_move, relative_position=relative_move())
        moved_pillars = pc.duplicate_move(objects_list=pillars_to_move, relative_position=relative_move())

        pc.set_object_id(category_information["chairs removed"], selection=moved_chairs)
        pc.set_object_id(category_information["tables removed"], selection=moved_tables)
        pc.set_object_id(category_information["pillars removed"], selection=moved_pillars)
                
        # render the instance segmentation mask1
        pc.unhide_objects(chairs_to_move+tables_to_move+pillars_to_move)
        
        
        cru_class.render_data_semantic_map(folder=masks_folder, path_affix=f"mask", save_combined=False, save_rgb=False, save_inst=True)   
        pc.clean_up_materials()
        
        pc.delete_objects(object_list = chairs_to_add + tables_to_add + pillars_to_add)
        
        # # create the map and combine the poitcloud and map to a single image, creating the input for the model
        cru_class.simple_render(folder=images_folder, file_prefix="map", file_affix="")
        
        cru_class.combine_simple_renders(path=images_folder, file_nr=f"{i}", make_black_and_white=False)
        
        instance_nr_df = cru_class.update_dataframe_with_metadata(instance_nr_df)

        
        pc.finalize()
        
        print("\033[94m" + f"Time for this image: {time.time() - start_time}" + "\033[0m")

    print(f"Finished dataset, Created {nr_of_images} images in {time.time() - total_start_time} seconds.")

    instance_nr_df.to_csv(os.path.join(metadata_folder, "object_count_metadata.csv"), index=False)
    data_gen_utils.save_metadata(metadata_path=metadata_folder,nr_of_images=i+1, modifiers_list= [walls_modifiers, chairs_modifiers, round_table_modifiers,  raytrace_modifiers, set_colors],time_taken= time.time() - total_start_time, ablation_parameter=ablation_parameter, map_resolution=map_resolution, LiDAR_height=LiDAR_height)
    print("Done!")
