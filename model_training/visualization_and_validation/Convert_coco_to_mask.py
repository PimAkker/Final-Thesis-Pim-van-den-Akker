"""
This script is used to convert the coco dataset to mask images, used after labeling data manually
"""
#%%
import os
import json
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from pycocotools import mask as maskutil

def convert_coco_to_mask(coco_json_path, output_path):
    with open(coco_json_path, 'r') as f:
        data = json.load(f)
    for img in data['images']:
        image_id = img['id']
        image_name = img['file_name']
        image_height = img['height']
        image_width = img['width']
        categories = data['categories']
        output_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        class_list_occurance_nr = np.zeros((100)) -1
        for ann in data['annotations']:
            if ann['image_id'] == image_id:
                
                mask = np.zeros((image_height, image_width), dtype=np.uint8)
                polygons = ann['segmentation']
                
                type_id = ann['category_id']-1
                type_name= int(list(categories[type_id].values())[1])
                
                print(f'type_id: {type_name}')
                
                # if the class is 999 then we ignore it by setting it to background
                if type_name == 999: 
                    type_name = 0
                class_list_occurance_nr[type_name] += 1
                instance_id = type_name*1000 + class_list_occurance_nr[type_name]
                
                for polygon in polygons:
                    vertices = np.array(polygon).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [vertices], 1) # fill in the polygon with the instance id)

                mask = mask.astype(bool)                                   
                instance_id_single = mask * instance_id
                    
                output_mask = np.where(mask, instance_id_single, output_mask)
                
        
        plt.imshow(output_mask)
        print(f"unique values: {np.unique(output_mask)}")
        if len(np.unique(output_mask)) == 0:
            print(f"Empty mask for image {image_name}")
            
        # Add a legend
        plt.legend()
        # plt.show()
        np.save(os.path.join(output_path, image_name.replace(".png", "")), output_mask)
#%%        
if __name__ == '__main__':
    
    coco_json_path = r'C:\Users\pimde\OneDrive\thesis\Blender\real_world_data\Real_world_data_V3\coco_format_no_walls_tables.json'
    output_path = r'C:\Users\pimde\OneDrive\thesis\Blender\real_world_data\Real_world_data_V3\Masks'
    convert_coco_to_mask(coco_json_path, output_path)
    print('Done')
# %%
