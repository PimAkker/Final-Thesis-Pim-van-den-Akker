#%%
"""
This script is to overlay the point cloud on the image of the room. For the generation of real world data

"""

import csv
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
    
def read_csv(file_path):
    data = np.genfromtxt(file_path, delimiter=',')
    return data

def plot_scan(cloud=None, rotate_cloud=None,path_name="", underlaying_image=None, marker_size=15,save_path=None):
    
    # set the fixsize the same as the aspect ratio of the image
    im_shape = underlaying_image.shape
    fig_size = (10, 10 * im_shape[0] / im_shape[1])
    
    cloud_settings = {'s': marker_size, 'c': 'r', 'marker': 's'}
    if cloud is not None:
        plt.figure(figsize=fig_size)
        plt.scatter(cloud[:, 0], cloud[:, 1], **cloud_settings)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Original Cloud')
        plt.imshow(underlaying_image, cmap='gray', aspect='auto')
        plt.gca().set_facecolor('black')
        plt.xlim(0, im_shape[1])
        plt.ylim(0, im_shape[0])
        plt.show()

    plt.figure(figsize=fig_size, facecolor='black')
    plt.scatter(rotate_cloud[:, 0], rotate_cloud[:, 1], **cloud_settings)
    plt.imshow(underlaying_image, aspect='auto')
    plt.gca().set_facecolor('black')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, pad_inches=0)
        
        
    plt.show()

  



def get_and_filter_all_files(path = "", filename_contains="room1"):
    files = os.listdir(path)
    files = [os.path.join(path, f) for f in files if filename_contains in f]
    return files

def transform_cloud(cloud, angle=0, translation=[0,0], flip_x=False, flip_y=False, scale_x=1.0, scale_y=1.0):
    angle = np.radians(angle)
    cloud = cloud[:, :2]
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_cloud = np.dot(cloud, rotation_matrix)
    if flip_x:
        rotated_cloud[:, 0] = -rotated_cloud[:, 0]
    if flip_y:
        rotated_cloud[:, 1] = -rotated_cloud[:, 1]
    
    scaled_cloud = rotated_cloud * np.array([scale_x, scale_y])
    translated_cloud = scaled_cloud + translation
    return translated_cloud


    

if __name__ == "__main__":
    pointcloud_path = r"../../real_world_data/raw_csv/csv_files"

    
    # room 1
    # room = "room1"
    # rotation_angle = 92
    # translation = [294,95]
    # flip_x = False
    # flip_y = False
    # scale_x = 79
    # scale_y = 85
    
    
    
    # room 2
    room = "room2"
    rotation_angle = -1
    translation = [132,127]
    flip_x = False
    flip_y = False
    scale_x = 78
    scale_y = 84

    #room = "room3"
    #rotation_angle = -2
    #translation = [125,164]
    #flip_x = False
    #flip_y = False
    #scale_x = 74
    #scale_y = 79
    
    #room = "room4"
   # rotation_angle = -178
  #  translation = [315,364]
 #   flip_x = False
#    flip_y = False
#    scale_x = 173
#    scale_y = 190
    
    save_folder = r"../../real_world_data/raw_csv/svg files"
    
    point_size = 40
    save_file_type = ".svg"
    
    
    
    pointcloud_paths = get_and_filter_all_files(path = pointcloud_path, filename_contains=room)
    image_paths = get_and_filter_all_files(path = r"../../real_world_data/Room maps/room_map_inputs", filename_contains=room)
    for image_path in image_paths:
        for cloud_path in pointcloud_paths:
            
            image = plt.imread(image_path)
            cloud = read_csv(cloud_path)
            rotated_cloud = transform_cloud(cloud, angle=rotation_angle, translation=translation, flip_x=flip_x, flip_y=flip_y, scale_x= scale_x, scale_y=scale_y)

    
            save_path = os.path.join(save_folder, os.path.basename(cloud_path).replace(".csv", save_file_type))
            plot_scan(cloud=None, rotate_cloud=rotated_cloud, underlaying_image=image, marker_size=point_size, save_path=save_path)


# %%
