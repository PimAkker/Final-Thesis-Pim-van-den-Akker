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

def plot_scan(cloud=None, rotate_cloud=None,path_name="", underlaying_image=None):
    
    # set the fixsize the same as the aspect ratio of the image
    im_shape = underlaying_image.shape
    fig_size = (10, 10 * im_shape[0] / im_shape[1])
    
    cloud_settings = {'s': 10, 'c': 'r', 'marker': 's'}
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

    plt.figure(figsize=fig_size)
    plt.scatter(rotate_cloud[:, 0], rotate_cloud[:, 1], **cloud_settings)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Rotated Cloud')
    plt.imshow(underlaying_image, cmap='gray', aspect='auto')
    plt.gca().set_facecolor('black')
    plt.xlim(0, im_shape[1])
    plt.ylim(0, im_shape[0])

    plt.show()


def get_and_filter_all_files(path = "", filename_contains="room1"):
    files = os.listdir(path)
    files = [os.path.join(path, f) for f in files if filename_contains in f]
    return files

def transform_cloud(cloud, angle=0, translation=[0,0], flip_x=False, flip_y=False, scale=1.0):
    angle = np.radians(angle)
    cloud = cloud[:, :2]
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_cloud = np.dot(cloud, rotation_matrix)
    if flip_x:
        rotated_cloud[:, 0] = -rotated_cloud[:, 0]
    if flip_y:
        rotated_cloud[:, 1] = -rotated_cloud[:, 1]
    translated_cloud = rotated_cloud + translation
    scaled_cloud = translated_cloud * scale
    return scaled_cloud


    

if __name__ == "__main__":
    path = r"../../real_world_data/raw_csv/csv_files"
    room = "room1"
    rotation_angle = 270
    translation = [0,0]
    flip_x = True
    flip_y = False
    scale_pointcloud = 100
    
    
    csvFile = get_and_filter_all_files(path = path, filename_contains=room)[0]
    image_path = get_and_filter_all_files(path = r"../../real_world_data/Room maps/room_map_inputs", filename_contains=room)[0]
    image = plt.imread(image_path)
    cloud = read_csv(csvFile)
    rotated_cloud = transform_cloud(cloud, angle=rotation_angle, translation=translation, flip_x=flip_x, flip_y=flip_y, scale= scale_pointcloud)

    plot_scan(cloud = None, rotate_cloud=rotated_cloud,underlaying_image =image ,path_name=image_path.split("/")[-1])
# %%
