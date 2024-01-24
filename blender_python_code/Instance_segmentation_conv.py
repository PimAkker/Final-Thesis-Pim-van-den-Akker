#  This code is adapted from: https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#create-custom-coco-dataset


import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)
from PIL import Image # (pip install Pillow)
import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)
import json
import cv2
import os
import numpy as np
import PIL
import time
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))

            # If the pixel is not black...
            if pixel != 0:
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks
def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)
            

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation
def show_shapes(file_path="", image_path=""):


    # Load the JSON file
    with open('annotations.json') as f:
        annotation = json.load(f)
        

    # Create a new matplotlib figure and axis
    fig, ax = plt.subplots()

    # Iterate over the annotations in the data
    # for annotation in data['annotations']:
        # Get the segmentation data
    segmentation = annotation[0]['segmentation']

    # Each segmentation is a list of polygons, and each polygon is a list of points
    for polygon in segmentation[0]:
        
        # Reshape the points into a 2D array
        polygon = np.array(polygon).reshape(-1, 2)

        # Create a Polygon patch
        poly_patch = patches.Polygon(polygon, fill=False)

        # Add the patch to the Axes
        ax.add_patch(poly_patch)
        pass

    # Display the shape and the figure over eachother
    
    # plt.imshow(np.load(image_path), alpha=0.5)
    #  fit the axis to the image
    # ax.set_xlim(0, 640)
    # ax.set_ylim(0, 480)
    plt.show()  
    pass



if __name__ == "__main__":
    image_path = r"C:\Users\pimde\OneDrive\thesis\Blender\blender_python_code\data\-inst-Prior1_inst.npy"
    image = np.load(image_path)
    image = PIL.Image.fromarray(image)
    mask_images = [image]

    start_time = time.time()

    is_crowd = 0


    annotation_id = 1
    image_id = 1

    # Create the annotations
    annotations = []
    for mask_image in mask_images:
        sub_masks = create_sub_masks(mask_image)
        
        for annotation_id, sub_mask in sub_masks.items():
            category_id = int(annotation_id[0])
            annotation_id = int(annotation_id)
            
            annotation = create_sub_mask_annotation(np.array(sub_mask), image_id, category_id, annotation_id, is_crowd)
            annotations.append(annotation)
            
        image_id += 1

    print(json.dumps(annotations))
    # save the annotations to a json file
    with open('annotations.json', 'w') as outfile:
        json.dump(annotations, outfile)
    print("--- %s seconds ---" % (time.time() - start_time))
    show_shapes(r"C:\Users\pimde\OneDrive\thesis\Blender\annotations.json",image_path )
    

