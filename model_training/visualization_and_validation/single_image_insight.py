#%%
import cv2 
import numpy as np 


def single_image_count(img):
    img = img.astype(np.uint8)
    unique_colors = np.unique(img.reshape(-1,3), axis=0, return_counts=True)
    # show only the colors that are used more than 100 times
    unique_colors = unique_colors[0][unique_colors[1]>10]
    for color in unique_colors:
        count = np.sum(np.all(img == color, axis=-1))
        print(f"Color: {color}, Count: {count}")
img = cv2.imread(r"C:\Users\pimde\OneDrive\thesis\Blender\data\Images\input-1-.png",cv2.IMREAD_UNCHANGED)
single_image_count(img)
# %%

