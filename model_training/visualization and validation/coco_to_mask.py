#%%

import json
import numpy as np
from pycocotools.coco import COCO
import cv2

# Load your COCO annotations file
annotation_file = r'D:\Downloads\labels_my-project-name_2024-04-08-04-04-53.json'

with open(annotation_file) as f:
    coco_data = json.load(f)

# Initialize an empty matrix
image_height = coco_data['images'][0]['height']  # Adjust as needed
image_width = coco_data['images'][0]['width']  # Adjust as needed
instance_matrix = np.zeros((image_height, image_width), dtype=np.uint8)  # Assuming labels are integers

# Get categories for mapping labels
categories = coco_data['categories']
category_id_to_label = {cat['id']: cat['name'] for cat in categories}

# Iterate over annotations and fill the matrix
for annotation in coco_data['annotations']:
    segmentation = annotation['segmentation']

    # Handle RLE format if needed
    if isinstance(segmentation, dict):  
        # You'll need a library like 'pycocotools.mask' to decode RLE
        import pycocotools.mask 
        mask = pycocotools.mask.decode(segmentation)
    else: 
        # Handle polygon format (list of [x, y] coordinates)
        polygon = np.array(segmentation).reshape((-1, 2))
        polygon = polygon.astype(np.int32)
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)  # You'll need OpenCV (cv2) for this

    label = category_id_to_label[annotation['category_id']]
    instance_matrix[mask == 1] = annotation['category_id']

# Now instance_matrix holds your pixel-wise labeled image
print(instance_matrix)
import matplotlib.pyplot as plt

# Display the instance matrix as an image
plt.imshow(instance_matrix, cmap='jet')
plt.colorbar()
plt.show()
# %%


