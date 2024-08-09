# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt
import numpy as np
import PIL

import os 

input_image = r"C:\Users\pimde\OneDrive\thesis\Blender\data\test1\wheredidthepillarsgo\[]\Images\input-0-.png"
image = PIL.Image.open(input_image)
output = r"C:\Users\pimde\OneDrive\thesis\Blender\data\test1\wheredidthepillarsgo\[]\Masks\inst-mask-0-.npy"
# get the path of the output folder
rootoutput = os.path.dirname(output)
output = np.load(output)
fig, axes = plt.subplots(1, 2)

axes[0].imshow(image)
axes[0].set_title('Input Image')

axes[1].imshow(output, alpha=0.5)
axes[1].set_title('Output Mask')

plt.show()
# %%
image1 = r'C:\Users\pimde\OneDrive\thesis\Blender\data\test1\wheredidthepillarsgo\[]\Images\visible_region_mask-4-.png'
image2= r'C:\Users\pimde\OneDrive\thesis\Blender\data\test1\wheredidthepillarsgo\[]\Images\input-4-.png'

image1 = PIL.Image.open(image1)
image2 = PIL.Image.open(image2)

plt.imshow(image1)
plt.imshow(image2, alpha=0.5)
# %%
