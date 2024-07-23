# %%
import cairosvg
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
def import_svg_output_png(input_path, output_image_height=270):
    
    # Get all the svg files in the input path
    svg_files =  glob.glob(os.path.join(input_path, "*.svg"))
    
    # Create the output folder
    output_folder = os.path.join(input_path, "output_png")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Convert the svg files to png
    for svg_file in svg_files:
        output_file = os.path.join(output_folder, os.path.basename(svg_file).replace(".svg", ".png"))
        cairosvg.svg2png(url=svg_file, write_to=output_file,dpi=500)     
        
        # Resize the image
        img = Image.open(output_file)
        input_image_width = img.size[0]
        input_image_height = img.size[1]
        
        scaled_output_image_width = int(output_image_height * input_image_width / input_image_height)
        # round to the nearest 10 
        # scaled_output_image_width = scaled_output_image_width + 10 - scaled_output_image_width % 10
        
        img = img.resize((scaled_output_image_width, output_image_height), Image.NEAREST)
        img.save(output_file)
        
        # Display the image
        display(img)
        
import_svg_output_png(r"C:\Users\pimde\OneDrive\thesis\Blender\real_world_data\raw_csv\svg_files_no_tables")
        
# %%
