import bpy
import numpy as np
from PIL import Image

# -*- coding: utf-8 -*-
import os
import sys

# set file as curdir
path = os.path.dirname(os.path.abspath(__file__))
path = "\\".join(path.split("\\")[:-1])

os.chdir(path)


# ensure we are in the correct directory
root_dir_name = 'Blender'
current_directory = os.getcwd().split("\\")
assert root_dir_name in current_directory, f"Current directory is {current_directory} and does not contain root dir name:  {root_dir_name}"
if current_directory[-1] != root_dir_name:
    # go down in the directory tree until the root directory is found
    while current_directory[-1] != root_dir_name:
        os.chdir("..")
        current_directory = os.getcwd().split("\\")


# add all the subdirectories to the path
dirs  = os.listdir()
root = os.getcwd()
for dir in dirs:
    sys.path.append(os.path.join(root, dir))

from category_information import class_factor






class SimpleInstancer:
    """
    For this to work properly the objects should be configured in the following way:
    To make the colors of this actually be reflected in the output image: 
        1. In the _Color Management_ panel, set the _View Transform_ to _Raw_ (if you're not saving as OpenEXR)
        2. Use the RGB slider to set the color values or be aware of gamma correction
        3. Be aware of floating point inaccuracies or use OpenEXR as file format with _Float(Full)_
        4. Disable anti-aliasing
    as per https://blender.stackexchange.com/questions/154471/render-solid-color-in-eevee
    """
    
    
    
    
    def __init__(self):
        pass

    def class_to_rgb(self, class_nr, instance_nr):
        """
        This function converts a class number to an rgb color where (0,class,instance_nr, alpha) is the color
        """
        return (0, class_nr, instance_nr, 255)
    
    def rgb_to_instance_id(self, color):
        """
        This function converts a color to a class number
        """
        return color[1]*class_factor + color[2]

    def set_object_color(self, object_name, material_name,  color):
        """
        Sets the color of the object in the scene, the color is set in the emission node of the shader nodes. This means that
        the object will have the exact color specified. However each object should have the correct shader nodes for this to work.
        
        input color: tuple of the color in rgb in range 0-255 (r,g,b, alpha)
        input object_name: name of the object to set the color of note that this has to be the same name as the shader material
        output: None
        """
        # set color from rgb to percentage
        
        assert object_name in bpy.data.objects, f"Object {object_name} does not exist"
        assert bpy.data.materials.get(material_name) is not None, f"Material {object_name} does not exist make sure that the object has a material with the same name as the object and that it is assigned in the geometry nodes of the object if the object is created with geometry nodes"

        assert len(color) == 4, f"Color: {color} is not of shape (r,g,b,a)"
        assert all([0 <= c <= 255 for c in color]), f"Color: {color} is not in the range 0-255"
        
        color = [c/255 for c in color]
        
        bpy.context.view_layer.objects.active = bpy.data.objects[object_name]
        bpy.data.materials[material_name].node_tree.nodes["Emission"].inputs[0].default_value = color
    
    def set_objects_instance_id(self, object_name="", class_nr=None,selection=None, select_all_duplicates=False):
        """This function will take any object which containts the object name and a "." 
        and set the color of the object to a unique color for each instance.
        or only the selection if selection is not None
        """
        assert object_name != "", "Object name should not be empty"
        assert class_nr is not None, f"Class number for: {object_name} should not be None"
        
        
        if selection is not None:
            objects = selection
        elif select_all_duplicates:
            objects = [obj for obj in bpy.data.objects if object_name in obj.name and '.' in obj.name]
        else:
            objects = [bpy.data.objects[object_name]]
        
        assert len(objects) <= 255, f"Too many objects with name {object_name} found, max is 255"        
        
        for i, obj in enumerate(objects):
            # Duplicate the material on the object so that each individual object get an unique id
            bpy.context.view_layer.objects.active = obj
            material = object_name
            materials = list(bpy.context.object.material_slots.keys())
            material_index = materials.index(material)
            
            bpy.context.object.active_material_index = material_index                     
            bpy.ops.material.new()
            new_material_name = bpy.context.object.active_material.name

            self.set_object_color(obj.name, new_material_name, self.class_to_rgb(class_nr,i))
            
    
    
    
    def color_image_to_instance_image(self, color_image=None, load_path= None, save_path=None, delete_orig_color_image=False):
        """
        This function converts a color image to an instance image, where each instance has a unique color
        the color is defined in class_to_rgb.
        input color_image: numpy array of shape (h,w,3) where each pixel is a color
        load_path(optional): path to the color image to load, will replace the color image input
        
        """       
        
        assert color_image is None or load_path is None, "Only one of color_image or load_path should be provided"
        
        if load_path is not None:
            color_image = np.array(Image.open(load_path))
        
        instance_image = np.zeros((color_image.shape[0], color_image.shape[1]), dtype=np.int64)
        
        green_channel = color_image[:,:,1].astype(np.int64)
        blue_channel = color_image[:,:,2].astype(np.int64)
        
        instance_image = self.rgb_to_instance_id((0,green_channel, blue_channel, 255))
        instance_image = instance_image.astype(np.uint8)
        
        if save_path is not None:
            file_type = os.path.splitext(save_path)[1]
            save_path = save_path.replace(file_type, ".npy")
            np.save(save_path, instance_image)
            
        if delete_orig_color_image:
            os.remove(load_path)
            
        
        
        return instance_image, save_path

    
if __name__ == "__main__":
        
    #  some test code 
    import os 
    from custom_render_utils import *
    x = SimpleInstancer()
    y = custom_render_utils()
    x.set_object_color("Cube", (0,230,230,255))
    image_folder_path= os.path.join(os.getcwd())
    image_name = "test.png"
    y.simple_render(image_folder_path, image_name)
    image = x.load_image(y.image_path_dict[image_name])
    instance_image = x.color_image_to_instance_image(image)
    
    
    
    
    
    
    




