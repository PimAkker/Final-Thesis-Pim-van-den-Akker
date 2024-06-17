import cv2
import bpy
import bpycv
import random
import numpy as np
import os
import time
from category_information import category_information, class_factor
from warnings import warn
import numbers
class blender_object_placement:
    """
    A class for managing object placement in Blender.

    Attributes:
 
    """


class blender_object_placement:
    
    def __init__(self, delete_duplicates=False):
        """
        Initializes the DataGeneratorUtils class.

        Args:
            delete_duplicates (bool): Whether to delete duplicate objects. This means that the scene will be 
            left empty after the script is run.
                delete_duplicates (bool): Whether to delete duplicate objects. This means that the scene will be 
                left empty after the script is run.
            class_multiplier (int): A multiplier for the class label when assigning object IDs. Should be globally 
                set in the category_information.py file.
            original_objects (list): A list of original objects in the scene. 
            room_center (tuple): The location of the room center.
            default_location (tuple): The default location for objects.
            subset_selection (list): A list of selected objects.
            modifier_identifier_dict (dict): A dictionary mapping object names to modifier identifiers.
            modifier_name_dict (dict): A dictionary mapping object names to modifier names.
            modifier_data_type_dict (dict): A dictionary mapping object names to modifier data types.
            original_obj_collection_name (str): The name of the collection containing original objects.
            temp_obj_collection_name (str): The name of the temporary collection.
            highest_instance_id_dict (dict): A dictionary mapping object names to the highest instance ID.
        """
        
        # make sure the user has opened a file in blender
        assert bpy.data.filepath != "", "No file is opened, have you opened a file in blender?"
        
        self.blend_deselect_all()
        self.delete_duplicates = delete_duplicates
        self.class_multiplier = class_factor
        # delete all objects which are copied
        [bpy.data.objects.remove(obj) for obj in bpy.data.objects if "." in obj.name]
        
        self.original_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']

        self.room_center = bpy.data.objects["walls"].location
        self.default_location = (0,0,0)   
        self.subset_selection = []
        self.modifier_identifier_dict = {}
        self.modifier_name_dict = {}
        self.modifier_data_type_dict = {}
        
        self.original_obj_collection_name = "placable objects"
        self.temp_obj_collection_name = "temp"
        self.highest_instance_id_dict = {}
        
        # make sure we are in object mode 
        assert bpy.context.mode == "OBJECT", "You are in the wrong mode, please switch to object mode in blender"
        
    def set_modifier(self, object_name, modifier_name, value):
        """ 
        Sets the value of a modifier in the geometry nodes modifier of an object.

        Args:
            object_name (str): Name of the object to set the modifier.
            modifier_name (str): Name of the modifier to set.
            value (int, float, bool, tuple): Value to set the modifier to.
                - If the modifier is of type int or float, the value can be a single value or a tuple representing a range.
                - If the modifier is of type bool, the value should be a boolean.
                - If the modifier is of any other type, a ValueError will be raised.

        Returns:
            None

        Raises:
            AssertionError: If the modifier_name is not in the list of supported modifiers.
            AssertionError: If the value is out of range for the modifier.

        Note:
            - Random choice is not supported for boolean values. So this should be done manually.
            - If the value is a tuple, the modifier value will be set to a random value within the tuple range if the modifier is of type int or float.
            - The object_name should be the same as the geometry node name. For example, if the object is named "walls", it should have the "walls" geometry node modifier.
        """
        self.blend_deselect_all()
        bpy.data.objects[object_name].select_set(True)
        
        bpy.context.view_layer.objects.active = bpy.data.objects[object_name]

        modifier_name = modifier_name.lower()
        
        geometry_nodes = bpy.data.objects[object_name].modifiers['GeometryNodes']
        
        if self.modifier_identifier_dict.get(object_name) is None:
            self.modifier_identifier_dict[object_name] = [input.identifier for input in geometry_nodes.node_group.inputs][1:]
            self.modifier_name_dict[object_name] = [input.name for input in geometry_nodes.node_group.inputs][1:]
            self.modifier_name_dict[object_name] = [name.lower() for name in self.modifier_name_dict[object_name]]
        
        modifier_identifier_list = self.modifier_identifier_dict[object_name]
        modifier_name_list = self.modifier_name_dict[object_name]


        assert modifier_name in modifier_name_list, f"modifier_name: {modifier_name} is not in the list of supported modifiers \
        choose from {modifier_name_list}"
        
        if self.modifier_data_type_dict.get(object_name) is None:
            self.modifier_data_type_dict[object_name] = [input.type for input in geometry_nodes.node_group.inputs][1:]
           
        modifier_data_type_list = self.modifier_data_type_dict[object_name]

        modifier_index = modifier_name_list.index(modifier_name)

        # get the identifier of the modifier
        modifier_identifier = modifier_identifier_list[modifier_index]
        modifier_number = int(modifier_identifier.split("_")[1])

        # retreive the maximum and minimum values of the modfier as defined in the geometry nodes modifier.
        # NOTE: for this to work the object_name should be the same as the geometry node name. So object 
        # "Walls" should have the "Walls" geometry node modifier


        if self.is_numeric(value) or type(value) == tuple:
            min_val = bpy.data.node_groups[object_name].inputs[modifier_index+1].min_value
            max_val = bpy.data.node_groups[object_name].inputs[modifier_index+1].max_value


        # if the modifier is a tuple then set the value as a random value bouded by the tuple

        if type(value) == tuple:
            assert value[0] >= min_val, f"Value: {value[0]} is too small for {modifier_name} modifier ensure that: {min_val} <= value <= {max_val}"
            assert value[1] <= max_val, f"Value: {value[1]} is too large for {modifier_name} modifier ensure that: {min_val} <= value <= {max_val}"
            
            if modifier_data_type_list[modifier_index] == "VALUE":
                value = np.random.uniform(float(value[0]), float(value[1]))
            elif modifier_data_type_list[modifier_index] == "INT":
                value = np.random.randint(int(value[0]), int(value[1])) 

        elif self.is_numeric(value):
            if modifier_data_type_list[modifier_index] == "VALUE":
                value = float(value)
            elif modifier_data_type_list[modifier_index] == "INT":
                value = int(value)
            assert value >= min_val, f"Value: {value} is too small for {modifier_name} modifier ensure that: {min_val} <= value <= {max_val}"
            assert value <= max_val, f"Value: {value} is too large for {modifier_name} modifier that: {min_val} <= value <= {max_val}"
        elif type(value) == np.bool_:
            value = bool(value)
        elif type(value) != bool:   
            raise ValueError(f"The value for {modifier_name}: {value} is not of type int, bool,  float or tuple")

        bpy.context.object.modifiers["GeometryNodes"][modifier_identifier] = value


        obj = bpy.context.active_object
        
        assert obj is not None, f"Object: {object_name} is a Nonetype, this may indicate that the object is set to hide in viewport, unhide the object in the viewport to set the modifier. To hide an object in the render use the hide_objects() function"        

        obj.data.update()
    def is_numeric(self,value):
        """Checks if a value is numeric, including both NumPy and Python numeric types. Excluding boolean types."""
        if type(value) == bool or type(value) == np.bool_:
            return False
        else:
            return (
                np.issubdtype(type(value), np.number)  # NumPy numeric types
                or isinstance(value, numbers.Number)    # Python numeric types
            )


        
    def set_object_id(self, class_label, object_name=None, selection=None):
        """ 
        Gives a unique instance id to all objects of the same type in the scene.

        Args:
            class_label (int): The label of the object type.
            object_name (str, optional): The name of the object to set the id of.
            selection (list, optional): A list of objects to set the id of.

        Returns:
            None
        """
        
        # Every time this function gets called this ensures that the object is a new instance
        if selection is None:
            # get a list of the objects which are copies of the given object
            objects = [obj for obj in bpy.data.objects if object_name+"." in obj.name]
        else:
            objects = selection

        for i in range(len(objects)):
            object_name = objects[i].name.split(".")[0]
            
            # check if the object has been given an instance id if not then give it one
            if object_name not in self.highest_instance_id_dict:
                self.highest_instance_id_dict[object_name] = 0
            else:    
                self.highest_instance_id_dict[object_name] += 1
                
            assert self.highest_instance_id_dict[object_name] < 1000, f"there are too many {object_name} in the scene, there can be no more than 999 objects of the same type in the scene"
            objects[i]["inst_id"] = class_label * self.class_multiplier + self.highest_instance_id_dict[object_name]
            
            
            
            
    def set_object_id_multiple(self, id_dictionary):
        """ 
        This function sets the object id of multiple objects in the scene.

        Args:
            id_dictionary (dict): A dictionary of the form {object_name: class_label}.

        Returns:
            None
        """
        for object_name, class_label in id_dictionary.items():
            self.set_object_id(class_label, object_name)
    
    def move_from_to_collection(self, object, from_collection, to_collection):
        """
        Moves an object from one collection to another in Blender.

        Args:
            object (bpy.types.Object): The object to move.
            from_collection (str): The name of the collection to move from.
            to_collection (str): The name of the collection to move to.
        """
        assert type(object) == bpy.types.Object, f"Object: {object} is not of type bpy.types.Object but of type {type(object)}"
        
        bpy.data.collections[from_collection].objects.unlink(object)
        bpy.data.collections[to_collection].objects.link(object)
        
    def set_object_color(self, object_name, color):
        """ 
        Sets the color of the object in the scene.

        The color is set in the emission node of the shader nodes, which means that
        the object will have the exact color specified. Shader should be set as in 
        :https://blender.stackexchange.com/questions/154471/render-solid-color-in-eevee
        and settings as per the first comment.

        Args:
            object_name (str): Name of the object to set the color of. Note that this
                has to be the same name as the shader material.
            color (tuple): Tuple of the color in RGB format, with values ranging from
                0 to 255 (r, g, b, alpha).

        Raises:
            AssertionError: If the object does not exist or if the material with the
                same name as the object does not exist. Also raised if the color tuple
                is not of shape (r, g, b, a) or if any color value is not in the range 0-255.

        Returns:
            None
        """
        # set color from RGB to percentage

        assert object_name in bpy.data.objects, f"Object {object_name} does not exist"
        assert bpy.data.materials.get(object_name) is not None, f"Material {object_name} does not exist. Make sure that the object has a material with the same name as the object and that it is assigned in the geometry nodes of the object if the object is created with geometry nodes"

        assert len(color) == 4, f"Color: {color} is not of shape (r, g, b, a)"
        assert all([0 <= c <= 255 for c in color]), f"Color: {color} is not in the range 0-255"

        color = [c/255 for c in color]

        bpy.context.view_layer.objects.active = bpy.data.objects[object_name]
        bpy.data.materials[object_name].node_tree.nodes["Emission"].inputs[0].default_value = color
    
    def duplicate_move(self, objects_list=[], relative_position=(0,0,0)):
        """ 
        Duplicates an object in the scene and moves it relative to its current position.
        
        Args:
            objects_list (list): A list of names or object classes of the objects to duplicate.
            relative_position (tuple): A tuple representing the relative position [x, y, z] to move the object.
        
        Returns:
            list: A list of the duplicated objects.
        """
        assert len(relative_position) == 3, "relative_position is not of length 3 give (x,y,z) to move object relative to"
        assert type(objects_list) == list, "objects_list is not of type list"
        
        self.blend_deselect_all()
        for object in objects_list:
            if type(object) == str:
                bpy.data.objects[object].select_set(True)
            else:
                object.select_set(True)
        
        # select active object
        bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked":False, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":relative_position, "orient_type":'GLOBAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'FACE_NEAREST'}, "use_snap_project":True, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
        
        return bpy.context.selected_objects
                                                           
    def place_objects(self, inst_id=255, object_name="", seperate_loose=True):
        """
        Place objects in the scene. This copies an object with name object_name and places it in the scene.
        However for ease of use the we call this as a function to place objects in the scene.

        Args:
            inst_id (int): Instance ID for the object.
            object_name (str): Name of the object to be placed. Must already exist in the scene.
            seperate_loose (bool): Flag indicating whether to split the mesh into individual objects. For example
            if an object consist of multiple chairs the flag will split the chairs into individual objects.
            Each seperate chair will have its own instance id.

        Returns:
            None
        """
        self.blend_deselect_all()
        bpy.data.objects[object_name].select_set(True)
        # set object as active
        bpy.context.view_layer.objects.active = bpy.data.objects[object_name]
        obj = bpy.context.active_object

        bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked":False, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(45.2641, 3.86619, -3.15708), "orient_type":'GLOBAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'FACE_NEAREST'}, "use_snap_project":True, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
        bpy.ops.object.convert(target='MESH')
        bpy.context.view_layer.objects.active = bpy.data.objects[f'{object_name}.001']
        obj = bpy.context.active_object
        obj.location = self.default_location
        self.move_from_to_collection(obj, self.original_obj_collection_name, self.temp_obj_collection_name)

        # Split mesh into individual objects
        if seperate_loose:
            bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        self.set_object_id(object_name=object_name, class_label=inst_id)
        
    def place_LiDAR(self, position=(0,0,0)):
        """ 
        This function places a raytrace in the scene. 
        Args:
            position (tuple): The x, y, z position of the raytrace.

        Returns:
            None
        """
        self.blend_deselect_all()
        object_name = "raytrace"
        bpy.data.objects[object_name].select_set(True)
        bpy.context.view_layer.objects.active = bpy.data.objects[object_name]
        cur_pos = bpy.data.objects[object_name].location
        bpy.ops.object.duplicate_move_linked(OBJECT_OT_duplicate={"linked":True, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(0,0,0), "orient_type":'GLOBAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'FACE_NEAREST'}, "use_snap_project":True, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
        bpy.context.view_layer.objects.active = bpy.data.objects[f'{object_name}.001']
        obj = bpy.context.active_object

        # bpy.ops.object.convert(target='MESH')
        self.raytrace_position = position
        obj.location = self.raytrace_position
        
    def hide_objects(self, object_list):
        """ 
        Hides a list of objects in the scene.

        Args:
            object_list (list): List of bpy objects to hide.
        Returns:
            None
        """
        for obj in object_list:
            obj.hide_render = True
            
    def unhide_objects(self, object_list):
        """ 
        Unhides a list of objects in the scene.

        Args:
            object_list (list): List of objects to unhide.

        Returns:
            None
        """
        for obj in object_list:
            obj.hide_render = False
            
    def clean_up_materials(self):
        """
        This method is used to clean up materials in Blender. Due to the way the data is labeled, 
        materials may not be cleaned up properly. This method identifies objects with a '.' in 
        their name and removes any associated materials from them. Prevents issues with rendering.

        Args:
            None

        Returns:
            None
        """
        # Remove materials from objects with a '.' in the name
        objects_to_remove_material = [obj for obj in bpy.data.objects if "." in obj.name]
        for obj in objects_to_remove_material:
            if obj.data.materials:
                obj.data.materials.pop(index=0)

        
    def isolate_object(self, object_name):
        """ 
        This function isolates an object in the scene by turning on hide render for all objects except object_name.
        This will make sure that only object_name is visible in the render.

        Args:
            object_name (str): Name of the object to be isolated.

        Returns:
            None
        """
        # only select the duplicate
        if "." not in object_name:
            object_name = f"{object_name}.001"
        
        # Ensure raytrace is not isolated
        self.unisolate()	
        
        for obj in bpy.data.objects:
            if obj.name != object_name:
                obj.hide_render = True
                  
    def unisolate(self):
        """ 
        Unisolates all objects in the scene, showing everything in the render.

        Parameters:
        - None

        Returns:
        - None
        """
        
        for obj in bpy.data.objects:
            obj.hide_render = False
            
    def delete_single_object(self, object_name):
        """Deletes an object from the scene.

        Args:
            object_name (str): The name of the object to be deleted.

        """
        bpy.data.objects.remove(bpy.data.objects[object_name])
    
    def select_subset_of_objects(self, object_type_name="Chairs display", selection_percentage=1, bbox=None):
        """
        Select a subset of objects of a given type or within a given bounding box.

        Args:
            object_type_name (str): Name of the object type to select, taken from the category_information.py file.
            selection_percentage (float): Percentage of objects to select (0-1).
            bbox (tuple): Optional tuple of the bounding box to select objects from. ()

        Returns:
            list: List of selected objects.

        """
        object_type_name = object_type_name.lower()

        objects = [obj for obj in bpy.data.objects if object_type_name + "." in obj.name]
        # Exclude objects that are in self.subset_selection
        objects = [obj for obj in objects if obj not in self.subset_selection]

        if bbox is not None:
            bbox = np.array(bbox)
            xmin, xmax = np.min(bbox[:, 0]), np.max(bbox[:, 0])
            ymin, ymax = np.min(bbox[:, 1]), np.max(bbox[:, 1])

            # Select objects within the bounding box
            objects_in_bbox = []
            for obj in objects:
                if xmin <= obj.location.x <= xmax and ymin <= obj.location.y <= ymax:
                    objects_in_bbox.append(obj)

            objects = objects_in_bbox

        select_amount = int(len(objects) * selection_percentage)
        select_indexes = random.sample(range(len(objects)), select_amount)
        selected_objects = [objects[i] for i in select_indexes]
        self.subset_selection += selected_objects

        return selected_objects
    
    def delete_objects(self, object_list=None, object_name=None):
        """
        Deletes either a list of objects or all objects of a given name from the scene.

        Args:
            object_list (list): List of objects to delete.
            object_name (str): Name of the object to delete. If None, all objects in object_list will be deleted.

        Returns:
            None
        """
        assert object_list is not None or object_name is not None, "Either object_list or object_name must be provided"
        assert not (object_list is not None and object_name is not None), "Only one of object_list or object_name should be provided"
        
        
        if object_name is not None:
            object_list = [obj for obj in bpy.data.objects if object_name in obj.name]
        
        for obj in object_list:
            bpy.data.objects.remove(obj)

    
    def move_objects_relative(self, object_list, relative_position):  
        """
        Move a list of objects in the scene relative to their current position.

        Args:
            object_list (list): List of bpy objects to move.
            relative_position (list or tuple): List of the relative position [x, y, z] to move the objects.

        Returns:
            None
        """
        relative_position = np.array(relative_position)
        object_positions = [np.array(obj.location) for obj in object_list]
        for i in range(len(object_list)):
            object_list[i].location = object_positions[i] + relative_position

    
    def configure_camera(self, position=(0,0,0)):
        """Set the camera position to the given position.

        Args:
            position (tuple): The position to set the camera to.
        """
        
        # Set the camera position to the given height
        bpy.data.objects["Camera"].location = position
        # TODO: implement camera dimensions
        
    def get_object_dims(self, object_name="walls"):
        """Get the dimensions of the object.

        Args:
            object_name (str): Name of the object to get the dimensions of.

        Returns:
            tuple: A tuple containing the bounding box, height, width, and depth of the object.
        """
        object = bpy.data.objects[object_name]
        bbox = object.bound_box
        height = np.abs(bbox[4][2]) + np.abs(bbox[0][2])
        depth = np.abs(bbox[4][1]) + np.abs(bbox[0][1])
        width = np.abs(bbox[4][0]) + np.abs(bbox[0][0])
        return bbox, height, width, depth
    
    def blend_deselect_all(self):
        """
        This function deselects all objects in the scene.
        
        Parameters:
        - self: The instance of the class.
        
        Returns:
        - None
        """
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        
    def delete_duplicates_func(self):
        """
        Deletes all objects which are copies of other objects, controlled
        by self.delete_duplicates
        """     
        [bpy.data.objects.remove(obj) for obj in bpy.data.objects if "." in obj.name]
    
    def finalize(self):
        """ 
        Cleans up the scene by performing various operations.
        
        This function is responsible for cleaning up the scene by performing
        various operations such as deleting duplicates, resetting dictionaries,
        and deselecting all objects. Should be called at the end of the script.
        
        Parameters:
            None
        
        Returns:
            None
        """
        if self.delete_duplicates:
            self.delete_duplicates_func()    
        
        self.highest_instance_id_dict = {}
        self.modifier_identifier_dict = {}
        self.modifier_name_dict = {}
        self.modifier_data_type_dict = {}
        
        self.unisolate()
        self.blend_deselect_all()
        self.subset_selection = []
        

        
def delete_folder_contents(folders, empty_folders=False):
    """ 
    Deletes all files in the given folders.

    Args:
        folders (list): List of folders to delete all files from.
        empty_folders (bool): Flag to confirm deletion of files. Defaults to False.

    Raises:
        ValueError: If the user does not confirm deletion of files. Will ask for confirmation if there are 
        more than 500 files. TODO: test if this still works in blender scripting

    Returns:
        None
    """
    # ask the user if they are sure when there are more than 500 files in the folders
    if empty_folders:
        total_files = sum(len(os.listdir(folder)) for folder in folders)
        if total_files > 500:
            user_input = input(f"Are you sure you want to delete all {total_files} files in the folders? (y/n): ")
            if user_input.lower() != "y":
                raise ValueError("User did not confirm deletion of files")
        for folder in folders:
            for file in os.listdir(folder):
                os.remove(os.path.join(folder, file))
    
def overwrite_data(overwrite_folder, overwrite_data=False): 
    """
    Return the highest file number in the overwrite folder if 
    overwrite_data is False, else return 0. Blender will automatically overwrite any file 
    with the same name in the folder (by default).

    Parameters:
    - overwrite_folder (str): The path to the folder where the data files are stored.
    - overwrite_data (bool): If True, the function will return 0. If False, the function 
      will return the highest file number in the folder plus 1.

    Returns:
    - file_number (int): The highest file number in the overwrite folder if overwrite_data is False, else 0.
    """
    file_number = 0
    if not overwrite_data:
        file_names = os.listdir(overwrite_folder)
        # split the files names by  "-"
        file_names = [file.split("-") for file in file_names]
        # get the file numbers
        if file_names != []:
            file_numbers = [int(file[-2]) for file in file_names]
            file_number = max(file_numbers)+1
                
    return file_number

def create_folders(paths):
    """
    Create folders if they do not exist.

    Args:
        paths (list): List of paths to create.

    Returns:
        None
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
def save_metadata(metadata_path= "",nr_of_images = 0, modifiers_list = [],time_taken = 0, ablation_parameter = []):
    import pandas as pd
    """
    Save the metadata to a csv file. Consisting of the ranges of data generation parameters, 
    the number of images, and the time taken to generate the dataset.

    Args:
        metadata_path (str): Path to save the metadata to.
        nr_of_images (int): Total number of images.
        modifiers_list (list): List of dictionaries containing the ranges of data generation parameters.
        time_taken (float): Time taken to generate the dataset.

    Returns:
        None
    """
    metadata_file = os.path.join(metadata_path, "metadata.txt")

    # Open the metadata file in write mode
    with open(metadata_file, "w") as f:
        if len(ablation_parameter) > 0:
            f.write(f'This dataset was created for the ablation study with the following parameter(s): {ablation_parameter}\n\n')
        f.write(f"This file contains the metadata for the generated dataset\n\n")
        f.write(f"This dataset was created on {time.ctime()}\n\n and took {time_taken} seconds to generate\n\n")
        f.write(f"Total number of images: {nr_of_images}\n\n")
        
        f.write("Ranges of data generation parameters:\n")
        for i,modifier in enumerate(modifiers_list):
            modifier_keys = modifier.keys()
            f.write("\n")
            f.write(f"")
            if 'color' in list(modifier_keys)[0].lower():
                f.write("Colors of objects in RGBA\n")
            for modifier_key in modifier_keys:
                
                f.write(f"{modifier_key}: {modifier[modifier_key]}\n")
                


    # Print a message to indicate that the metadata file has been created
    print(f"Metadata file created: {metadata_file}")