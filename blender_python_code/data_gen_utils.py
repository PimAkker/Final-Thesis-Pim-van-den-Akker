import cv2
import bpy
import bpycv
import random
import numpy as np
import os
import time
class blender_object_placement:
    """ This class is used to place objects in the scene. It is used to place the room and tables in the scene.

    """
    def __init__(self, delete_duplicates=False, class_multiplier=1000):
        self.blend_deselect_all()
        self.delete_duplicates = delete_duplicates
        self.class_multiplier = class_multiplier
        # delete all objects which are copied
        [bpy.data.objects.remove(obj) for obj in bpy.data.objects if "." in obj.name]
        
        self.room_center = bpy.data.objects["walls"].location
        self.default_location = (0,0,0)   
        self.subset_selection = []
        self.modifier_identifier_dict = {}
        self.modifier_name_dict = {}
        self.modifier_data_type_dict = {}
        
        # make sure we are in object mode 
        assert bpy.context.mode == "OBJECT", "You are in the wrong mode, please switch to object mode in blender"
        
    def set_modifier(self, object_name, modifier_name, value):
        """ 
        this function sets the value of a modifier in the geometry nodes modifier of an object.
        input: 
            object_name: name of the object to set the modifier of type string
            modifier_name: name of the modifier to set of type string
            value: value to set the modifier to of type int, float, boolean or tuple,
            NOTE: random choice is not supported for boolean values 
            if tuple then the value is set to a random value between the tuple values if the modifier is of type int or float
            then the value is set to the that value.
        output: None
        """
        self.blend_deselect_all()
        bpy.data.objects[object_name].select_set(True)
        # select active object
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


        if type(value) == int or type(value) == float or type(value) == tuple:
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

        elif type(value) == int or type(value) == float:
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

        obj.data.update()


    def set_object_id(self,class_label, object_name = "walls", selection = None):
        """ 
        Gives an unique instance id to all objects of the same type in the scene.
        input class_label: label of the object type
        input object_name: (optional) name of the object to set the id of
        input selection: (optional) list of objects to set the id of
        output: None"""
        
        if selection is None:
            # get a list of the objects which are copies of the given object
            objects = [obj for obj in bpy.data.objects if object_name+"." in obj.name]
        else:
            objects = selection
        # Here there first two digits are the inst_id and 
        # are the type of object, the 3rd and 4rth digits are the type of mismatch
        # and the last 3 digits are the instance number of the object.
        for i in range(len(objects)):
            assert i<1000, f"there are too many {object_name} in the scene, there can be no more than 999 objects of the same type in the scene"
            objects[i]["inst_id"] = class_label*self.class_multiplier+i
            
    def set_object_color(self, object_name, color):
        """ 
        Sets the color of the object in the scene, the color is set in the emission node of the shader nodes. This means that
        the object will have the exact color specified
        input color: tuple of the color in rgb in range 0-255 (r,g,b, alpha)
        input object_name: name of the object to set the color of note that this has to be the same name as the shader material
        output: None
        """
        # set color from rgb to percentage
        
        assert object_name in bpy.data.objects, f"Object {object_name} does not exist"
        assert bpy.data.materials.get(object_name) is not None, f"Material {object_name} does not exist make sure that the object has a material with the same name as the object and that it is assigned in the geometry nodes of the object if the object is created with geometry nodes"

        assert len(color) == 4, f"Color: {color} is not of shape (r,g,b)"
        assert all([0 <= c <= 255 for c in color]), f"Color: {color} is not in the range 0-255"
        
        color = [c/255 for c in color]
        
        bpy.context.view_layer.objects.active = bpy.data.objects[object_name]
        bpy.data.materials[object_name].node_tree.nodes["Emission"].inputs[0].default_value = color

        
    def place_walls(self,inst_id=255):
        """ 
        this function places the room in the scene. 
        input: inst_id: instance id of the room
        output: None
        """
        object_name = "walls"
        self.blend_deselect_all()
        bpy.data.objects[object_name].select_set(True)
        # select active object
        bpy.context.view_layer.objects.active = bpy.data.objects[object_name]
        
                
        obj = bpy.context.active_object
        bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked":False, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(45.2641, 3.86619, -3.15708), "orient_type":'GLOBAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'FACE_NEAREST'}, "use_snap_project":True, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
        
        
        bpy.ops.object.convert(target='MESH')
        assert f'{object_name}.001' in bpy.data.objects, f"Object {object_name}.001 does not exist"
        bpy.context.view_layer.objects.active = bpy.data.objects[f'{object_name}.001']
        obj = bpy.context.active_object
        obj.location = self.default_location
        
        self.set_object_id(object_name=object_name, class_label=inst_id)
        
        
    def place_doors(self,inst_id=255):

    
        object_name = "doors"
        self.blend_deselect_all()
        bpy.data.objects[object_name].select_set(True)
        # set object as active
        bpy.context.view_layer.objects.active = bpy.data.objects[object_name]
        obj = bpy.context.active_object

        bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked":False, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(45.2641, 3.86619, -3.15708), "orient_type":'GLOBAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'FACE_NEAREST'}, "use_snap_project":True, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
        bpy.ops.object.convert(target='MESH')
        bpy.context.view_layer.objects.active = bpy.data.objects[f'{object_name}.001']
        obj = bpy.context.active_object
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        obj.location = self.default_location
        self.set_object_id(object_name=object_name, class_label=inst_id)
        
        
    def place_objects(self,inst_id=255, object_name=""):

        
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
        
        # Split mesh into individual objects
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        self.set_object_id(object_name=object_name, class_label=inst_id)
    
    def place_raytrace(self, position=(0,0,0)):
        """ 
        This function places a raytrace in the scene. 
        Input: position: tuple x, y,z position of the raytrace
        output: None
        
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
        self.raytrace_position = cur_pos-self.room_center
        obj.location = self.raytrace_position 
        
    def hide_objects(self, object_list):
        """ 
        this function hides a list of objects in the scene.
        input: object_list: list of objects to hide
        output: None
        """
        for obj in object_list:
            obj.hide_render = True
            
    def clean_up_materials(self):
        """
        Because of the way that the data is labeled in blender, the materials are not cleaned up properly.
        """
        # Remove materials from objects with a '.' in the name
        objects_to_remove_material = [obj for obj in bpy.data.objects if "." in obj.name]
        for obj in objects_to_remove_material:
            if obj.data.materials:
                obj.data.materials.pop(index=0)

        
    def isolate_object(self, object_name):
        """ 
        this function isolates an object in the scene by turning on hide render for all objects except object_name.
        This will make sure that only object_name is visible in the render.
        input: object_name: name of the object to be isolated
        output: None
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
        this function unisolates all objects in the scene. Showing everything in the render.
        input: object_name: name of the object to be unisolated
        output: None
        
        """
        
        for obj in bpy.data.objects:
            obj.hide_render = False
            
    def delete_single_object(self, object_name):
        """Deletes an object from the scene"""
        bpy.data.objects.remove(bpy.data.objects[object_name])
    
    def select_subset_of_objects(self,object_type_name="Chairs display", selection_percentage=1, bbox=None):
        """
        select a subset of objects of a given type or within a given bounding box
        input: object_type_name: name of the object type to select taken from the category_information.py file
        selection: percentage of objects to select
        within_bouding_box: (optional) tuple of the bounding box to select objects from
        output: tuple of all objects of the given type and the objects to select
        """
        object_type_name = object_type_name.lower()
        
        
        objects = [obj for obj in bpy.data.objects if object_type_name+"." in obj.name]
        # exclude objects that are in self.subset_selection
        objects = [obj for obj in objects if obj not in self.subset_selection]
        if bbox is not None:
            bbox = np.array(bbox)
            xmin, xmax = np.min(bbox[:, 0]), np.max(bbox[:, 0])
            ymin, ymax = np.min(bbox[:, 1]), np.max(bbox[:, 1])

            
            # select objects within the bounding box
            objects_in_bbox = []
            for obj in objects:
                if xmin <= obj.location.x <= xmax and ymin <= obj.location.y <= ymax:
                    objects_in_bbox.append(obj)
            
            objects = objects_in_bbox
            
        
        
        select_amount = int(len(objects)*selection_percentage)
        
        select_indexes = random.sample(range(len(objects)), select_amount)
        selected_objects = [objects[i] for i in select_indexes]
        self.subset_selection += selected_objects

        return selected_objects
    
    def delete_objects(self, object_list):
        """
        Deletes a number of random objects from the scene
        input: object_list: list of objects to delete
        output: None
        """
        for obj in object_list:
            bpy.data.objects.remove(obj)       
    
    def move_objects_relative(self, object_list, relative_position):  
        """
        move a list of objects in the scene relative to their current position
        input: object_list: list of objects to move
        input: relative_position: list of the relative position [x,y,z] to move the objects
        output: None
        """
        relative_position = np.array(relative_position)
        object_positions = [np.array(obj.location) for obj in object_list]
        for i in range(len(object_list)):
            object_list[i].location = object_positions[i] + relative_position

    
    def configure_camera(self, position=(0,0,0)):
        """ Set the camera position to the given position"""
        # Set the camera position to the given height
        bpy.data.objects["Camera"].location = position

        # Extract walls object and get the height width and depth
        
    def get_object_dims(self, object_name="walls"):
        """Get the dimensions of the object
        input: object_name: name of the object to get the dimensions of
        output: height, width, depth of the object
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
        input: None
        output: None
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
        this function cleans up the scene.
        input: None
        output: None
        
        """
        if self.delete_duplicates:
            self.delete_duplicates_func()    
        
        self.unisolate()
        self.blend_deselect_all()

        
def delete_folder_contents(masks_folder, images_folder,empty_folders=False):
    """ 
    this function deletes all files in the given folders
    input: masks_folder: folder to delete all files from
    input: images_folder: folder to delete all files from
    output: None
    """
    

    # ask the user if they are sure when there are more than 500 images in the folder
    if empty_folders:
        if len(os.listdir(masks_folder)) > 500:
            user_input = input(f"Are you sure you want to delete all {len(os.listdir(masks_folder))} files in {masks_folder} and {images_folder}? (y/n): ")
            if user_input.lower() == "y":    
                pass
            else:
                # crash out of the program
                raise ValueError("User did not confirm deletion of files")
        for folder in [masks_folder, images_folder]:
                    for file in os.listdir(folder):
                        os.remove(os.path.join(folder, file))
    
def overwrite_data(overwrite_folder, overwrite_data=False): 
    """
    Return the highest file number in the overwrite folder if 
    overwrite_data is False, else return 0
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
    Create folders if they do not exist
    input: paths: list of paths to create
    output: None
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)	
