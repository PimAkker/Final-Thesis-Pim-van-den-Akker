import cv2
import bpy
import bpycv
import random
import numpy as np

class object_placement:
    """ This class is used to place objects in the scene. It is used to place the room and tables in the scene.

    """
    def __init__(self, delete_duplicates=False):
        self.blend_deselect_all()
        self.delete_duplicates = delete_duplicates
        
        # delete all objects which are copied
        [bpy.data.objects.remove(obj) for obj in bpy.data.objects if "." in obj.name]
        
        
        
        self.room_center = bpy.data.objects["Walls"].location
        self.default_location = (0,0,0)   
        
    def set_modifier(self, object_name, modifier_name, value):
        """ 
        this function sets the value of a modifier in the geometry nodes modifier of an object.
        input: 
            object_name: name of the object to set the modifier of type string
            modifier_name: name of the modifier to set of type string
            value: value to set the modifier to of type int, float or tuple, 
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
        
        modifier_identifier_list = [input.identifier for input in geometry_nodes.node_group.inputs][1:]
        
        modifier_name_list = [input.name for input in geometry_nodes.node_group.inputs][1:]
        modifier_name_list = [name.lower() for name in modifier_name_list]
        
        assert modifier_name in modifier_name_list, f"modifier_name: {modifier_name} is not in the list of supported modifiers \
        choose from {modifier_name_list}"
        
        modifier_data_type_list = [input.type for input in geometry_nodes.node_group.inputs][1:]


        modifier_index = modifier_name_list.index(modifier_name)
         
        # get the identifier of the modifier
        modifier_identifier = modifier_identifier_list[modifier_index]
        modifier_number = int(modifier_identifier.split("_")[1])
        
        # retreive the maximum and minimum values of the modfiier as defined in the geometry nodes modifier.
        # NOTE: for this to work the object_name should be the same as the geometry node name. So object 
        # "Walls" should have the "Walls" geometry node modifier

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
        
        else:   
            raise ValueError(f"Value: {value} is not of type int, float or tuple")
        
        bpy.context.object.modifiers["GeometryNodes"][modifier_identifier] = value


        obj = bpy.context.active_object
        obj.data.update()
        

    def set_object_id(self,class_label, object_name = "Walls", selection = None):
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
            objects[i]["inst_id"] = class_label*1000+i

        
    def place_walls(self,inst_id=255):
        """ 
        this function places the room in the scene. 
        input: inst_id: instance id of the room
        output: None
        
        """
        
        object_name = "Walls"
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

    
        object_name = "Doors"
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
        bpy.ops.object.convert(target='MESH')
        self.raytrace_position = cur_pos-self.room_center
        obj.location = self.raytrace_position
        
        
 
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
    
    def select_subset_of_objects(self,object_type_name="Chairs display", delete_percentage=1, bbox=None):
        """
        select a subset of objects of a given type or within a given bounding box
        input: object_type_name: name of the object type to delete
        delete_percentage: percentage of objects to delete
        within_bouding_box: (optional) tuple of the bounding box to delete objects from
        output: tuple of all objects of the given type and the objects to delete
        """
        
        
        objects = [obj for obj in bpy.data.objects if object_type_name+"." in obj.name]
        if bbox is not None:
            # get min max of x and y of the bounding box
            x_min =0
            x_max = 0
            y_min = 0
            y_max = 0
            for i in range(8):
                x_min = min(x_min, bbox[i][0])
                x_max = max(x_max, bbox[i][0])
                y_min = min(y_min, bbox[i][1])
                y_max = max(y_max, bbox[i][1])
            bbox = (x_min, x_max, y_min, y_max)
            
            object_bbox_corners = np.empty((0,3))
            for object in objects:
                object_bbox_corners = np.vstack((object_bbox_corners, np.array([np.array(corner) for corner in object.bound_box])))
            # Calculate the center of the bounding box
            
            object_bbox_corners = object_bbox_corners.reshape(-1, 8, 3)
            
            object_centers = np.sum(object_bbox_corners,axis=1) / 8
            objects_in_bbox =[]
            for i in range(len(objects)):
                if object_centers[i][0] > bbox[0] and object_centers[i][0] < bbox[1] and object_centers[i][1] > bbox[2] and object_centers[i][1] < bbox[3]:
                    objects_in_bbox.append(objects[i])
            
            objects = objects_in_bbox
            
        delete_amount = int(len(objects)*delete_percentage)
        
        delete_indexes = random.sample(range(len(objects)), delete_amount)
        # make list of objects to delete
        objects_to_delete = [objects[i] for i in delete_indexes]
        return objects_to_delete
    
    def delete_objects(self, object_list):
        """
        Deletes a number of random objects from the scene
        input: object_list: list of objects to delete
        output: None
        """
        for obj in object_list:
            bpy.data.objects.remove(obj)
            
            
    def configure_camera(self, position=(0,0,0)):
        """ Set the camera position to the given position"""
        # Set the camera position to the given height
        bpy.data.objects["Camera"].location = position

        # Extract walls object and get the height width and depth
        
    def get_object_dims(self, object_name="Walls"):
        """Get the dimensions of the object
        input: object_name: name of the object to get the dimensions of
        output: height, width, depth of the object
        """
        object_name = f"{object_name}"
        walls = bpy.data.objects[object_name]
        bbox = walls.bound_box
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
        if bpy.context.selected_objects != []:
            bpy.ops.object.select_all(action='DESELECT')

        
    def finalize(self):
        """ 
        this function cleans up the scene.
        input: None
        output: None
        
        """
        if self.delete_duplicates:
            # delete all objects which are copied
            [bpy.data.objects.remove(obj) for obj in bpy.data.objects if "." in obj.name]
            
        # turn on hide render for all objects except object_name
        self.unisolate()
        self.blend_deselect_all()