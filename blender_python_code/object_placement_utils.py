import cv2
import bpy
import bpycv
import random
import numpy as np

class object_placement:
    """ This class is used to place objects in the scene. It is used to place the room and tables in the scene.
    TODO: make this a more general class which can place any object in the scene
    """
    def __init__(self, delete_duplicates=False):
        self.blend_deselect_all()
        self.delete_duplicates = delete_duplicates
        
        # delete all objects which are copied
        [bpy.data.objects.remove(obj) for obj in bpy.data.objects if "." in obj.name]
        
        bpy.ops.object.select_all(action='DESELECT')
        
        self.room_center = bpy.data.objects["Walls"].location
        self.default_location = (0,0,0)   
    
    def set_object_id(self, object_name, inst_id):
        """ 
        Gives unique object id to all the copies of a given object"""
        
        # get a list of the objects which are copies of the given object
        objects = [obj for obj in bpy.data.objects if object_name+"." in obj.name]
        for i in range(len(objects)):
            objects[i]["inst_id"] = inst_id+i
        
        
        
    def place_walls(self,inst_id=255):
        """ 
        this function places the room in the scene. 
        input: inst_id: instance id of the room
        output: None
        
        """
        
        object_name = "Walls"
        self.blend_deselect_all()
        bpy.data.objects[object_name].select_set(True)
        # set object as active
        bpy.context.view_layer.objects.active = bpy.data.objects[object_name]

        bpy.context.object.modifiers["GeometryNodes"]["Socket_9"] = np.random.randint(0, 100000)
        obj = bpy.context.active_object

        bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked":False, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(45.2641, 3.86619, -3.15708), "orient_type":'GLOBAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'FACE_NEAREST'}, "use_snap_project":True, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
        bpy.ops.object.convert(target='MESH')
        bpy.context.view_layer.objects.active = bpy.data.objects[f'{object_name}.001']
        obj = bpy.context.active_object
        obj.location = self.default_location
        
        self.set_object_id(object_name=object_name, inst_id=inst_id)
        
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
        self.set_object_id(object_name=object_name, inst_id=inst_id)
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
        self.set_object_id(object_name=object_name, inst_id=inst_id)
        
        


    
    def place_raytrace(self, position=(0,0,0)):
        """ 
        this function places a raytrace in the scene. 
        input: position: tuple x, y,z position of the raytrace
        output: None
        
        """
        object_name = "raytrace"
        bpy.data.objects[object_name].select_set(True)
        bpy.context.view_layer.objects.active = bpy.data.objects[object_name]
        cur_pos = bpy.data.objects[object_name].location
        bpy.ops.object.duplicate_move_linked(OBJECT_OT_duplicate={"linked":True, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(0,0,0), "orient_type":'GLOBAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'FACE_NEAREST'}, "use_snap_project":True, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
        bpy.context.view_layer.objects.active = bpy.data.objects[f'{object_name}.001']
        obj = bpy.context.active_object
        bpy.ops.object.convert(target='MESH')
        obj.location = cur_pos-self.room_center

       
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
        
    def delete_random(self, object_type_name="Chairs display", delete_percentage=1):
        """Deletes a number of random objects from the scene"""
        
        # get all objects of object_type_name
        objects = [obj for obj in bpy.data.objects if object_type_name+"." in obj.name]
        
        # calculate the number of objects to delete
        delete_amount = int(len(objects)*delete_percentage)
        
    
        # delete delete_amount random objects
        for _ in range(delete_amount):
            obj = random.choice(objects)
            bpy.data.objects.remove(obj)
            objects.remove(obj)

        
        
        
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
        
        walls = bpy.data.objects["Walls"]
        bbox = walls.bound_box
        height = np.abs(bbox[4][2]) + np.abs(bbox[0][2])
        depth = np.abs(bbox[4][1]) + np.abs(bbox[0][1])
        width = np.abs(bbox[4][0]) + np.abs(bbox[0][0])
        return height, width, depth
    
        
    
    def blend_deselect_all(self):
        try:
            for ob in bpy.context.selected_objects:
                ob.select = False
        except:
            pass
        try:
            bpy.ops.outliner.item_activate(deselect_all=True)
        except:
            pass
        try:
            bpy.ops.object.select_all(action='DESELECT')
        except:
            pass    
     
        
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