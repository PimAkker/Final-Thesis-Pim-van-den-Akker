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
        if self.delete_duplicates:
            # delete all objects which are copied
            [bpy.data.objects.remove(obj) for obj in bpy.data.objects if "." in obj.name]
        
        bpy.ops.object.select_all(action='DESELECT')
    
    def place_room(self,inst_id=255):
        """ 
        this function places the room in the scene. 
        input: inst_id: instance id of the room
        output: None
        
        """
        default_location = (0,0,0)   
        object_name = "room"
        self.blend_deselect_all()
        bpy.data.objects[object_name].select_set(True)
        # set object as active
        bpy.context.view_layer.objects.active = bpy.data.objects[object_name]

        bpy.context.object.modifiers["GeometryNodes"]["Socket_6"] = np.random.randint(0, 100000)
        obj = bpy.context.active_object

        bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked":False, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(45.2641, 3.86619, -3.15708), "orient_type":'GLOBAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'FACE_NEAREST'}, "use_snap_project":True, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
        bpy.ops.object.convert(target='MESH')
        bpy.context.view_layer.objects.active = bpy.data.objects[f'{object_name}.001']
        obj = bpy.context.active_object
        obj.location = default_location
        obj["inst_id"] = 255

    def place_tables(self,num_tables=5,inst_id=100,leg_nr_range=(4,4)):
        """ 
        this function places the tables in the scene. 
        input: num_tables: number of tables to be placed
                inst_id: instance id of the first table
                
        TODO: leg_nr_range: range of number of legs of the table
        TODO: better randomization of table placement
        TODO: better randomization of table rotation
        TODO: better randomization of table scale
        
        """
        for i in range(num_tables):
            # deselect all
            bpy.ops.object.select_all(action='DESELECT')
            bpy.data.objects['table square'].select_set(True)
            # set object as active
            bpy.context.view_layer.objects.active = bpy.data.objects['table square']
            rand_pos = (np.random.uniform(17, 40), np.random.uniform(7, -30), 0)
            bpy.ops.object.duplicate_move_linked(OBJECT_OT_duplicate={"linked":True, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(0,0,0), "orient_type":'GLOBAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'FACE_NEAREST'}, "use_snap_project":True, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
            
            obj = bpy.context.active_object
            obj.location = rand_pos
            bpy.ops.object.convert(target='MESH')
            obj = bpy.context.active_object
            obj["inst_id"] = inst_id+i
            
        # do bpy.data.objects["table square"].data.copy() to each instance of table square
        for obj in bpy.data.objects:
            if obj.name.startswith("table square"):
                obj.data = bpy.data.objects["table square"].data.copy()
                pass
    
    def place_raytrace(self, position=(2,1,0)):
        """ 
        this function places a raytrace in the scene. 
        input: position: tuple x, y,z position of the raytrace
        output: None
        
        """
        object_name = "raytrace"
        bpy.data.objects[object_name].select_set(True)
        bpy.context.view_layer.objects.active = bpy.data.objects[object_name]
        bpy.ops.object.duplicate_move_linked(OBJECT_OT_duplicate={"linked":True, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":position, "orient_type":'GLOBAL', "orient_matrix":((1, 0, 0), (0, 1, 0), (0, 0, 1)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'FACE_NEAREST'}, "use_snap_project":True, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
        bpy.context.view_layer.objects.active = bpy.data.objects[f'{object_name}.001']
        obj = bpy.context.active_object
        obj.location = position
        
    
    def blend_deselect_all(self):
        try:
            for ob in bpy.context.selected_objects:
                ob.select = False
            bpy.ops.outliner.item_activate(deselect_all=True)
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
            
