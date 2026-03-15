from isaacgym import gymapi

import os
import cv2
import torch
import numpy as np
import open3d as o3d
import xml.etree.ElementTree as ET
from collections import defaultdict
from abc import ABC, abstractmethod


class ObjectABC(ABC):
    def __init__(self, gym):
        self.gym = gym
        self.assets = {}
        self.actor_handles = {}
        self.actor_ids_sim = defaultdict(list)
        self.actor_id_env = None
        self.body_id_env = None
        self.size = None # size is arbitrary, depends on the object
        self.geometric_center = torch.zeros(3)

    @abstractmethod
    def initialize_pose(self):
        raise NotImplementedError

    @abstractmethod
    def setup_asset(self, sim):
        raise NotImplementedError

    @abstractmethod
    def create_actor(self, env_ptr0, env_ptr, actor_count, collision_group_id):
        raise NotImplementedError

def add_box_geometry(parent, box_params, origin, rgba):
    # box_params: (length_x, length_y, length_z)
    # origin: (x,y,z)
    # rgba: (r,g,b,a)
    collision = ET.SubElement(parent, 'collision')
    origin_elem = ET.SubElement(collision, 'origin')
    origin_elem.set('xyz', f'{origin[0]} {origin[1]} {origin[2]}')
    geometry = ET.SubElement(collision, 'geometry')
    box = ET.SubElement(geometry, 'box')
    box.set('size', f'{box_params[0]} {box_params[1]} {box_params[2]}')
    
    visual = ET.SubElement(parent, 'visual')
    visual_origin = ET.SubElement(visual, 'origin')
    visual_origin.set('xyz', f'{origin[0]} {origin[1]} {origin[2]}')    
    visual_geometry = ET.SubElement(visual, 'geometry')
    visual_box = ET.SubElement(visual_geometry, 'box')
    visual_box.set('size', f'{box_params[0]} {box_params[1]} {box_params[2]}')
    
    material = ET.SubElement(visual, 'material')
    material.set('name', 'c')
    material_color = ET.SubElement(material, 'color')
    material_color.set('rgba', f'{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}')

def autogenerate_obstacle_urdf(filename, obstacle_width, obstacle_height, obstacle_depth):
    root = ET.Element('robot')
    root.set('name', 'obstacle')
    
    link = ET.SubElement(root, 'link')
    link.set('name', 'obstacle_body')

    add_box_geometry(link, (obstacle_width, obstacle_height, obstacle_depth), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))
    
    tree = ET.ElementTree(root)
    tree.write(filename, encoding='utf-8', xml_declaration=True)

class Obstacle(ObjectABC):
    def __init__(self, gym, initial_obstacle_pose, obstacle_width, obstacle_height, obstacle_depth):
        super().__init__(gym)
        self.initial_obstacle_pose = initial_obstacle_pose  # gymapi.Transform()
        self.texture_filename = '../assets/furniture/mesh/textures/wood1.jpeg'
        self.obstacle_width = obstacle_width
        self.obstacle_height = obstacle_height
        self.obstacle_depth = obstacle_depth
        self.autogenerate_fn = autogenerate_obstacle_urdf  # function to autogenerate the bin urdf file

    def initialize_pose(self):
        raise NotImplementedError("Bin does not need to initialize poses, it is static")
    
    def setup_asset(self, sim):

        obstacle_options = gymapi.AssetOptions()
        obstacle_options.flip_visual_attachments = False
        obstacle_options.fix_base_link = True
        obstacle_options.thickness = 0.0  # default = 0.02
        obstacle_options.armature = 0.0  # default = 0.0
        obstacle_options.use_physx_armature = True
        obstacle_options.linear_damping = 0.0  # default = 0.0
        obstacle_options.max_linear_velocity = 1000.0  # default = 1000.0
        obstacle_options.angular_damping = 0.0  # default = 0.5
        obstacle_options.max_angular_velocity = 64.0  # default = 64.0
        obstacle_options.disable_gravity = False
        obstacle_options.enable_gyroscopic_forces = True
        obstacle_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        obstacle_options.use_mesh_materials = False
        urdf_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'urdf') # path: rl/assets/urdf
        self.autogenerate_fn(os.path.join(urdf_root, 'obstacle.urdf'), 
                              obstacle_width=self.obstacle_width,
                              obstacle_height=self.obstacle_height,
                              obstacle_depth=self.obstacle_depth
        )
        self.urdf_path = os.path.join(urdf_root, 'obstacle.urdf')
        obstacle_asset  = self.gym.load_asset(sim, urdf_root, 'obstacle.urdf', obstacle_options)
        self.assets['obstacle'] = obstacle_asset
        
        texture_img = cv2.imread(self.texture_filename)
        texture_img = cv2.cvtColor(texture_img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        texture_img = np.dstack((texture_img, np.ones((texture_img.shape[0], texture_img.shape[1]), dtype=np.uint8) * 255))  # add alpha channel
        H, W, _ = texture_img.shape
        texture_img = texture_img.reshape((H, W*4))
        self.texture_handle = self.gym.create_texture_from_buffer(sim, W, H, texture_img)


    def create_actor(self, env_ptr0, env_ptr, actor_count, collision_group_id):
        ## create obstacle 
        obstacle_handle = self.gym.create_actor(env_ptr, self.assets['obstacle'], self.initial_obstacle_pose, 'obstacle', collision_group_id, 0, 0)
        self.actor_handles['obstacle'] = obstacle_handle
        self.actor_ids_sim['obstacle'].append(actor_count)
        
        obstacle_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, obstacle_handle)
        obstacle_shape_props[0].friction = 1.0 # higher
        obstacle_shape_props[0].rolling_friction = 0.0  # default = 0.0
        obstacle_shape_props[0].torsion_friction = 0.0  # default = 0.0
        obstacle_shape_props[0].restitution = 0.0  # default = 0.0
        obstacle_shape_props[0].compliance = 0.0  # default = 0.0
        obstacle_shape_props[0].thickness = 0.0  # default = 0.0
        self.gym.set_actor_rigid_shape_properties(env_ptr, obstacle_handle, obstacle_shape_props)
        
        if self.actor_id_env is None:
            self.actor_id_env = self.gym.find_actor_index(env_ptr, 'obstacle', gymapi.DOMAIN_ENV)
            self.obstacle_rb_names = self.gym.get_actor_rigid_body_names(env_ptr0, self.actor_id_env)
            self.body_id_env = self.gym.find_actor_rigid_body_index(env_ptr0, self.actor_id_env,
                                                                        self.obstacle_rb_names[0], gymapi.DOMAIN_ENV)
        
        actor_rigid_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, self.actor_id_env, self.obstacle_rb_names[0], gymapi.DOMAIN_ACTOR)
        self.gym.set_rigid_body_texture(env_ptr, self.actor_handles['obstacle'], actor_rigid_body_id_env, gymapi.MESH_VISUAL_AND_COLLISION, self.texture_handle)
        
        actor_count += 1
        return actor_count


class URDFObject(ObjectABC):
    def __init__(self, gym, name, urdf_filepath, scale=1.0):
        super().__init__(gym)
        self.name = name
        self.scale_factor = scale
        self.urdf_filepath = urdf_filepath
        self.size = None
        self.dims = None
    
    def get_surface_points(self, mesh_points=1000):
        min_init_sample_points = 200
        sample_num_points = max(min_init_sample_points, 2 * mesh_points) # increase sampling points to ensure enough points are sampled
        surface_points = np.asarray(self.mesh.sample_points_uniformly(number_of_points=sample_num_points).points) # perform uniform sampling on the cube surface
        surface_points = np.random.permutation(surface_points)[:mesh_points] # randomly sample mesh_points points from the surface points
        surface_points = torch.tensor(surface_points, dtype=torch.float32)  # convert to torch tensor
        return surface_points
    
    def initialize_pose(self):
        raise NotImplementedError

    def setup_asset(self, sim):
        asset_options = gymapi.AssetOptions()
        asset_options.density = 1000.0
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.thickness = 0.0  # default = 0.02
        asset_options.armature = 0.0  # default = 0.0
        asset_options.use_physx_armature = True
        asset_options.linear_damping = 0.0  # default = 0.0
        asset_options.max_linear_velocity = 1000.0  # default = 1000.0
        asset_options.angular_damping = 0.0  # default = 0.5
        asset_options.max_angular_velocity = 64.0  # default = 64.0
        asset_options.disable_gravity = False
        asset_options.enable_gyroscopic_forces = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.use_mesh_materials = True
        asset_options.vhacd_enabled = True
        asset_options.override_inertia = True
        asset_options.override_com = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        
        urdf_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'urdf')
        file_root = os.path.join(urdf_root, os.path.dirname(self.urdf_filepath))
        asset = self.gym.load_asset(sim, file_root, os.path.basename(self.urdf_filepath), asset_options)
        self.urdf_path = os.path.join(file_root, os.path.basename(self.urdf_filepath))
        
        self.assets[self.name] = asset

        
        urdf_tree = ET.parse(os.path.join(file_root, os.path.basename(self.urdf_filepath)))
        urdf_root = urdf_tree.getroot()
        urdf_robot = urdf_root[0]
        scale = None
        mesh_filename = None
        for child in urdf_robot:
            if child.tag == 'collision':
                for subchild in child:
                    if subchild.tag == 'geometry':
                        for subsubchild in subchild:
                            if subsubchild.tag == 'mesh':
                                mesh_filename = subsubchild.attrib['filename']
                                scale = subsubchild.attrib.get('scale', '1 1 1')
                                scale = np.array([float(s) for s in scale.split()])
        assert mesh_filename is not None, "Mesh filename not found in URDF file"
        assert scale is not None, "Scale not found in URDF file"
        
        self.mesh_path = os.path.join(file_root, mesh_filename)
        self.mesh = o3d.io.read_triangle_mesh(self.mesh_path)
        self.mesh.vertices = o3d.utility.Vector3dVector(np.asarray(self.mesh.vertices) * scale)
        self.mesh.compute_triangle_normals()
        self.dims = self.mesh.get_axis_aligned_bounding_box().get_extent()
        self.geometric_center = torch.tensor(self.mesh.get_axis_aligned_bounding_box().get_center())
        self.size = self.mesh.get_axis_aligned_bounding_box().get_extent()[2] # get z's extent
        
    def create_actor(self, env_ptr0, env_ptr, actor_count, collision_group_id):
        ## create cube actors with properties
        pose = gymapi.Transform()
        pose.p.x = 0.7
        pose.p.y = 0.0
        pose.p.z = 0.025
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        handle = self.gym.create_actor(env_ptr, self.assets[self.name], pose, self.name, collision_group_id, 0, 0)
        self.actor_handles[self.name] = handle
        self.actor_ids_sim[self.name].append(actor_count)
    
        shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
        shape_props[0].friction = 0.5
        shape_props[0].rolling_friction = 0.0  # default = 0.0
        shape_props[0].torsion_friction = 0.0  # default = 0.0
        shape_props[0].restitution = 0.0  # default = 0.0
        shape_props[0].compliance = 0.0  # default = 0.0
        shape_props[0].thickness = 0.0  # default = 0.0
        self.gym.set_actor_rigid_shape_properties(env_ptr, handle, shape_props)
        
        if self.actor_id_env is None:
            self.actor_id_env = self.gym.find_actor_index(env_ptr, self.name, gymapi.DOMAIN_ENV)
            rb_names = self.gym.get_actor_rigid_body_names(env_ptr0, self.actor_id_env)
            self.body_id_env = self.gym.find_actor_rigid_body_index(env_ptr0, self.actor_id_env,
                                                                        rb_names[0], gymapi.DOMAIN_ENV)
        self.gym.set_actor_scale(env_ptr, self.actor_handles[self.name], self.scale_factor)
        actor_count += 1
        return actor_count