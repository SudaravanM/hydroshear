from collections import defaultdict
import hydra
import numpy as np
import os
import torch

from isaacgym import gymapi, torch_utils
from rl.tasks.factory.factory_schema_class_env import FactoryABCEnv
from rl.tasks.factory.factory_base import FactoryBase
from rl.tasks.factory.factory_schema_config_env import FactorySchemaConfigEnv
from rl.tasks.tacsl.tacsl_base import TacSLBase
from rl.tasks.tacsl.tacsl_env_insertion import TacSLSensors
from rl.utils.urdf_object import add_box_geometry, ObjectABC
from rl.hydroshear_sensors.hydroshear_sensors import HydroFotsSensor, HydroFotsFieldSensor
from rl.fots_sensors.fots_sensors import FotsSensor, FotsFieldSensor

import cv2
import open3d as o3d
import xml.etree.ElementTree as ET
'''
    Author: An Dang
    
    Description:
        Environment code for insertion tasks in ideal scenario for Amazon Warehouse Dense Packing
'''

from abc import ABC, abstractmethod

def add_bin_geometry(link, freespace_width, freespace_height, freespace_depth, thickness=0.01, rgba=(0.6,0.6,0.6,1.0)):
    '''
    y
    ^
    |
    |----> x
    (frame)
    
                <---------height-------->
                        left wall (does not fully span left and right because of front/back wall)
                -------------------------
                | |-------------------| |
                | |                   | |
    back wall   | |   bottom of bin   | | front wall (spans to the top and bottom of bin)
                | |   (freespace)     | |
                | |                   | |
                | |                   | |
                | |-------------------| |
                -------------------------
                        right wall
    freespace is the dimensions of the cube volume where objects can be placed.
    height and width are flipped because +x is front of robot (meaning it will be the height of bin from where robot is facing).
    
    '''
    
    # add bottom of bin
    bottom_bin_origin = (0,0,-freespace_depth/2-thickness/2)
    bottom_bin_params = (freespace_height, freespace_width, thickness)
    rgba = (0.6, 0.6, 0.6, 1.0)  # gray color
    add_box_geometry(link, bottom_bin_params, bottom_bin_origin, rgba)

    # add front wall of bin
    front_wall_origin = (freespace_height/2 + thickness/2, 0, -thickness/2)
    front_wall_params = (thickness, freespace_width + thickness*2, freespace_depth + thickness)
    add_box_geometry(link, front_wall_params, front_wall_origin, rgba)
    
    # add back wall of bin
    back_wall_origin = (-freespace_height/2 - thickness/2, 0, -thickness/2)
    back_wall_params = (thickness, freespace_width + thickness*2, freespace_depth + thickness)
    add_box_geometry(link, back_wall_params, back_wall_origin, rgba)
    
    # add left wall of bin
    left_wall_origin = (0, freespace_width/2 + thickness/2, -thickness/2)
    left_wall_params = (freespace_height, thickness, freespace_depth + thickness)
    add_box_geometry(link, left_wall_params, left_wall_origin, rgba)
    
    # add right wall of bin
    right_wall_origin = (0, -freespace_width/2 - thickness/2, -thickness/2)
    right_wall_params = (freespace_height, thickness, freespace_depth + thickness)
    add_box_geometry(link, right_wall_params, right_wall_origin, rgba)

def autogenerate_bin_urdf(filename, freespace_width, freespace_height, freespace_depth, thickness=0.01):
    root = ET.Element('robot')
    root.set('name', 'bin')
    
    link = ET.SubElement(root, 'link')
    link.set('name', 'bin_body')

    add_bin_geometry(link, freespace_width, freespace_height, freespace_depth, thickness=thickness)
    
    tree = ET.ElementTree(root)
    tree.write(filename, encoding='utf-8', xml_declaration=True)
    
class Bin(ObjectABC):
    def __init__(self, gym, initial_bin_pose, freespace_width=0.3, freespace_height=0.21, freespace_depth=0.024, thickness=0.01):
        super().__init__(gym)
        self.initial_bin_pose = initial_bin_pose  # gymapi.Transform()
        self.texture_filename = './assets/furniture/mesh/textures/wood1.jpeg'
        self.freespace_width = freespace_width
        self.freespace_height = freespace_height
        self.freespace_depth = freespace_depth
        self.thickness = thickness  # thickness of the bin walls
        self.autogenerate_fn = autogenerate_bin_urdf  # function to autogenerate the bin urdf file
    
    def initialize_pose(self):
        raise NotImplementedError("Bin does not need to initialize poses, it is static")
    
    def setup_asset(self, sim):
        bin_options = gymapi.AssetOptions()
        bin_options.flip_visual_attachments = False
        bin_options.fix_base_link = True
        bin_options.thickness = 0.0  # default = 0.02
        bin_options.armature = 0.0  # default = 0.0
        bin_options.use_physx_armature = True
        bin_options.linear_damping = 0.0  # default = 0.0
        bin_options.max_linear_velocity = 1000.0  # default = 1000.0
        bin_options.angular_damping = 0.0  # default = 0.5
        bin_options.max_angular_velocity = 64.0  # default = 64.0
        bin_options.disable_gravity = False
        bin_options.enable_gyroscopic_forces = True
        bin_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        bin_options.use_mesh_materials = False
        urdf_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'urdf')
        self.autogenerate_fn(os.path.join(urdf_root, 'bin.urdf'), 
                              freespace_width=self.freespace_width,
                              freespace_height=self.freespace_height,
                              freespace_depth=self.freespace_depth,
                              thickness=self.thickness
        )
        bin_asset  = self.gym.load_asset(sim, urdf_root, 'bin.urdf', bin_options)
        self.assets['bin'] = bin_asset
        
        texture_img = cv2.imread(self.texture_filename)
        texture_img = cv2.cvtColor(texture_img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        texture_img = np.dstack((texture_img, np.ones((texture_img.shape[0], texture_img.shape[1]), dtype=np.uint8) * 255))  # add alpha channel
        H,W, _ = texture_img.shape
        texture_img = texture_img.reshape((H, W*4))
        self.texture_handle = self.gym.create_texture_from_buffer(sim, W, H, texture_img)
        # self.texture_handle = self.gym.create_texture_from_file(sim, self.texture_filename)
        
    def create_actor(self, env_ptr0, env_ptr, actor_count, collision_group_id, friction = 1.0):
        ## create bin 
        bin_handle = self.gym.create_actor(env_ptr, self.assets['bin'], self.initial_bin_pose, 'bin', collision_group_id, 0, 0)
        self.actor_handles['bin'] = bin_handle
        self.actor_ids_sim['bin'].append(actor_count)
        
        bin_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, bin_handle)
        bin_shape_props[0].friction = friction # higher
        bin_shape_props[0].rolling_friction = 0.0  # default = 0.0
        bin_shape_props[0].torsion_friction = 0.0  # default = 0.0
        bin_shape_props[0].restitution = 0.0  # default = 0.0
        bin_shape_props[0].compliance = 0.0  # default = 0.0
        bin_shape_props[0].thickness = 0.0  # default = 0.0
        self.gym.set_actor_rigid_shape_properties(env_ptr, bin_handle, bin_shape_props)
        
        if self.actor_id_env is None:
            self.actor_id_env = self.gym.find_actor_index(env_ptr, 'bin', gymapi.DOMAIN_ENV)
            self.bin_rb_names = self.gym.get_actor_rigid_body_names(env_ptr0, self.actor_id_env)
            self.body_id_env = self.gym.find_actor_rigid_body_index(env_ptr0, self.actor_id_env,
                                                                        self.bin_rb_names[0], gymapi.DOMAIN_ENV)
        
        actor_rigid_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, self.actor_id_env, self.bin_rb_names[0], gymapi.DOMAIN_ACTOR)
        self.gym.set_rigid_body_texture(env_ptr, self.actor_handles['bin'], actor_rigid_body_id_env, gymapi.MESH_VISUAL_AND_COLLISION, self.texture_handle)
        
        actor_count += 1
        return actor_count

class Cube(ObjectABC):
    def __init__(self, gym, cube_name, cube_size):
        super().__init__(gym)
        self.cube_name = cube_name
        self.size = cube_size # size of cube in meters, e.g. 0.05 for 5cm cube
        
    def get_surface_points(self, mesh_points=1000):
        o3d_cube = o3d.geometry.TriangleMesh.create_box(width=self.size, height=self.size, depth=self.size)
        min_init_sample_points = 200
        sample_num_points = max(min_init_sample_points, 2 * mesh_points) # increase sampling points to ensure enough points are sampled
        surface_points = np.asarray(o3d_cube.sample_points_uniformly(number_of_points=sample_num_points).points) # perform uniform sampling on the cube surface
        surface_points = np.random.permutation(surface_points)[:mesh_points] # randomly sample mesh_points points from the surface points
        surface_points = torch.tensor(surface_points, dtype=torch.float32)  # convert to torch tensor
        surface_points = surface_points - self.size / 2.0 # center of cube from o3d is at bottom left corner of lower half, so we need to shift it to center
        return surface_points

    def initialize_pose(self):
        raise NotImplementedError
    def setup_asset(self, sim):
        cube_options = gymapi.AssetOptions()
        cube_options.density = 1000.0
        cube_options.flip_visual_attachments = False
        cube_options.fix_base_link = False
        cube_options.thickness = 0.0  # default = 0.02
        cube_options.armature = 0.0  # default = 0.0
        cube_options.use_physx_armature = True
        cube_options.linear_damping = 0.0  # default = 0.0
        cube_options.max_linear_velocity = 1000.0  # default = 1000.0
        cube_options.angular_damping = 0.0  # default = 0.5
        cube_options.max_angular_velocity = 64.0  # default = 64.0
        cube_options.disable_gravity = False
        cube_options.enable_gyroscopic_forces = True
        cube_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        cube_options.use_mesh_materials = False
        # cube_asset = self.gym.create_box(sim, self.cube_size, self.cube_size, self.cube_size, cube_options)
        # load cube asset from urdf
        
        assert self.size == 0.05, "Cube size must be 0.05 meters (5cm) for the URDF to work correctly"
        urdf_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'urdf')
        cube_asset = self.gym.load_asset(sim, urdf_root, 'cube.urdf', cube_options)
        self.urdf_path = os.path.join(urdf_root, 'cube.urdf')
        self.assets[self.cube_name] = cube_asset
        
    def create_actor(self, env_ptr0, env_ptr, actor_count, collision_group_id, friction=0.5):
        ## create cube actors with properties
        cube_pose = gymapi.Transform()
        cube_pose.p.x = 0.7
        cube_pose.p.y = 0.0
        cube_pose.p.z = 0.025
        cube_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        cube_handle = self.gym.create_actor(env_ptr, self.assets[self.cube_name], cube_pose, self.cube_name, collision_group_id, 0, 0)
        self.actor_handles[self.cube_name] = cube_handle
        self.actor_ids_sim[self.cube_name].append(actor_count)
    
        cube_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, cube_handle)
        cube_shape_props[0].friction = friction
        cube_shape_props[0].rolling_friction = 0.0  # default = 0.0
        cube_shape_props[0].torsion_friction = 0.0  # default = 0.0
        cube_shape_props[0].restitution = 0.0  # default = 0.0
        cube_shape_props[0].compliance = 0.0  # default = 0.0
        cube_shape_props[0].thickness = 0.0  # default = 0.0
        self.gym.set_actor_rigid_shape_properties(env_ptr, cube_handle, cube_shape_props)

        # set mass
        cube_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, cube_handle)
        cube_rb_props[0].mass = 0.5 # set to 500g (like real-world)
        self.gym.set_actor_rigid_body_properties(env_ptr, cube_handle, cube_rb_props, recomputeInertia=True)
        
        if self.actor_id_env is None:
            self.actor_id_env = self.gym.find_actor_index(env_ptr, self.cube_name, gymapi.DOMAIN_ENV)
            cube_rb_names = self.gym.get_actor_rigid_body_names(env_ptr0, self.actor_id_env)
            self.body_id_env = self.gym.find_actor_rigid_body_index(env_ptr0, self.actor_id_env,
                                                                        cube_rb_names[0], gymapi.DOMAIN_ENV)
        
        actor_count += 1
        return actor_count

    
class ObjectSetupABC(ABC):
    def __init__(self, gym, freespace_width, freespace_height, freespace_depth):
        self.gym = gym
        self.freespace_width = freespace_width
        self.freespace_height = freespace_height
        self.freespace_depth = freespace_depth
        self.assets = {}
        self.actor_handles = {}
        self.actor_ids_sim = defaultdict(list)
        self.actor_id_env_list = []
        self.body_id_env_list = []
        self.sizes = []
        self.created_once = False
        self.geometric_centers = None
    @abstractmethod
    def __len__(self):
        raise NotImplementedError
    @abstractmethod
    def initialize_poses(self):
        raise NotImplementedError
    @abstractmethod
    def setup_assets(self, sim):
        raise NotImplementedError
    @abstractmethod
    def create_actors(self, env_ptr0, env_ptr, actor_count, collision_group_id):
        raise NotImplementedError
    def get_freespace_parameters(self):
        min_x = -self.freespace_height / 2
        max_x = self.freespace_height / 2
        min_y = -self.freespace_width / 2
        max_y = self.freespace_width / 2
        min_z = -self.freespace_depth / 2
        max_z = self.freespace_depth / 2
        return (min_x, min_y, min_z), (max_x, max_y, max_z)

class PresetCubeSetup(ObjectSetupABC):
    '''
    
    Bin space
    ------------------  -> y
    |                |
    |   Free Space   | HEIGHT
    |                |   x
    |                |   ^
    ------------------   |
          WIDTH
    
    -------------------------------------
    |           |           |           |
    |  Cube0    |   Cube1   |   Cube2   |
    |           |           |           |
    |           |           |           |
    -------------------------------------
    |           |           |///////////|
    |   Cube3   |  Cube4    |///////////|
    |           |           |///////////| <----- FREE SLOT
    |           |           |///////////|
    -------------------------------------
    
                 size
            -------------
            |           |
            |   Cube5   | <--- grasped by robot
            |           |
            |           |
            -------------
          
    /| Bin        -----         -----
    /| Wall       |   | <- p -> |   |
    /| <- p ->    -----         -----
    /|
    

    z = height of bin
    W = WIDTH
    H = HEIGHT
    s = size
    p = padding
    
    # cubes per row is = floor( (W-p) / (s + p))
    # cubes per column is = floor( (H-p) / (s + p))
    '''
    def __init__(self, 
                 initial_bin_pose,
                 cube_size,
                 gym,
                 freespace_width,
                 freespace_height,
                 freespace_depth,
                 padding_x = 0.0005,
                 padding_y = 0.0005,
                 squish_padding=0.001,
                 squish_randomize_range=(0.0,0.0),
                 squish_vert=False,
                 random_squish_prob=0.5,
                 use_random_squish_dir=True,
                 squish_single=False,
                 random_single_squish=0.5,
                 use_random_squish_single=True,
                 ):
        # by default give cubes half mm of clearance when initializing
        super().__init__(gym, freespace_width, freespace_height, freespace_depth)
        self.cube_size = cube_size
        self.padding_y = padding_y
        self.padding_x = padding_x
        self.initial_bin_pose = initial_bin_pose  # gymapi.Transform()
        
        self.cubes_per_row = int( (freespace_width - padding_y) / (cube_size + padding_y))
        self.cubes_per_column = int( (freespace_height - padding_x) / (cube_size + padding_x))
        
        self.num_cubes = self.cubes_per_row * self.cubes_per_column - 1 # -1 because we leave one slot empty for insertion
        self.sizes = [[self.cube_size, self.cube_size, self.cube_size] for _ in range(len(self))]  # sizes of the cubes
        self.squish_padding = squish_padding  # padding to squish the cubes together, if needed
        self.geometric_centers = torch.zeros(len(self), 3, dtype=torch.float32)  # geometric centers of the books
        self.squish_randomize_range = squish_randomize_range  # range to randomize the squish padding
        
        self.random_squish_prob = random_squish_prob
        self.use_random_squish_dir = use_random_squish_dir
        self.squish_vert = squish_vert
        
        self.random_single_squish = random_single_squish
        self.use_random_squish_single = use_random_squish_single
        self.squish_single = squish_single
    
    def get_surface_points(self, mesh_points=1000):
        o3d_cube = o3d.geometry.TriangleMesh.create_box(width=self.cube_size, height=self.cube_size, depth=self.cube_size)
        min_init_sample_points = 200
        sample_num_points = max(min_init_sample_points, 2 * mesh_points) # increase sampling points to ensure enough points are sampled
        surface_points = np.asarray(o3d_cube.sample_points_uniformly(number_of_points=sample_num_points).points) # perform uniform sampling on the cube surface
        surface_points = np.random.permutation(surface_points)[:mesh_points] # randomly sample mesh_points points from the surface points
        surface_points = torch.tensor(surface_points, dtype=torch.float32)  # convert to torch tensor
        surface_points = surface_points - self.cube_size / 2.0 # center of cube from o3d is at bottom left corner of lower half, so we need to shift it to center
        return surface_points
    
    def __len__(self):
        return self.num_cubes

    def setup_assets(self, sim):
        cube_options = gymapi.AssetOptions()
        # cube_options.density = 1000.0
        cube_options.flip_visual_attachments = False
        cube_options.fix_base_link = False
        cube_options.thickness = 0.0  # default = 0.02
        cube_options.armature = 0.0  # default = 0.0
        cube_options.use_physx_armature = True
        cube_options.linear_damping = 0.0  # default = 0.0
        cube_options.max_linear_velocity = 1000.0  # default = 1000.0
        cube_options.angular_damping = 0.0  # default = 0.5
        cube_options.max_angular_velocity = 64.0  # default = 64.0
        cube_options.disable_gravity = False
        cube_options.enable_gyroscopic_forces = True
        cube_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        cube_options.use_mesh_materials = False
        cube_asset = self.gym.create_box(sim, self.cube_size, self.cube_size, self.cube_size, cube_options)
        self.assets['cube'] = cube_asset
    def reset_colors(self, env_ptr):
        
        for i in range(len(self)):
            rb_names = self.gym.get_actor_rigid_body_names(env_ptr, self.actor_id_env_list[i])
            actor_rigid_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, self.actor_id_env_list[i], rb_names[0], gymapi.DOMAIN_ACTOR)
            # randomize colors
            random_color = gymapi.Vec3(torch.rand(1).item(), torch.rand(1).item(), torch.rand(1).item())
            self.gym.set_rigid_body_color(env_ptr, self.actor_handles[f'obj_{i}'], actor_rigid_body_id_env, gymapi.MESH_VISUAL_AND_COLLISION, random_color)
            # current_color = self.gym.get_rigid_body_color(env_ptr, self.actor_handles[f'cube_{i}'], actor_rigid_body_id_env, gymapi.MESH_VISUAL)
        
    def create_actors(self, env_ptr0, env_ptr, actor_count, collision_group_id, friction = 0.5):
        ## create cube actors with properties
        bin_pos = torch.tensor([self.initial_bin_pose.p.x, self.initial_bin_pose.p.y, self.initial_bin_pose.p.z], dtype=torch.float32)
        bin_quat = torch.tensor([self.initial_bin_pose.r.x, self.initial_bin_pose.r.y, self.initial_bin_pose.r.z, self.initial_bin_pose.r.w], dtype=torch.float32)
        cube_poses, _, _ = self.initialize_poses(bin_pos=bin_pos, bin_quat=bin_quat)
        for cube_idx in range(len(self)):
            cube_pose = cube_poses[cube_idx]
            cube_handle = self.gym.create_actor(env_ptr, self.assets['cube'], cube_pose, f'obj_{cube_idx}', collision_group_id, 0, 0)
            self.actor_handles[f'obj_{cube_idx}'] = cube_handle
            self.actor_ids_sim[f'obj_{cube_idx}'].append(actor_count)

            ## add properties to cube actor
            cube_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, cube_handle)
            cube_shape_props[0].friction = friction
            cube_shape_props[0].rolling_friction = 0.0  # default = 0.0
            cube_shape_props[0].torsion_friction = 0.0  # default = 0.0
            cube_shape_props[0].restitution = 0.0  # default = 0.0
            cube_shape_props[0].compliance = 0.0  # default = 0.0
            cube_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, cube_handle, cube_shape_props)
            actor_count += 1
            
        if not self.created_once:
            ## get body_id_env_list and actor_id_env_list which is useful root pos/quat
            for cube_idx in range(len(self)):
                cube_actor_id_env = self.gym.find_actor_index(env_ptr, f'obj_{cube_idx}', gymapi.DOMAIN_ENV)
                self.actor_id_env_list.append(cube_actor_id_env)
                
                cube_rb_names = self.gym.get_actor_rigid_body_names(env_ptr0, cube_actor_id_env)
                cube_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr0, cube_actor_id_env,
                                                                        cube_rb_names[0], gymapi.DOMAIN_ENV)
                self.body_id_env_list.append(cube_body_id_env)
        self.created_once = True
        
        self.reset_colors(env_ptr)
            
        return actor_count
    def initialize_poses(self, randomize=False, squish_goal=False, bin_pos=torch.zeros(3), bin_quat=torch.tensor([0.0,0.0,0.0,1.0]), return_desired_packed_poses=False):
        
        ## row-major packing
        (min_x, min_y, _), (max_x, max_y, _) = self.get_freespace_parameters()
        # x is related to height
        # y is related to width
        
        # /| <- po -> | cube 0 | <- p -> | cube 1| <- p -> | cube 2 | <- po -> |\
        p_x = self.padding_x
        p_y = self.padding_y
        po_y = ((max_y - min_y) - ((self.cube_size * self.cubes_per_row) + (p_y * (self.cubes_per_row - 1))))/2.0
        po_x = ((max_x - min_x) - ((self.cube_size * self.cubes_per_column) + (p_x * (self.cubes_per_column - 1))))/2.0
        
        pos_ys = torch.linspace(min_y + po_y + self.cube_size/2, max_y - self.cube_size/2 - po_y, self.cubes_per_row)
        pos_xs = torch.linspace(min_x + po_x + self.cube_size/2, max_x - self.cube_size/2 - po_x, self.cubes_per_column)
        
        # pos_ys = torch.linspace(min_y + self.padding + self.cube_size/2, max_y - self.cube_size/2 - self.padding, self.cubes_per_row)
        # pos_xs = torch.linspace(min_x + self.padding + self.cube_size/2, max_x - self.cube_size/2 - self.padding, self.cubes_per_column)
        
        cube_positions = torch.zeros((self.cubes_per_row, self.cubes_per_column, 3), dtype=torch.float32)  # shape (cubes_per_row, cubes_per_column, 3)
        cube_positions[:,:,0] = pos_xs.unsqueeze(0).repeat(self.cubes_per_row, 1)  # x positions
        cube_positions[:,:,1] = pos_ys.unsqueeze(1).repeat(1, self.cubes_per_column)  # y positions
        z_off = self.initial_bin_pose.p.z + self.padding_x + 0.025
        
        choose_goal_location = torch.randint(0, len(self)+1,(1,)).item() if randomize else len(self)
        if return_desired_packed_poses:
            world_cube_pos = torch_utils.tf_apply(bin_quat, bin_pos, cube_positions) # new poses
            world_cube_pos[:, :, 2] = self.initial_bin_pose.p.z - self.freespace_depth/2.0 + self.cube_size / 2.0
            world_cube_quat = bin_quat
            desired_packed_poses = [torch.cat([world_cube_pos[i % self.cubes_per_row, i // self.cubes_per_row], world_cube_quat]) for i in range(len(self)+1) if i != choose_goal_location]
            desired_packed_poses = torch.stack(desired_packed_poses, dim=0)  # shape (num_cubes, 7)


        keypoint_dim = 0 # 0 => x dir, 1 => y dir, 2 => z dir
        if squish_goal:
            
            guard = np.random.rand() < self.random_squish_prob if self.use_random_squish_dir else not self.squish_vert
            guard2 = np.random.rand() < self.random_single_squish if self.use_random_squish_single else self.squish_single
            for i in range(len(self)+1):
                '''
                    SQUISH GOAL:
                    - if squish_goal is True, we take neighboring cubes of the goal cube (row-wise) and make them close to the goal cube
                    - this is intended to block the goal cube space from being reached by the robot
                    - forces the robot to push the cubes out of the way to reach the goal cube
                '''
                if guard:
                    keypoint_dim = 0
                    # row-wise squish
                    if i // self.cubes_per_row == choose_goal_location // self.cubes_per_row:
                        random_squish = np.random.rand() * (self.squish_randomize_range[1] - self.squish_randomize_range[0]) + self.squish_randomize_range[0]
                        if i % self.cubes_per_row == (choose_goal_location % self.cubes_per_row) - 1:
                            cube_positions[i % self.cubes_per_row, i // self.cubes_per_row, 1] += self.cube_size/2 + self.padding_y - (self.squish_padding + random_squish)/2.0
                            if guard2:
                                break
                        elif i % self.cubes_per_row == (choose_goal_location % self.cubes_per_row) + 1:
                            cube_positions[i % self.cubes_per_row, i // self.cubes_per_row, 1] += -(self.cube_size/2 + self.padding_y - (self.squish_padding + random_squish)/2.0)
                            if guard2:
                                break
                else:
                    keypoint_dim = 1
                    # column-wise squish
                    if i % self.cubes_per_row == choose_goal_location % self.cubes_per_row:
                        random_squish = np.random.rand() * (self.squish_randomize_range[1] - self.squish_randomize_range[0]) + self.squish_randomize_range[0]
                        if i // self.cubes_per_row == (choose_goal_location // self.cubes_per_row) - 1:
                            cube_positions[i % self.cubes_per_row, i // self.cubes_per_row, 0] += self.cube_size/2 + self.padding_x - (self.squish_padding + random_squish)/2.0
                            if guard2:
                                break
                        elif i // self.cubes_per_row == (choose_goal_location // self.cubes_per_row) + 1:
                            cube_positions[i % self.cubes_per_row, i // self.cubes_per_row, 0] += -(self.cube_size/2 + self.padding_x - (self.squish_padding + random_squish)/2.0)
                            if guard2:
                                break
                '''
                if (i == choose_goal_location-1 or i == choose_goal_location+1) and i // self.cubes_per_row == choose_goal_location // self.cubes_per_row:
                    random_squish = np.random.rand() * (self.squish_randomize_range[1] - self.squish_randomize_range[0]) + self.squish_randomize_range[0]
                    # if i == choose_goal_location-1:
                    #     cube_positions[i % self.cubes_per_row, i // self.cubes_per_row, 1] += (self.cube_size/2 + self.padding_y - (self.squish_padding  + random_squish)/2.0)
                    #     # edge case is when choose_goal_location+1 is not on the same row
                    #     if not (0<=(choose_goal_location+1)<=len(self)) or i // self.cubes_per_row != (choose_goal_location+1) // self.cubes_per_row:
                    #         cube_positions[i % self.cubes_per_row, i // self.cubes_per_row, 1] += self.cube_size/2 - (self.squish_padding  + random_squish)/2.0
                    # else:
                    #     cube_positions[i % self.cubes_per_row, i // self.cubes_per_row, 1] += -(self.cube_size/2 + self.padding_y - (self.squish_padding  + random_squish)/2.0)
                    #     if not (0<=(choose_goal_location-1)<=len(self)) or i // self.cubes_per_row != (choose_goal_location-1) // self.cubes_per_row:
                    #         cube_positions[i % self.cubes_per_row, i // self.cubes_per_row, 1] += -(self.cube_size/2 - (self.squish_padding  + random_squish)/2.0)

                    cube_positions[i % self.cubes_per_row, i // self.cubes_per_row, 1] += (1 if i == choose_goal_location-1 else -1) * (self.cube_size/2 + self.padding_y - (self.squish_padding  + random_squish)/2.0)
                '''
                
        
        # cube_positions = torch.stack((pos_xs, pos_ys, pos_zs), dim=1)  # shape (num_cubes+1, 3)
        cube_positions = torch_utils.tf_apply(bin_quat, bin_pos, cube_positions) # new poses
        cube_quat = gymapi.Quat(bin_quat[0].item(), bin_quat[1].item(), bin_quat[2].item(), bin_quat[3].item())
        cube_poses = []
        for i in range(len(self)+1):
            if i == choose_goal_location:
                goal = torch.zeros(7)
                goal[:2] = cube_positions[i % self.cubes_per_row, i // self.cubes_per_row, :2]
                goal[2] = self.initial_bin_pose.p.z - (self.freespace_depth / 2.0) + (self.cube_size / 2.0)
                goal[3:] = torch.tensor([cube_quat.x, cube_quat.y, cube_quat.z, cube_quat.w], dtype=torch.float32)
            else:
                cube_pose = gymapi.Transform()
                cube_pose.p.x = cube_positions[i % self.cubes_per_row, i // self.cubes_per_row, 0]
                cube_pose.p.y = cube_positions[i % self.cubes_per_row, i // self.cubes_per_row, 1]
                cube_pose.p.z = z_off
                cube_pose.r = cube_quat
                cube_poses.append(cube_pose)
        if return_desired_packed_poses:
            return cube_poses, goal, keypoint_dim, desired_packed_poses
        return cube_poses, goal, keypoint_dim

class BinEnvPacking(TacSLBase, TacSLSensors, HydroFotsFieldSensor, FotsFieldSensor, FactoryABCEnv):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render,
                 thickness=0.01):
        
        #bin pose
        self.bin_pose = gymapi.Transform()
        self.thickness = thickness
        self.bin = None
        
        self.packed_objects = None # will be initialized in create_envs()
        
        self.insertion_obj = None
        
        self._get_env_yaml_params()
        self.freespace_width, self.freespace_height, self.freespace_depth = self.cfg_task.env.get("freespace_dims",[0.23, 0.23, 0.04])
        # this init will run .create_envs()
        FactoryBase.__init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        
        self.acquire_base_tensors()  # defined in superclass
        self._acquire_env_tensors()
        self.refresh_base_tensors()  # defined in superclass
        self.refresh_env_tensors()
        self.nominal_tactile = None
        
        elastomer_urdf_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'tacsl', 'urdf')
        indenter_urdf_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'urdf')
        
        hydrofots_cfg = self.cfg_task.sensor.hydroshear
        fots_cfg = self.cfg_task.sensor.fots
        old_fots_cfg = self.cfg_task.sensor.get("oldfots", None)
        
        if self.cfg_task.env.get('use_shear_force', False):
            if self.cfg_task.env.get('use_hydrosoft_model', False):
                hydrofots_left = HydroFotsSensor(
                    hydrofots_cfg=hydrofots_cfg,
                    elastomer_urdf_root=elastomer_urdf_root,
                    indenter_urdf_root=indenter_urdf_root,
                    elastomer_urdf=self.cfg_task.env.franka_urdf_file,
                    elastomer_link_name='elastomer_left',
                    indenter_urdf=self.plug_file,
                    indenter_link_name=self.plug_file.split('.')[0],
                    device=self.device,
                    elastomer_resolution=self.cfg_task.env.get('elastomer_resolution', 0.001),
                    indenter_resolution=self.cfg_task.env.get('indenter_resolution', 0.001)
                )
                hydrofots_right = HydroFotsSensor(
                    hydrofots_cfg=hydrofots_cfg,
                    elastomer_urdf_root=elastomer_urdf_root,
                    indenter_urdf_root=indenter_urdf_root,
                    elastomer_urdf=self.cfg_task.env.franka_urdf_file,
                    elastomer_link_name='elastomer_right',
                    indenter_urdf=self.plug_file,
                    indenter_link_name=self.plug_file.split('.')[0],
                    device=self.device,
                    indenter_sdf_sensor=hydrofots_left.indenter_sdf_sensor, # share sdf sensor between left and right
                    elastomer_resolution=self.cfg_task.env.get('elastomer_resolution', 0.001),
                    indenter_resolution=self.cfg_task.env.get('indenter_resolution', 0.001)
                )
                HydroFotsFieldSensor.__init__(self, 
                    hydrofots_sensors=[hydrofots_left, hydrofots_right],
                    elastomer_link_names=['elastomer_left', 'elastomer_right'],
                    elastomer_actor_names=['franka', 'franka'],
                    # visualize=True
                )
            elif self.cfg_task.env.get('use_fots_model', False) or self.cfg_task.env.get('use_old_fots_model', False):
                
                sensor_fots_cfg = fots_cfg if self.cfg_task.env.get('use_fots_model', False) else old_fots_cfg
                fots_left = FotsSensor(
                    fots_cfg=sensor_fots_cfg,
                    elastomer_urdf_root=elastomer_urdf_root,
                    indenter_urdf_root=indenter_urdf_root,
                    elastomer_urdf=self.cfg_task.env.franka_urdf_file,
                    elastomer_link_name='elastomer_left',
                    indenter_urdf=self.plug_file,
                    indenter_link_name=self.plug_file.split('.')[0],
                    device=self.device,
                    use_original_implementation=self.cfg_task.env.get('use_old_fots_model', False)
                )
                
                fots_right = FotsSensor(
                    fots_cfg=sensor_fots_cfg,
                    elastomer_urdf_root=elastomer_urdf_root,
                    indenter_urdf_root=indenter_urdf_root,
                    elastomer_urdf=self.cfg_task.env.franka_urdf_file,
                    elastomer_link_name='elastomer_right',
                    indenter_urdf=self.plug_file,
                    indenter_link_name=self.plug_file.split('.')[0],
                    device=self.device,
                    indenter_sdf_sensor=fots_left.indenter_sdf_sensor, # share sdf sensor between left and right
                    use_original_implementation=self.cfg_task.env.get('use_old_fots_model', False)
                )
                FotsFieldSensor.__init__(self,
                    fots_sensors=[fots_left, fots_right],
                    elastomer_link_names=['elastomer_left', 'elastomer_right'],
                    elastomer_actor_names=['franka', 'franka'],
                )
                
    def set_elastomer_compliance(self, compliance_stiffness, compliant_damping):
        for elastomer_link_name in ['elastomer_left', 'elastomer_right']:
            self.configure_compliant_dynamics(actor_handle=self.actor_handles['franka'],
                                              elastomer_link_name=elastomer_link_name,
                                              compliance_stiffness=compliance_stiffness,
                                              compliant_damping=compliant_damping,
                                              use_acceleration_spring=False)
    
    def set_friction_damping_params(self, joint_friction=None, joint_damping=None):
        """
        Set friction and damping parameters for the robot joints.

        Args:
            joint_friction: Friction values for the joints.
            joint_damping: Damping values for the joints.
        """
        if joint_friction is None and joint_damping is None:
            return

        for env_id in range(self.num_envs):
            env_ptr, franka_handle = self.env_ptrs[env_id], self.actor_handles['franka']

            franka_dof_props = self.gym.get_actor_dof_properties(env_ptr, franka_handle)

            if joint_friction is not None:
                franka_dof_props['friction'][:9] = joint_friction[:9]
            if joint_damping is not None:
                franka_dof_props['damping'][:9] = joint_damping[:9]

            self.gym.set_actor_dof_properties(env_ptr, franka_handle, franka_dof_props)        
    ## FactoryABCEnv methods
    #########################################
    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""
        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_env', node=FactorySchemaConfigEnv)
        config_path = 'task/BinEnvPacking.yaml'  # relative to Gym's Hydra search path (cfg dir)
        
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env['task']  # strip superfluous nesting

    def create_envs(self):
        """Set env options. Import assets. Create actors."""
        bin_pos = self.cfg_task.randomize.initial_bin_pos if "initial_bin_pos" in self.cfg_task.randomize else [0.5, 0.0, 0.0]
        bin_rot = self.cfg_task.randomize.initial_bin_rot if "initial_bin_rot" in self.cfg_task.randomize else [0.0, 0.0, 0.0]  # euler angles in radians
        bin_quat = torch_utils.quat_from_euler_xyz(torch.tensor(bin_rot[0]), torch.tensor(bin_rot[1]), torch.tensor(bin_rot[2]))
        self.bin_pose.p.x = bin_pos[0]
        self.bin_pose.p.y = bin_pos[1]
        self.bin_pose.p.z = self.freespace_depth / 2 + self.thickness
        self.bin_pose.r = gymapi.Quat(bin_quat[0], bin_quat[1], bin_quat[2], bin_quat[3])
        
        packed_obj_padding = 0.006
        self.packed_objects = PresetCubeSetup(cube_size=0.05, gym=self.gym, initial_bin_pose=self.bin_pose,
                                            freespace_width=self.freespace_width, freespace_height=self.freespace_height,
                                            freespace_depth=self.freespace_depth, padding_x=packed_obj_padding, padding_y=packed_obj_padding,
                                            squish_padding=self.cfg_task.randomize.preset_object_squish_padding,
                                            squish_randomize_range=self.cfg_task.randomize.preset_object_squish_noise_range,
                                            squish_vert=self.cfg_task.randomize.get("preset_object_squish_vert", False),
                                            random_squish_prob=self.cfg_task.randomize.get("preset_object_random_squish_prob", 0.5),
                                            use_random_squish_dir=self.cfg_task.randomize.get("preset_object_use_random_squish_dir", False),
                                            squish_single=self.cfg_task.randomize.get("preset_object_squish_single", False),
                                            use_random_squish_single=self.cfg_task.randomize.get("preset_object_use_random_squish_single", False),
                                            random_single_squish=self.cfg_task.randomize.get("preset_object_random_single_squish", 0.5)
                                            )

        self.bin = Bin(gym=self.gym, initial_bin_pose=self.bin_pose, freespace_width=self.freespace_width,
                       freespace_height=self.freespace_height, freespace_depth=self.freespace_depth,
                       thickness=self.thickness)
        
        self.insertion_obj = Cube(gym=self.gym, cube_name='plug', cube_size=0.05)
        
        lower = gymapi.Vec3(-self.asset_info_franka_table.table_depth * 0.6,
                            -self.asset_info_franka_table.table_width * 0.6,
                            0.0)
        upper = gymapi.Vec3(self.asset_info_franka_table.table_depth * 0.6,
                            self.asset_info_franka_table.table_width * 0.6,
                            self.asset_info_franka_table.table_height)
        num_per_row = int(np.sqrt(self.num_envs))

        self.print_sdf_warning()
        self.assets = dict()
        self.asset_file_paths = dict()
        self.assets['franka'], self.assets['table'] = self.import_franka_assets()
        self._import_env_assets()
        self._create_actors(lower, upper, num_per_row)
        self.parse_controller_spec()
        
        self.plug_actor_id_env = self.insertion_obj.actor_id_env
        self.asset_file_paths['plug'] = self.insertion_obj.urdf_path
        self.plug_file = 'cube.urdf'
        self._create_sensors()
    
    def _import_env_assets(self):
        self.insertion_obj.setup_asset(self.sim)
        self.bin.setup_asset(self.sim)
        self.packed_objects.setup_assets(self.sim)
        
        # merge assets
        self.assets.update(self.packed_objects.assets) # merge dicts
        self.assets.update(self.bin.assets)  # merge dicts
        self.assets.update(self.insertion_obj.assets)  # merge dicts
    
    def _create_actors(self, lower, upper, num_per_row):
        
        # initialize franka and table  transforms 
        franka_pose = gymapi.Transform()
        franka_pose.p.x = 0.0
        franka_pose.p.y = 0.0
        franka_pose.p.z = 0.0
        franka_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        table_pose = gymapi.Transform()
        table_pose.p.x = self.asset_info_franka_table.robot_base_to_table_offset_x
        table_pose.p.y = 0.0
        table_pose.p.z = -self.asset_info_franka_table.table_height * 0.5
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        self.env_ptrs = []
        self.table_handles = []
        self.shape_ids = []
        self.table_actor_ids_sim = []  # within-sim indices
        self.actor_handles = {}
        self.actor_ids_sim = defaultdict(list)
        self.rbs_com = defaultdict(list)
        actor_count = 0
        
        for i in range(self.num_envs):
            '''
                Create environment.
                    -> create actors from assets
                        -> franka
                        -> 
            '''
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            
            ## create franka actor
            franka_handle = self.gym.create_actor(
                env_ptr,
                self.assets['franka'],
                franka_pose,
                'franka',
                i + self.num_envs if self.cfg_env.sim.disable_franka_collisions else i,
                0,
                0
            )
            self.actor_handles['franka'] = franka_handle
            self.actor_ids_sim['franka'].append(actor_count)
            actor_count += 1
            
            ## get ids for adding franka properties
            link7_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_link7', gymapi.DOMAIN_ACTOR)
            hand_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_hand', gymapi.DOMAIN_ACTOR)
            left_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_leftfinger',
                                                                  gymapi.DOMAIN_ACTOR)
            right_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_rightfinger',
                                                                   gymapi.DOMAIN_ACTOR)
            rb_ids = [link7_id, hand_id, left_finger_id, right_finger_id]
            rb_shape_indices = self.gym.get_asset_rigid_body_shape_indices(self.assets['franka'])
            self.shape_ids = [rb_shape_indices[rb_id].start for rb_id in rb_ids]
            
            ## add franka properties (friction)
            franka_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, franka_handle)
            for shape_id in self.shape_ids:
                franka_shape_props[shape_id].friction = self.cfg_base.env.franka_friction
                franka_shape_props[shape_id].rolling_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].torsion_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].restitution = 0.0  # default = 0.0
                franka_shape_props[shape_id].compliance = 0.0  # default = 0.0
                franka_shape_props[shape_id].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, franka_handle, franka_shape_props)
            self.franka_actor_id_env = self.gym.find_actor_index(env_ptr, 'franka', gymapi.DOMAIN_ENV)
            
            ## create table
            table_handle = self.gym.create_actor(env_ptr, self.assets['table'], table_pose, 'table', i, 0, 0)
            self.actor_handles['table'] = table_handle
            self.actor_ids_sim['table'].append(actor_count)
            actor_count += 1

            ## adding table property
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            table_shape_props[0].friction = self.cfg_base.env.table_friction
            table_shape_props[0].rolling_friction = 0.0  # default = 0.0
            table_shape_props[0].torsion_friction = 0.0  # default = 0.0
            table_shape_props[0].restitution = 0.0  # default = 0.0
            table_shape_props[0].compliance = 0.0  # default = 0.0
            table_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)
            
            # enabling dof sensors
            self.franka_num_dofs = self.gym.get_actor_dof_count(env_ptr, franka_handle)
            self.gym.enable_actor_dof_force_sensors(env_ptr, franka_handle)

            self.env_ptrs.append(env_ptr)
            
            actor_count = self.packed_objects.create_actors(self.env_ptrs[0], env_ptr, actor_count, i, friction=self.cfg_task.env.get("friction_preset_objs", 0.5))
            actor_count = self.bin.create_actor(self.env_ptrs[0], env_ptr, actor_count, i, friction=self.cfg_task.env.get("friction_bin", 1.0))
            actor_count = self.insertion_obj.create_actor(self.env_ptrs[0], env_ptr, actor_count, i, friction=self.cfg_task.env.get("friction_insertion_obj", 0.5))
            
        self.actor_handles.update(self.packed_objects.actor_handles)
        self.actor_ids_sim.update(self.packed_objects.actor_ids_sim)
        self.actor_handles.update(self.bin.actor_handles)
        self.actor_ids_sim.update(self.bin.actor_ids_sim)
        self.actor_handles.update(self.insertion_obj.actor_handles)
        self.actor_ids_sim.update(self.insertion_obj.actor_ids_sim)
        
        
        if self.cfg_task.env.use_compliant_contact:
            # Set compliance params
            self.set_elastomer_compliance(self.cfg_task.env.compliance_stiffness, self.cfg_task.env.compliant_damping)
        
        # do some post-processing
        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env
        self.actor_ids_sim_tensors = {key: torch.tensor(self.actor_ids_sim[key], dtype=torch.int32, device=self.device)
                                      for key in self.actor_ids_sim.keys()}

        ## franka actor rigid body index
        self.hand_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_hand',
                                                                     gymapi.DOMAIN_ENV)
        self.left_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_leftfinger',
                                                                    gymapi.DOMAIN_ENV)
        self.right_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                             'panda_rightfinger', gymapi.DOMAIN_ENV)
        
        self.left_fingertip_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                               'panda_leftfingertip',
                                                                               gymapi.DOMAIN_ENV)
        self.right_fingertip_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                'panda_rightfingertip',
                                                                                gymapi.DOMAIN_ENV)
        self.fingertip_centered_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                   'panda_fingertip_centered',
                                                                                   gymapi.DOMAIN_ENV)
        self.graspcenter_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                            'panda_finger_graspcenter',
                                                                            gymapi.DOMAIN_ENV)
        self.hand_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_hand',
                                                                           gymapi.DOMAIN_ACTOR)
        self.left_finger_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                  'panda_leftfinger',
                                                                                  gymapi.DOMAIN_ACTOR)
        self.right_finger_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                   'panda_rightfinger',
                                                                                   gymapi.DOMAIN_ACTOR)
        self.left_fingertip_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                     'panda_leftfingertip',
                                                                                     gymapi.DOMAIN_ACTOR)
        self.right_fingertip_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                      'panda_rightfingertip',
                                                                                      gymapi.DOMAIN_ACTOR)
        self.fingertip_centered_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                         'panda_fingertip_centered',
                                                                                         gymapi.DOMAIN_ACTOR)
        self.graspcenter_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                'panda_finger_graspcenter',
                                                                                gymapi.DOMAIN_ACTOR)
        self.table_body_id = self.gym.find_actor_rigid_body_index(self.env_ptrs[0], self.actor_handles['table'],
                                                                  'box', gymapi.DOMAIN_ENV)
        
        self.franka_body_names = self.gym.get_actor_rigid_body_names(env_ptr, franka_handle)
        self.franka_body_ids_env = dict()
        for b_name in self.franka_body_names:
            self.franka_body_ids_env[b_name] = self.gym.find_actor_rigid_body_index(self.env_ptrs[0],
                                                                                    self.actor_handles['franka'],
                                                                                    b_name, gymapi.DOMAIN_ENV)
        
        
    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""

        self.franka_base_pos = self.root_pos[:, self.franka_actor_id_env, 0:3]
        self.franka_base_quat = self.root_quat[:, self.franka_actor_id_env, 0:4]
 
        self.packed_objects_pos = self.root_pos[:, self.packed_objects.actor_id_env_list, 0:3]
        self.packed_objects_quat = self.root_quat[:, self.packed_objects.actor_id_env_list, 0:4]
        self.packed_objects_linvel = self.root_linvel[:, self.packed_objects.actor_id_env_list, 0:3]
        self.packed_objects_angvel = self.root_angvel[:, self.packed_objects.actor_id_env_list, 0:3]
        self.packed_objects_origin_linvel = torch.zeros_like(self.packed_objects_linvel)
        
        self.bin_pos = self.root_pos[:, self.bin.actor_id_env, 0:3]
        self.bin_quat = self.root_quat[:, self.bin.actor_id_env, 0:4]
        self.bin_linvel = self.root_linvel[:, self.bin.actor_id_env, 0:3]
        self.bin_angvel = self.root_angvel[:, self.bin.actor_id_env, 0:3]
        self.bin_origin_linvel = torch.zeros_like(self.bin_linvel)
    
        self.insertion_obj_pos = self.root_pos[:, self.insertion_obj.actor_id_env, 0:3]
        self.insertion_obj_quat = self.root_quat[:, self.insertion_obj.actor_id_env, 0:4]
        self.insertion_obj_linvel = self.root_linvel[:, self.insertion_obj.actor_id_env, 0:3]
        self.insertion_obj_angvel = self.root_angvel[:, self.insertion_obj.actor_id_env, 0:3]
        self.insertion_obj_origin_linvel = torch.zeros_like(self.insertion_obj_linvel)
        
        self.plug_pos = self.insertion_obj_pos
        self.plug_quat = self.insertion_obj_quat
    
    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.
        self.identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).unsqueeze(0).expand(self.num_envs, 4)
    #########################################