from collections import defaultdict
import hydra
import numpy as np
import os
from sympy import per
import torch

from isaacgym import gymapi, gymtorch, torch_utils
from isaacgym import torch_utils
import cv2
import open3d as o3d
import xml.etree.ElementTree as ET
from copy import deepcopy
from rl.tasks.bin_packing.bin_env_packing import ObjectABC, ObjectSetupABC, BinEnvPacking, Bin, add_box_geometry, add_bin_geometry


def autogenerate_shelf_urdf(filename, freespace_width, freespace_height, freespace_depth, inner_rows=0, thickness=0.01):
    # set inner_rows = 0, so that we only get 1 row of shelf like in real-world
    
    root = ET.Element('robot')
    root.set('name', 'shelf')
    
    link = ET.SubElement(root, 'link')
    link.set('name', 'shelf_body')
    
    # build shelf such that it is just a bin with inner walls
    add_bin_geometry(link, freespace_width, freespace_height, freespace_depth, thickness=thickness)
    '''
            x
            ^
            |
    y <-----|
    
                width (front wall)
    ------------------------------- 
    | |-------------------------| |
    | |                         | |
    | |                         | | height
    | |                         | |    
    ------------------------------- (column)
    -------------------------------
    | |                         | |
    | |                         | |
    | |                         | |
    | |-------------------------| |
    -------------------------------
    
    we split along the height dimension (freespace) into even pieces.
    the characteristics are the same as the bin in v0.
    '''
    
    # add inner walls
    # x-location of inner walls
    inner_wall_x = torch.linspace(-freespace_height/2, freespace_height/2, 2+inner_rows)
    inner_wall_x = inner_wall_x[1:-1]  # remove the outer points
    for x in inner_wall_x:
        box_params = (thickness, freespace_width, freespace_depth)
        box_origin = (x, 0.0, 0.0) # x, y, z
        add_box_geometry(link, box_params, box_origin, rgba=(0.6, 0.6, 0.6, 1.0))
    
    # write
    tree = ET.ElementTree(root)
    tree.write(filename, encoding='utf-8', xml_declaration=True)

class PresetBookShelfSetup(ObjectSetupABC):
    def __init__(self, 
                 initial_bin_pose, 
                 gym, 
                 book_urdf_path, 
                 freespace_width, 
                 freespace_height, 
                 freespace_depth,
                 goal_z_modif = 0.0,
                 shelf_columns=1, 
                 padding_y=0.01, 
                 thickness=0.01, 
                 squish_padding = 0.001, 
                 squish_randomize_range=(0.0,0.0), 
                 squish_tilt=True, 
                 random_tilt_prob=0.5,
                 use_random_tilt=False):
        super().__init__(gym, freespace_width, freespace_height, freespace_depth)
        self.padding_y = padding_y
        self.initial_bin_pose = initial_bin_pose
        self.goal_z_modif = goal_z_modif
        
        # get book dimension from book.urdf
        self.urdf = book_urdf_path
        urdf_tree = ET.parse(book_urdf_path)
        urdf_root = urdf_tree.getroot()
        urdf_robot = urdf_root[0]
        for child in urdf_robot:
            if child.tag == 'collision':
                for subchild in child:
                    if subchild.tag == 'geometry':
                        for subsubchild in subchild:
                            if subsubchild.tag == 'box':
                                dims = subsubchild.attrib.get('size', '0.05 0.05 0.05').split()
                                dims = [float(d) for d in dims]
                                break
        self.book_dims = dims # (span in x, y, z)

        self.books_per_row = int((freespace_width - self.padding_y) / (self.book_dims[1] + padding_y))
        self.books_per_column = (shelf_columns + 1)
        self.shelf_thickness = thickness  # thickness of the shelf walls
        
        self.sizes = torch.tensor([(self.book_dims[0], self.book_dims[1], self.book_dims[2]) for _ in range(len(self))])  # sizes of the books
        self.sizes_arranged = torch.tensor([(self.book_dims[0], self.book_dims[1], self.book_dims[2]) for _ in range(len(self)+1)]).reshape(self.books_per_row, self.books_per_column, 3)  # 
        self.geometric_centers = torch.zeros(len(self), 3, dtype=torch.float32)  # geometric centers of the books
        
        
        self.squish_padding = squish_padding  # padding to squish the cubes together, if needed
        self.squish_randomize_range = squish_randomize_range  # range to randomize the squish padding
        self.squish_tilt = squish_tilt  # True -> we tilt the book to occlude goal. False -> we push books closer
        self.random_tilt_prob = random_tilt_prob  # probability to apply random tilt when squish_tilt is True
        self.use_random_tilt = use_random_tilt
    
    def get_surface_points(self, mesh_points=1000):
        o3d_cube = o3d.geometry.TriangleMesh.create_box(width=self.book_dims[0], height=self.book_dims[1], depth=self.book_dims[2])
        min_init_sample_points = 200
        sample_num_points = max(min_init_sample_points, 2 * mesh_points) # increase sampling points to ensure enough points are sampled
        surface_points = np.asarray(o3d_cube.sample_points_uniformly(number_of_points=sample_num_points).points) # perform uniform sampling on the cube surface
        surface_points = np.random.permutation(surface_points)[:mesh_points] # randomly sample mesh_points points from the surface points
        surface_points = torch.tensor(surface_points, dtype=torch.float32)  # convert to torch tensor
        surface_points = surface_points - torch.tensor(self.book_dims) / 2.0
        return surface_points
    
    def __len__(self):
        return self.books_per_row * self.books_per_column - 1
    
    def setup_assets(self, sim):
        asset_options = gymapi.AssetOptions()
        asset_options.density = 10.0 # before it was 1000.0
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
        asset_options.use_mesh_materials = False
        
        asset = self.gym.load_asset(sim, os.path.dirname(self.urdf), os.path.basename(self.urdf), asset_options)
        self.assets['book'] = asset
        
        asset_options_fixed = deepcopy(asset_options)
        asset_options_fixed.fix_base_link = True
        asset_fixed = self.gym.load_asset(sim, os.path.dirname(self.urdf), "book_fixed.urdf", asset_options_fixed)
        self.assets['book_fixed'] = asset_fixed
    
    def create_actors(self, env_ptr0, env_ptr, actor_count, collision_group_id, friction=0.5):
        ## create cube actors with properties
        bin_pos = torch.tensor([self.initial_bin_pose.p.x, self.initial_bin_pose.p.y, self.initial_bin_pose.p.z], dtype=torch.float32)
        bin_quat = torch.tensor([self.initial_bin_pose.r.x, self.initial_bin_pose.r.y, self.initial_bin_pose.r.z, self.initial_bin_pose.r.w], dtype=torch.float32)
        poses, _, _ = self.initialize_poses(bin_pos=bin_pos, bin_quat=bin_quat)
        
        '''
            first two books are unfixed (to be manipulated)
            the rest are fixed (static in the shelf)
        '''
        for idx in range(len(self)):
            pose = poses[idx]
            handle = self.gym.create_actor(env_ptr, self.assets['book' if idx < 2 else 'book_fixed'], pose, f'obj_{idx}', collision_group_id, 0, 0)
            self.actor_handles[f'obj_{idx}'] = handle
            self.actor_ids_sim[f'obj_{idx}'].append(actor_count)

            ## add properties to cube actor
            shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
            shape_props[0].friction = friction
            shape_props[0].rolling_friction = 0.0  # default = 0.0
            shape_props[0].torsion_friction = 0.0  # default = 0.0
            shape_props[0].restitution = 0.0  # default = 0.0
            shape_props[0].compliance = 0.0  # default = 0.0
            shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, handle, shape_props)
            actor_count += 1

            # set mass
            book_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, handle)
            book_rb_props[0].mass = 1.0
            self.gym.set_actor_rigid_body_properties(env_ptr, handle, book_rb_props, recomputeInertia=True)
            
        if not self.created_once:
            ## get body_id_env_list and actor_id_env_list which is useful root pos/quat
            for idx in range(len(self)):
                actor_id_env = self.gym.find_actor_index(env_ptr, f'obj_{idx}', gymapi.DOMAIN_ENV)
                self.actor_id_env_list.append(actor_id_env)
                
                rb_names = self.gym.get_actor_rigid_body_names(env_ptr0, actor_id_env)
                body_id_env = self.gym.find_actor_rigid_body_index(env_ptr0, actor_id_env,
                                                                        rb_names[0], gymapi.DOMAIN_ENV)
                self.body_id_env_list.append(body_id_env)
        self.created_once = True
        return actor_count
    
    def reset_colors(self, env_ptr):
        pass
    
    def initialize_poses(self, randomize=False, squish_goal=False, bin_pos=torch.zeros(3), bin_quat=torch.tensor([0.0,0.0,0.0,1.0]), return_desired_packed_poses=False):
        (min_x, min_y, _), (max_x, max_y, _) = self.get_freespace_parameters()
        
        # /| <- po -> | cube 0 | <- p -> | cube 1| <- p -> | cube 2 | <- po -> |\
        p_y = self.padding_y
        po_y = ((max_y - min_y) - ((self.book_dims[1] * self.books_per_row) + (p_y * (self.books_per_row - 1))))/2.0
        
        inner_wall_x = torch.linspace(-self.freespace_height/2, self.freespace_height/2, 2+self.books_per_column-1)[:-1]
        pos_ys = torch.linspace(min_y + po_y + self.book_dims[1]/2, max_y - self.book_dims[1]/2 - po_y, self.books_per_row)
        pos_xs = inner_wall_x + self.book_dims[0] / 2
        pos_xs[1:] += + self.shelf_thickness / 2.0
        
        book_positions = torch.zeros((self.books_per_row, self.books_per_column, 3), dtype=torch.float32)
        book_positions[:,:,0] = pos_xs.unsqueeze(0).repeat(self.books_per_row, 1)
        book_positions[:,:,1] = pos_ys.unsqueeze(1).repeat(1, self.books_per_column)
        book_quats = torch.zeros((self.books_per_row, self.books_per_column, 4), dtype=torch.float32)
        book_quats[..., 3] = 1.0  # w component of quaternion
        
        choose_goal_location = torch.randint((len(self)-1) // 2 - 2, (len(self)-1) // 2 + 2 + 1,(1,)).item() if randomize else 7
        # modify goal based on booktaskshelving config
        book_positions[choose_goal_location % self.books_per_row, choose_goal_location // self.books_per_row, 2] = self.goal_z_modif

        book_positions[(choose_goal_location+1) % self.books_per_row, (choose_goal_location+1) // self.books_per_row, 1] += self.padding_y
        book_positions[(choose_goal_location-1) % self.books_per_row, (choose_goal_location-1) // self.books_per_row, 1] -= self.padding_y

        if return_desired_packed_poses:
            world_book_pos = torch_utils.tf_apply(bin_quat, bin_pos, book_positions) # new poses
            # world_book_pos[:, :, 2] = self.initial_bin_pose.p.z - self.freespace_depth/2.0 + self.sizes_arranged[:, :, 0] / 2.0
            world_book_quat = bin_quat
            desired_packed_poses = [torch.cat([world_book_pos[i % self.books_per_row, i // self.books_per_row], world_book_quat]) for i in range(len(self)+1)]
            
            desired_packed_poses[0], desired_packed_poses[choose_goal_location-1] = desired_packed_poses[choose_goal_location-1], desired_packed_poses[0]
            desired_packed_poses[1], desired_packed_poses[choose_goal_location+1] = desired_packed_poses[choose_goal_location+1], desired_packed_poses[1]
            desired_packed_poses.pop(choose_goal_location)
            desired_packed_poses = torch.stack(desired_packed_poses, dim=0)  # shape (num_cubes, 7)
        
        keypoint_dir = 0
        if squish_goal:
            guard = np.random.rand() < self.random_tilt_prob if self.use_random_tilt else self.squish_tilt
            for i in range(len(self)+1):
                if guard:
                    keypoint_dir = 1
                    if (i == choose_goal_location+1) and i // self.books_per_row == choose_goal_location // self.books_per_row:
                        rotz = -(torch.pi/2 - torch.arccos( torch.tensor( (4 * self.padding_y + self.book_dims[1]) / self.book_dims[0])))
                        quat_rotz = torch_utils.quat_from_euler_xyz(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(rotz))
                        book_center_to_botright_corner = torch.tensor([self.book_dims[0]/2.0, self.book_dims[1]/2.0, 0.0])
                        rotated_book_center_to_botright_corner = torch_utils.quat_apply(quat_rotz, book_center_to_botright_corner)
                        # botright_corner was the same as before the rotation
                        botright_corner_to_bin = book_positions[i % self.books_per_row, i // self.books_per_row] - torch.tensor([self.book_dims[0]/2.0, self.book_dims[1]/2.0, 0.0])
                        book_center_to_bin = rotated_book_center_to_botright_corner + botright_corner_to_bin
                        book_positions[i % self.books_per_row, i // self.books_per_row] = book_center_to_bin
                        book_quats[i % self.books_per_row, i // self.books_per_row, :] = torch_utils.quat_mul(quat_rotz, book_quats[i % self.books_per_row, i // self.books_per_row, :])
                
                else:
                    keypoint_dir = 0
                    if (i == choose_goal_location-1 or i == choose_goal_location+1) and i // self.books_per_row == choose_goal_location // self.books_per_row:
                        random_squish = np.random.rand() * (self.squish_randomize_range[1] - self.squish_randomize_range[0]) + self.squish_randomize_range[0]

                        book_positions[i % self.books_per_row, i // self.books_per_row, 1] += (1 if i == choose_goal_location-1 else -1) * (self.book_dims[1]/2 + 2 * self.padding_y - (self.squish_padding  + random_squish)/2.0)
            
        book_positions = torch_utils.tf_apply(bin_quat, bin_pos, book_positions)
        book_quats = torch_utils.quat_mul(bin_quat.unsqueeze(0).unsqueeze(0).expand(self.books_per_row, self.books_per_column, 4), book_quats)
        
        poses = []
        for i in range(len(self)+1):
            book_quat = book_quats[i % self.books_per_row, i // self.books_per_row, :]
            if i == choose_goal_location:
                goal = torch.zeros(7)
                goal[:3] = book_positions[i % self.books_per_row, i // self.books_per_row, :3]
                # goal[2]  = book_positions[i % self.books_per_row, i // self.books_per_row, 2]
                goal[3:] = torch.tensor([book_quat[0].item(), book_quat[1].item(), book_quat[2].item(), book_quat[3].item()], dtype=torch.float32)
            else:
                pose = gymapi.Transform()
                pose.p.x = book_positions[i % self.books_per_row, i // self.books_per_row, 0]
                pose.p.y = book_positions[i % self.books_per_row, i // self.books_per_row, 1]
                pose.p.z = book_positions[i % self.books_per_row, i // self.books_per_row, 2]
                book_quat = gymapi.Quat(book_quat[0].item(), book_quat[1].item(), book_quat[2].item(), book_quat[3].item())
                pose.r = book_quat
            poses.append(pose)
        # re-arrange poses
        
        # re-order poses such that the first two are neighbors of the goal
        # poses_idx = [goal-1, goal+1, 2, 3, ..., 0, 1, ..., N-1, N]
        poses[0], poses[choose_goal_location-1] = poses[choose_goal_location-1], poses[0]
        poses[1], poses[choose_goal_location+1] = poses[choose_goal_location+1], poses[1]
        # pop out goal from poses
        poses.pop(choose_goal_location)
        
        if return_desired_packed_poses:
            return poses, goal, keypoint_dir, desired_packed_poses
        return poses, goal, keypoint_dir
    
class Shelf(Bin):
    def __init__(self, gym, initial_bin_pose, freespace_width=0.3, freespace_height=0.21, freespace_depth=0.024, thickness=0.01, shelf_columns=1):
        super().__init__(gym, initial_bin_pose, freespace_width, freespace_height, freespace_depth, thickness)
        self.shelf_columns = shelf_columns
        self.autogenerate_fn = autogenerate_shelf_urdf  # function to autogenerate the shelf urdf file

class Book(ObjectABC):
    def __init__(self, gym, book_name, book_urdf_path):
        super().__init__(gym)
        self.name = book_name
        
        # get book dimension from book.urdf
        self.urdf = book_urdf_path
        urdf_tree = ET.parse(book_urdf_path)
        urdf_root = urdf_tree.getroot()
        urdf_robot = urdf_root[0]
        for child in urdf_robot:
            if child.tag == 'collision':
                for subchild in child:
                    if subchild.tag == 'geometry':
                        for subsubchild in subchild:
                            if subsubchild.tag == 'box':
                                dims = subsubchild.attrib.get('size', '0.05 0.05 0.05').split()
                                dims = [float(d) for d in dims]
                                break
        dims = [0.23, 0.024, 0.156]
        self.params = dims # (span in x, y, z)
        self.size = self.params[2]
        self.urdf_path = book_urdf_path
        
        
    def get_surface_points(self, mesh_points=1000):
        o3d_cube = o3d.geometry.TriangleMesh.create_box(width=self.params[0], height=self.params[1], depth=self.params[2])
        min_init_sample_points = 200
        sample_num_points = max(min_init_sample_points, 2 * mesh_points)  # increase sampling points to ensure enough points are sampled
        surface_points = np.asarray(o3d_cube.sample_points_uniformly(number_of_points=sample_num_points).points)
        surface_points = np.random.permutation(surface_points)[:mesh_points]  # randomly sample mesh_points points from the surface points
        surface_points = torch.tensor(surface_points, dtype=torch.float32)  # convert to torch
        surface_points = surface_points - torch.tensor(self.params) / 2.0 # center the points around the origin
        return surface_points
    def initialize_pose(self):
        raise NotImplementedError
    def setup_asset(self, sim):
        asset_options = gymapi.AssetOptions()
        asset_options.density = 50.0
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
        asset_options.use_mesh_materials = False
        
        asset = self.gym.load_asset(sim, os.path.dirname(self.urdf), os.path.basename(self.urdf), asset_options)
        self.assets[self.name] = asset
    def create_actor(self, env_ptr0, env_ptr, actor_count, collision_group_id, friction=0.5):
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
        shape_props[0].friction = friction
        shape_props[0].rolling_friction = 0.0  # default = 0.0
        shape_props[0].torsion_friction = 0.0  # default = 0.0
        shape_props[0].restitution = 0.0  # default = 0.0
        shape_props[0].compliance = 0.0  # default = 0.0
        shape_props[0].thickness = 0.0  # default = 0.0
        self.gym.set_actor_rigid_shape_properties(env_ptr, handle, shape_props)


        

        # set mass
        # book_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, handle)
        # print(book_rb_props[0].mass)
        # book_rb_props[0].mass = 0.200 # set to 500g (like real-world)
        # self.gym.set_actor_rigid_body_properties(env_ptr, handle, book_rb_props, recomputeInertia=True)
        
        if self.actor_id_env is None:
            self.actor_id_env = self.gym.find_actor_index(env_ptr, self.name, gymapi.DOMAIN_ENV)
            rb_names = self.gym.get_actor_rigid_body_names(env_ptr0, self.actor_id_env)
            self.body_id_env = self.gym.find_actor_rigid_body_index(env_ptr0, self.actor_id_env,
                                                                        rb_names[0], gymapi.DOMAIN_ENV)
        
        actor_count += 1
        return actor_count

class BookShelvingEnv(BinEnvPacking):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        BinEnvPacking.__init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        if self.cfg_task.env.use_hydrosoft_model:
            self.local_indenter_pts = self.hydrofots_sensors[0].indenter_sdf_sensor.poisson_sample_points(radius=0.00075, initial_num_points=int(1e6))[0].cpu().numpy()
            self.local_indenter_pts = self.local_indenter_pts[self.local_indenter_pts[:,2] >= 0.0]

            self.num_indenter_pts = self.local_indenter_pts.shape[0]
        
    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        bin_pos = self.cfg_task.randomize.initial_bin_pos if "initial_bin_pos" in self.cfg_task.randomize else [0.5, 0.0, 0.0]
        bin_rot = self.cfg_task.randomize.initial_bin_rot if "initial_bin_rot" in self.cfg_task.randomize else [0.0, -torch.pi/2, 0.0]  # euler angles in radians
        bin_quat = torch_utils.quat_from_euler_xyz(torch.tensor(bin_rot[0]), torch.tensor(bin_rot[1]), torch.tensor(bin_rot[2]))
        self.bin_pose.p.x = bin_pos[0]
        self.bin_pose.p.y = bin_pos[1]
        self.bin_pose.p.z = bin_pos[2]
        self.bin_pose.r = gymapi.Quat(bin_quat[0], bin_quat[1], bin_quat[2], bin_quat[3])
        
        # self.packed_objects = PresetCubeSetup(cube_size=0.05, gym=self.gym, initial_bin_pose=self.bin_pose,
        #                                     freespace_width=self.freespace_width, freespace_height=self.freespace_height,
        #                                     freespace_depth=self.freespace_depth, padding_x=0.0015, padding_y=0.008)

        # self.insertion_obj = self.insertion_obj = Cube(gym=self.gym, cube_name='plug', cube_size=0.05)
        self.insertion_obj = Book(gym=self.gym, book_name='plug', book_urdf_path=os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'urdf', 'insert_book.urdf'))

        # goal_z_modif = (self.freespace_depth / 2.0) - (self.insertion_obj.params[2] / 2.0)
        goal_z_modif = (self.freespace_depth / 2.0)
        self.packed_objects = PresetBookShelfSetup(initial_bin_pose=self.bin_pose, gym=self.gym,
                                                   book_urdf_path=os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'urdf', 'book.urdf'),
                                                   freespace_width=self.freespace_width, freespace_height=self.freespace_height,
                                                   freespace_depth=self.freespace_depth, goal_z_modif=goal_z_modif, shelf_columns=0, padding_y=0.014, thickness=0.01,
                                                   squish_padding=self.cfg_task.randomize.preset_object_squish_padding,
                                                   squish_randomize_range=self.cfg_task.randomize.preset_object_squish_noise_range,
                                                   squish_tilt=self.cfg_task.randomize.get("preset_object_use_squish_tilt",False),
                                                   random_tilt_prob=self.cfg_task.randomize.get("preset_object_random_tilt_prob",0.0),
                                                   use_random_tilt=self.cfg_task.randomize.get("preset_object_use_random_tilt",False)
                                                   )
        self.bin = Shelf(gym=self.gym, initial_bin_pose=self.bin_pose, freespace_width=self.freespace_width,
                       freespace_height=self.freespace_height, freespace_depth=self.freespace_depth,
                       thickness=0.01)
        
        
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
        self.plug_file = 'insert_book.urdf'
        self._create_sensors()