import vedo
vedo.settings.enable_default_keyboard_callbacks = False
vedo.settings.enable_pipeline = False
import trimesh
import numpy as np
import argparse
from abc import ABC
from scipy.spatial.transform import Rotation as R_scipy
from rl.demo_utils.hydrosoft import HydroSoftSensor
from rl.demo_utils.utils.mesh_utils2 import MeshSDFSensor
import torch
import torch.nn.functional as F
from rl.demo_utils.utils.viz_utils import visualize_tactile_shear_image
import cv2
import time
import os

class VedoObject(ABC):
    def assemble_mesh(self, mesh_list: [vedo.Mesh]) -> vedo.Assembly:
        self.mesh = vedo.Assembly(mesh_list)
        return self.mesh

    def shift(self, translation: np.ndarray):
        self.mesh.AddPosition(translation[0], translation[1], translation[2])

    def rotate(self, axis: np.ndarray, angle_deg: float):
        self.mesh.RotateWXYZ(angle_deg, axis[0], axis[1], axis[2])

    def set_mesh_pose_matrix(self, target_pose):
        current_pose = self.get_mesh_pose_matrix()
        # T @ current = target => T = target @ inv(current)
        transform = target_pose @ np.linalg.inv(current_pose)
        self.mesh.apply_transform(transform)

    def get_vedo_mesh(self):
        return self.mesh

    def get_mesh_pose(self):
        pos = self.mesh.GetPosition()
        ori = self.mesh.GetOrientation()
        return pos, ori

    def get_mesh_pose_matrix(self):
        mat = self.mesh.GetMatrix()
        pose = np.eye(4)
        for i in range(4):
            for j in range(4):
                pose[i, j] = mat.GetElement(i, j)
        return pose

class GSMiniFingerVedo(VedoObject):
    def __init__(self):
        mesh_obj_path = "demo_assets/gsmini_finger_attachment.obj"
        mesh_mtl_path = "demo_assets/gsmini_finger_attachment.mtl"
        self.finger_mesh : [vedo.Mesh] = vedo.file_io.load_obj(mesh_obj_path, mesh_mtl_path)
        mesh_obj_path = "demo_assets/gsmini_fingerbox.obj"
        mesh_mtl_path = "demo_assets/gsmini_fingerbox.mtl"
        self.fingerbox_mesh : [vedo.Mesh] = vedo.file_io.load_obj(mesh_obj_path, mesh_mtl_path)
        mesh_obj_path = 'demo_assets/gsmini_elastomer.obj'
        mesh_mtl_path = 'demo_assets/gsmini_elastomer.mtl'
        self.elastomer_mesh : [vedo.Mesh] = vedo.file_io.load_obj(mesh_obj_path, mesh_mtl_path)

        # color mesh to be half as bright
        self.finger_mesh = [mesh.c(mesh.color() * 0.5) for mesh in self.finger_mesh]
        self.fingerbox_mesh = [mesh.c(mesh.color() * 0.3) for mesh in self.fingerbox_mesh]
        self.elastomer_mesh = [mesh.c(mesh.color() * 0.5) for mesh in self.elastomer_mesh]

        # make elastomer alpha low
        self.elastomer_mesh = [mesh.alpha(0.3) for mesh in self.elastomer_mesh]

        self.mesh = self.assemble_mesh(self.finger_mesh + self.fingerbox_mesh + self.elastomer_mesh)
        self.mesh.rotate_x(180)
        self.mesh.pos(0.0, -0.06, -0.01)

        # get elastomer mesh bbox position
        elastomer_bbox = self.elastomer_mesh[0].bounds()
        self.elastomer_center = np.array([
            (elastomer_bbox[0] + elastomer_bbox[1]) / 2,
            (elastomer_bbox[2] + elastomer_bbox[3]) / 2,
            (elastomer_bbox[4] + elastomer_bbox[5]) / 2,
        ])
        # set up meshsdfsensor (note this is in torch)
        self.sdf_sensor = MeshSDFSensor(
            mesh_path='demo_assets/gsmini_elastomer.obj',
            device='cpu'
        )
        self.num_divs = [10, 15]
        self.elastomer_tactile_pts = self.sdf_sensor.generate_tactile_points(
            num_divs=self.num_divs,
            margin=0.002,
            local_z_dir=-1
        )
        
class GSMiniElastomerVedo(VedoObject):
    def __init__(self):
        mesh_obj_path = 'demo_assets/gsmini_elastomer.obj'
        mesh_mtl_path = 'demo_assets/gsmini_elastomer.mtl'
        self.elastomer_mesh : [vedo.Mesh] = vedo.file_io.load_obj(mesh_obj_path, mesh_mtl_path)

        self.elastomer_mesh = [mesh.c(mesh.color() * 0.5) for mesh in self.elastomer_mesh]
        self.elastomer_mesh = [mesh.alpha(0.3) for mesh in self.elastomer_mesh]

        self.mesh = self.assemble_mesh(self.elastomer_mesh)
        self.mesh.rotate_x(180)
        self.mesh.pos(0.0, -0.06, -0.01)

        # get elastomer mesh bbox position
        elastomer_bbox = self.elastomer_mesh[0].bounds()
        self.elastomer_center = np.array([
            (elastomer_bbox[0] + elastomer_bbox[1]) / 2,
            (elastomer_bbox[2] + elastomer_bbox[3]) / 2,
            (elastomer_bbox[4] + elastomer_bbox[5]) / 2,
        ])

        self.sdf_sensor = MeshSDFSensor(
            mesh_path='demo_assets/gsmini_elastomer.obj',
            device='cpu'
        )
        self.num_divs = [7, 9]
        self.elastomer_tactile_pts = self.sdf_sensor.generate_tactile_points(
            num_divs=self.num_divs,
            margin=0.002,
            local_z_dir=-1
        )
        
class FrameVedo(VedoObject):

    def __init__(self, size=0.1, pose=np.eye(4)):
        self.size = size
        arrow_ends = np.eye(3) # in frame reference
        colors = ['r', 'g', 'b']
        # get the arrows in world frame
        w_X_f = pose
        arrow_ends_w = w_X_f[:3, :3] @ arrow_ends.T + w_X_f[:3, 3:4] # (3, 3)
        start_p = w_X_f[:3, 3]
        arrows_geoms = []
        for i, color_i in enumerate(colors):
            arrow_i = vedo.Arrow(start_pt=start_p, end_pt=arrow_ends_w[i], c=color_i)
            arrow_i = arrow_i.scale(self.size)
            arrows_geoms.append(arrow_i)
        self.mesh = self.assemble_mesh(arrows_geoms)

    def update_pose(self, pose):
        # Update frame pose
        self.set_mesh_pose_matrix(pose)

class ObjectVedo(VedoObject):

    def __init__(self, mesh_path):
        self.obj_mesh = trimesh.load(mesh_path)
        self.obj_mesh = vedo.Mesh(mesh_path)

        self.obj_mesh = self.obj_mesh.c(self.obj_mesh.color() * 0.8)
        self.obj_mesh = self.obj_mesh.alpha(0.3)

        # Calculate radius from local mesh bounds before assembly transformation
        bounds = self.obj_mesh.bounds()
        self.radius = (bounds[1] - bounds[0]) / 2.0

        self.mesh = self.assemble_mesh([self.obj_mesh])

        # set up meshsdfsensor (note this is in torch)
        self.sdf_sensor = MeshSDFSensor(
            mesh_path=mesh_path,
            device='cpu'
        )

        self.num_obj_pts = 100
        self.obj_pts, self.obj_normals = self.sdf_sensor.sample_points(self.num_obj_pts)

class ForceArrowsVedo(VedoObject):

    def __init__(self):
        self.arrows = []
        self.mesh = self.assemble_mesh([])

    def update(self, forces, positions, world_pose):
        """
        forces: (N, 3) force vectors
        positions: (N, 3) positions of forces
        world_pose: (4, 4) pose of the object to transform arrows to world frame
        """
        arrows = []
        
        # Transform positions and forces to world frame
        R = world_pose[:3, :3]
        t = world_pose[:3, 3]
        
        # Forces are vectors, so only rotate them
        if len(forces) == 0:
            forces_world = np.array([])
            positions_world = np.array([])
        else:
            forces_world = np.einsum('ij,kj->ki', R, forces)
            # Positions are points, so rotate and translate
            positions_world = np.einsum('ij,kj->ki', R, positions) + t
        
        for i in range(len(forces)):
            force = forces_world[i]
            if np.linalg.norm(force) < 1e-7:
                continue
            
            start_pt = positions_world[i]
            end_pt = start_pt + force
            
            # Use vedo Arrow which handles orientation automatically
            # Scale factor to make arrows visible similar to open3d implementation
            # Scale length by magnitude
            length = np.linalg.norm(force)
            # Apply scaling factor to match open3d visualization
            # open3d code: cylinder_height=(np.linalg.norm(force) + 1e-8)
            # but usually we need some visual scaling. 
            # In hydrosoft.py it seems to use raw force magnitude for height.
            # But earlier user said they look super large.
            # Let's reduce the scale.
            
            arrow = vedo.Arrow(
                start_pt=start_pt + force * 1.0, # Scale down length,
                end_pt=start_pt,
                s=0.00001, # shaft radius (thinner)
            ).c("red")
            arrows.append(arrow)

            # Add start and end points
            # Start of arrow (at the surface/contact) is start_pt
            # End of arrow (tip) is start_pt + force (calculated above as end_pt for loop logic but vedo.Arrow args are specific)
            arrows.append(vedo.Point(start_pt, r=10, c='red')) # Contact point (Tip)
            arrows.append(vedo.Point(start_pt + force, r=10, c='red')) # Tail
            
        self.mesh = self.assemble_mesh(arrows)
    
    def get_vedo_mesh(self):
        return self.mesh

class ShearArrowsVedo(VedoObject):
    def __init__(self):
        self.arrows = []
        self.mesh = self.assemble_mesh([])

    def update(self, shear_vectors, positions, world_pose):
        """
        shear_vectors: (N, 3) shear vectors in local frame
        positions: (N, 3) positions of tactile points in local frame
        world_pose: (4, 4) pose of the elastomer to transform arrows to world frame
        """
        if len(shear_vectors) == 0:
            self.mesh = self.assemble_mesh([])
            return

        arrows = []
        
        # Project shear vectors onto the local XY plane (remove Z component)
        # This makes the arrows "flat" on the elastomer plane
        shear_vectors_planar = shear_vectors.copy()
        shear_vectors_planar[:, 2] = 0 
        
        # Transform positions and vectors to world frame
        R = world_pose[:3, :3]
        t = world_pose[:3, 3]
        
        # Shear vectors are vectors, so only rotate them
        if len(shear_vectors_planar) == 0:
            shear_world = np.array([])
            positions_world = np.array([])
        else:
            # Important: Rotate planar vectors to world frame using only Rotation matrix
            # shear_vectors_planar are vectors, not points
            shear_world = np.einsum('ij,kj->ki', R, shear_vectors_planar)
            
            # Positions are points, so rotate and translate
            positions_world = np.einsum('ij,kj->ki', R, positions) + t
        
        for i in range(len(shear_vectors_planar)):
            shear = shear_world[i]
            # Check magnitude of planar vector
            if np.linalg.norm(shear) < 1e-6:
                continue
            
            start_pt = positions_world[i]
            # Arbitrary scale as requested, user will tune
            # Increase scale if not visible
            end_pt = start_pt + shear * 0.001
            
            arrow = vedo.Arrow(
                start_pt=start_pt,
                end_pt=end_pt,
                s=0.000015, # slightly thicker than force arrows?
            ).c("green")
            arrows.append(arrow)
            
        self.mesh = self.assemble_mesh(arrows)
    
    def get_vedo_mesh(self):
        return self.mesh

class ContactCircleVedo(VedoObject):
    def __init__(self):
        self.mesh = self.assemble_mesh([])

    def update(self, radius, center, axis=np.array([0, 0, 1])):
        if radius <= 0:
            self.mesh = self.assemble_mesh([])
            return
            
        # Create a circle (ring)
        circle = vedo.Circle(pos=center, r=radius, c='red', res=50).wireframe().lw(3)
        
        # Orient the circle
        current_normal = np.array([0, 0, 1])
        if np.linalg.norm(axis) > 1e-6:
            axis = axis / np.linalg.norm(axis)
            v = np.cross(current_normal, axis)
            s = np.linalg.norm(v)
            c = np.dot(current_normal, axis)
            
            if s > 1e-6:
                angle_deg = np.degrees(np.arccos(c))
                circle.rotate(angle_deg, axis=v, point=center)
            elif c < 0:
                circle.rotate(180, axis=[1,0,0], point=center)

        self.mesh = self.assemble_mesh([circle])

    def get_vedo_mesh(self):
        return self.mesh

if __name__ == '__main__':
    VISUALIZE_TACTILE_PCD = [True]
    VISUALIZE_CONTACT_CIRCLE = [True]
    VISUALIZE_INDENTER_PCD = [False]
    VISUALIZE_SLIDER = False
    VISUALIZE_PROJECTION_IMAGE = False
    VISUALIZE_ELASTOMER_CENTER_SPHERE = False
    VISUALIZE_INDENTER_FRAME = [False]
    VISUALIZE_FORCE_ON_VEDO = [False]
    VISUALIZE_SHEAR_ON_VEDO = [True]

    TACTILE_PCD_SIZE = 10
    INDENTER_PCD_SIZE = 10

    light1 = vedo.Light(
        pos=(0, 0, 1),
        focal_point=(0, 0, 0),
        c="white",
        intensity=0.5
    )
    

    shift_amount = np.array([-5.0e-4, -3.0e-4, 0.007])

    # gsminifinger_vedo = GSMiniFingerVedo()
    gsminifinger_vedo = GSMiniElastomerVedo()
    object_vedo = ObjectVedo("demo_assets/sphere.obj")
    sphere_radius = object_vedo.radius
    num_pts = object_vedo.num_obj_pts
    points = []
    offset = 2.0 / num_pts
    increment = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(num_pts):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y * y)
        phi = ((i + 1) % num_pts) * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append([x * sphere_radius, y * sphere_radius, z * sphere_radius])
    object_vedo.obj_pts = torch.tensor(points)
    
    # Initialize frame at object pose
    object_pose = object_vedo.get_mesh_pose_matrix()
    frame_vedo = FrameVedo(size=0.0025, pose=object_pose)
    object_vedo.shift(shift_amount)
    frame_vedo.shift(shift_amount)
    object_pose = object_vedo.get_mesh_pose_matrix()
    initial_object_pose = object_pose.copy()
    force_arrows_vedo = ForceArrowsVedo()
    shear_arrows_vedo = ShearArrowsVedo()
    contact_circle_vedo = ContactCircleVedo()

    hydrosoft = HydroSoftSensor(
        mu=3,
        lambda_d=50_000,
        lambda_s=100_000,
        dilate_scale=38461.5385 * 5,
        shear_scale=790000 / 40 * 10,
        normal_axis=2
    )
    hydrosoft.initialize(1, object_vedo.num_obj_pts)

    viz = vedo.Plotter(shape=(1,1))
    viz.at(0)
    '''
    Camera pos:
    (0.2026717078204835, -0.02588413672850003, 0.01689244996347192)
    Camera focal point:
    (-1.7236867889953002e-05, -0.028585628562743243, 0.016816503663089)
    Camera view up:
    (-0.08440718641137869, 0.0021052786059308952, 0.9964291217563377)
    '''
    cam = viz.camera
    cam.SetPosition(0.023415281951362058, 0.033288135031922586, 0.06029433054766119)
    cam.SetFocalPoint(-0.0004999996162950941, -0.0002499999850988366, 0.004617669895691867)
    cam.SetViewUp(-0.36158070842044415, -0.721818140926492, 0.5901169059835455)
    if VISUALIZE_INDENTER_FRAME[0]:
        viz += frame_vedo.get_vedo_mesh()
    viz += gsminifinger_vedo.get_vedo_mesh()
    viz += object_vedo.get_vedo_mesh()
    viz += force_arrows_vedo.get_vedo_mesh()
    viz += shear_arrows_vedo.get_vedo_mesh()
    if VISUALIZE_CONTACT_CIRCLE[0]:
        viz += contact_circle_vedo.get_vedo_mesh()

    elastomer_pose = gsminifinger_vedo.get_mesh_pose_matrix()
    elastomer_tactile_points = gsminifinger_vedo.elastomer_tactile_pts
    elastomer_tactile_point_in_world = np.einsum('ij,kj->ki', elastomer_pose[:3, :3], elastomer_tactile_points) + elastomer_pose[:3, 3]  # (num_pts, 3)


    indenter_pose = object_vedo.get_mesh_pose_matrix()
    indenter_pts = object_vedo.obj_pts.cpu().numpy()
    indenter_pts_in_world = np.einsum('ij,kj->ki', indenter_pose[:3, :3], indenter_pts) + indenter_pose[:3, 3]  # (num_obj_pts, 3)

    if VISUALIZE_ELASTOMER_CENTER_SPHERE:
        # visualize elastomer center as big sphere
        elastomer_center_sphere = vedo.Sphere(r=0.002, c='magenta').pos(gsminifinger_vedo.elastomer_center)
        viz += elastomer_center_sphere

    # vedo point cloud
    # Initialize with empty points to allow updating
    elastomer_tactile_pcd = vedo.Points(r=TACTILE_PCD_SIZE, c='yellow')
    indenter_pcd = vedo.Points(r=INDENTER_PCD_SIZE, c='red')
    
    if VISUALIZE_TACTILE_PCD[0]:
        elastomer_tactile_pcd = vedo.Points(elastomer_tactile_point_in_world, r=TACTILE_PCD_SIZE, c='yellow')
        viz += elastomer_tactile_pcd

    if VISUALIZE_INDENTER_PCD[0]:
        indenter_pcd = vedo.Points(indenter_pts_in_world, r=INDENTER_PCD_SIZE, c='red')
        viz += indenter_pcd

    step_size_container = [0.0005]
    camera_step_size_container = [1.0]
    def slider_step_size(widget, event):
        step_size_container[0] = widget.value
    def camera_slider_step_size(widget, event):
        camera_step_size_container[0] = widget.value
    # Slider variable
    slider_widget = None
    slider_camera_widget = None
    if VISUALIZE_SLIDER:
        slider_widget = viz.add_slider(
            slider_step_size,
            xmin=0.0001,
            xmax=0.01,
            value=0.0001,
            pos=[(0.35, 0.05), (0.65, 0.05)],
            title="Step Size",
            tformat="%.4f"
        )
        slider_camera_widget = viz.add_slider(
            camera_slider_step_size,
            xmin=0.0,
            xmax=5.0,
            value=1.0,
            pos=[(0.35, 0.1), (0.65, 0.1)],
            title="Camera Step Size",
            tformat="%.2f"
        )


    # Recording state
    recording_container = [False]
    recorded_poses = []
    
    # Auto-rotation state
    auto_rotate_container = [False]

    # Camera view state
    camera_mode_container = [0] # 0: Normal, 1: Top-Down
    saved_camera_pose = {'pos': None, 'focal': None, 'up': None}
    
    # Container to share shear image with callback
    shear_image_container = [None]

    def update_simulation(save_frames=False, frame_idx=0, save_folder="frames"):
        # Update frame pose to match object pose
        frame_vedo.update_pose(object_vedo.get_mesh_pose_matrix())
        
        # Handle Frame Visualization
        if VISUALIZE_INDENTER_FRAME[0]:
            if frame_vedo.mesh not in viz.actors:
                viz.add(frame_vedo.mesh)
        else:
            if frame_vedo.mesh in viz.actors:
                viz.remove(frame_vedo.mesh)

        # Auto-rotate camera if enabled
        if auto_rotate_container[0]:
            # Rotate camera around the frame origin
            # Get current camera position and focal point
            cam_pos = np.array(viz.camera.GetPosition())
            focal_point = np.array(frame_vedo.get_mesh_pose()[0]) # Use frame position as focal point
            
            # Calculate vector from focal point to camera
            vec = cam_pos - focal_point
            
            # Rotate vector around Z axis (assuming Z-up world)
            angle_rad = np.radians(1.0) # 1 degree
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            
            new_vec = R @ vec
            new_cam_pos = focal_point + new_vec
            
            viz.camera.SetPosition(new_cam_pos)
            viz.camera.SetFocalPoint(focal_point)
            
        # update hydrosoft
        indenter_pose = object_vedo.get_mesh_pose_matrix()
        inv_indenter_pose = np.linalg.inv(indenter_pose)
        elastomer_pose = gsminifinger_vedo.get_mesh_pose_matrix()
        inv_elastomer_pose = np.linalg.inv(elastomer_pose)

        elastomer_tactile_points = gsminifinger_vedo.elastomer_tactile_pts
        elastomer_tactile_point_in_world = np.einsum('ij,kj->ki', elastomer_pose[:3, :3], elastomer_tactile_points) + elastomer_pose[:3, 3]  # (num_pts, 3)

        elastomer_tactile_points_in_indenter = np.einsum('ij,kj->ki', inv_indenter_pose[:3, :3], elastomer_tactile_point_in_world) + inv_indenter_pose[:3, 3]  # (num_pts, 3)
        sdf, _ = object_vedo.sdf_sensor.get_sdf(torch.tensor(elastomer_tactile_points_in_indenter, dtype=torch.float32).unsqueeze(0))
        height = F.relu(-sdf)  # (1, num_pts)

        Mdilate = hydrosoft.get_dilation_displacement(
            torch.tensor(elastomer_tactile_points, dtype=torch.float32).unsqueeze(0),
            height
        )
        
        indenter_pts = object_vedo.obj_pts.cpu().numpy()
        indenter_pts_in_world = np.einsum('ij,kj->ki', indenter_pose[:3, :3], indenter_pts) + indenter_pose[:3, 3]  # (num_obj_pts, 3)

        indenter_pts_in_elastomer = np.einsum('ij,kj->ki', inv_elastomer_pose[:3, :3], indenter_pts_in_world) + inv_elastomer_pose[:3, 3]  # (num_obj_pts, 3)

        sdf, _ = gsminifinger_vedo.sdf_sensor.get_sdf(torch.tensor(indenter_pts_in_elastomer, dtype=torch.float32).unsqueeze(0))
        Mshear, _ = hydrosoft.get_hydrosoft_displacement(
            torch.tensor(gsminifinger_vedo.elastomer_tactile_pts, dtype=torch.float32).unsqueeze(0),
            torch.tensor(indenter_pts_in_elastomer, dtype=torch.float32).unsqueeze(0),
            sdf,
            reverse_z=True
        )
        

        if VISUALIZE_SHEAR_ON_VEDO[0]:
            total_displacement = Mdilate + Mshear
            total_displacement_local = total_displacement.squeeze(0).cpu().numpy()
            
            # tactile points are in local elastomer frame
            tactile_points_local = gsminifinger_vedo.elastomer_tactile_pts
            
            viz.remove(shear_arrows_vedo.mesh)
            shear_arrows_vedo.update(total_displacement_local, tactile_points_local, elastomer_pose)
            viz.add(shear_arrows_vedo.mesh)
        else:
            viz.remove(shear_arrows_vedo.mesh)
            shear_arrows_vedo.update(np.array([]), np.array([]), np.eye(4))
            
        # Update Contact Circle
        if contact_circle_vedo.mesh in viz.actors:
            viz.remove(contact_circle_vedo.mesh)
            
        if VISUALIZE_CONTACT_CIRCLE[0]:
            # Calculate intersection
            # 1. Get Sphere info
            # Radius: Use pre-calculated radius from ObjectVedo to avoid AABB rotation issues
            sphere_radius = object_vedo.radius
            
            # Center: Use geometric center (bounding box center) instead of pose origin
            # because the mesh might be offset from the assembly origin/pivot
            # Assembly doesn't have center_of_mass, use GetCenter (bbox center)
            sphere_center = np.array(object_vedo.mesh.GetCenter())
            
            # 2. Get Elastomer Surface info
            # Use tactile points mean Z in world frame as approximation of surface plane
            # Or use elastomer pose origin if tactile points are at z=0 local.
            # tactile_points_local Z is roughly 0 (or determined by generation).
            # Let's check mean Z of tactile points in world.
            
            # elastomer_tactile_point_in_world is available
            # surface_center = np.mean(elastomer_tactile_point_in_world, axis=0) # This might be biased by distribution
            # Better: Transform local (0,0,0) to world? 
            # Or check local points Z.
            # local_pts = gsminifinger_vedo.elastomer_tactile_pts
            # mean_local_z = np.mean(local_pts[:, 2])
            
            # Let's define the plane by the elastomer frame + offset
            # Plane normal: Z axis of elastomer_pose
            plane_normal = elastomer_pose[:3, 2]
            plane_point = elastomer_pose[:3, 3] 
            # Note: tactile points are offset from elastomer origin usually.
            # gsminifinger_vedo.elastomer_center is bounding box center.
            # Let's use the mean of tactile points as the "Surface Point"
            surface_point = np.mean(elastomer_tactile_point_in_world, axis=0)
            
            # 3. Calculate Distance
            # Project sphere_center onto plane normal relative to surface_point
            vec = sphere_center - surface_point
            dist = np.dot(vec, plane_normal)
            
            # 4. Check penetration
            # If sphere is "above" surface, dist > Radius -> No intersection
            # Note: "above" depends on normal direction. 
            # If normal points OUT of elastomer (towards indenter), positive dist means outside.
            # We assume normal points towards indenter (Z axis usually up/out).
            
            # Intersection radius
            # if dist < sphere_radius and dist > -sphere_radius:
            # We only care if it's penetrating (touching).
            # dist is distance from plane.
            
            if abs(dist) < sphere_radius:
                r_intersect = np.sqrt(sphere_radius**2 - dist**2)
                
                # Center of circle is projection of sphere center onto plane
                c_intersect = sphere_center - dist * plane_normal
                
                contact_circle_vedo.update(r_intersect, c_intersect, plane_normal)
            else:
                contact_circle_vedo.update(0, np.zeros(3))
            
            # Add updated mesh
            viz.add(contact_circle_vedo.mesh)
        
        # Visualize force arrows
        # Get forces from hydrosoft (computed in get_hydrosoft_displacement step)
        if hydrosoft.hydrosoft_forces is not None:
            # fbar is in elastomer frame (same as indenter_pts_in_elastomer)
            forces = hydrosoft.hydrosoft_forces.squeeze(0).cpu().numpy()
            
            if VISUALIZE_FORCE_ON_VEDO[0]:
                if np.max(np.linalg.norm(forces, axis=1)) < 1e-6:
                    # If forces are negligible (e.g. no contact), clear arrows
                    viz.remove(force_arrows_vedo.mesh)
                    force_arrows_vedo.update(np.array([]), np.array([]), np.eye(4))
                else:
                    # ... transformation logic ...
                    contact_pts_elastomer = indenter_pts_in_elastomer
                    
                    # Update the mesh and the visualizer
                    viz.remove(force_arrows_vedo.mesh)
                    force_arrows_vedo.update(forces, contact_pts_elastomer, elastomer_pose)
                    viz.add(force_arrows_vedo.mesh)
            else:
                viz.remove(force_arrows_vedo.mesh)
                force_arrows_vedo.update(np.array([]), np.array([]), np.eye(4))
        else:
            if VISUALIZE_FORCE_ON_VEDO[0]:
                # If no forces (e.g. no contact), clear arrows
                viz.remove(force_arrows_vedo.mesh)
                force_arrows_vedo.update(np.array([]), np.array([]), np.eye(4))

        total_shear = Mdilate + Mshear  # (1, num_pts, 3)
        total_shear = total_shear.squeeze().reshape(gsminifinger_vedo.num_divs[1], gsminifinger_vedo.num_divs[0], 3)

        total_shear = total_shear[..., :2].cpu().numpy()
        u = total_shear[..., 0][::-1,::-1]
        v = total_shear[..., 1][::-1,::-1]
        sim_shear = np.stack([-v, -u], axis=-1) # (H, W, 2)

        sim_shear_img = visualize_tactile_shear_image(sim_shear, shear_force_threshold=5.0, resolution=70) * 255.0
        sim_shear_img_uint8 = sim_shear_img.astype(np.uint8)
        shear_image_container[0] = sim_shear_img_uint8
        cv2.imshow("Simulated Shear Image", sim_shear_img_uint8)

        if VISUALIZE_PROJECTION_IMAGE:
            # Projection Image Visualization (from demo_viser.py)
            tactile_pts = gsminifinger_vedo.elastomer_tactile_pts
            min_x = np.min(tactile_pts[:, 0])
            max_x = np.max(tactile_pts[:, 0])
            min_y = np.min(tactile_pts[:, 1])
            max_y = np.max(tactile_pts[:, 1])
            
            im_size_x = 600
            im_size_y = 800
            
            scale_x = im_size_x / (max_x - min_x + 1e-8)
            scale_y = im_size_y / (max_y - min_y + 1e-8)
            
            projim = np.zeros((im_size_y, im_size_x, 3), dtype=np.uint8)
            
            # Draw tactile points (green circles) and Shear arrows (green lines)
            shear_vecs = Mshear.squeeze(0).cpu().numpy()
            
            for i in range(tactile_pts.shape[0]):
                x, y = tactile_pts[i, :2]
                u = int((x - min_x) * scale_x)
                v = int((y - min_y) * scale_y)
                
                # Clamp coordinates to be safe
                u = np.clip(u, 0, im_size_x - 1)
                v = np.clip(v, 0, im_size_y - 1)
                
                # Green circle for tactile point
                cv2.circle(projim, (u, im_size_y - v), 2, (0, 255, 0), -1)
                
                # Shear arrow
                sx, sy = shear_vecs[i, :2]
                shear_scale = 100.0
                
                if np.linalg.norm([sx, sy]) > 1e-6:
                    end_u = int(u + sx * shear_scale)
                    end_v = int(im_size_y - (v + sy * shear_scale))
                    # cv2 will clip lines automatically, but good to be aware
                    cv2.arrowedLine(projim, (end_u, end_v), (u, im_size_y - v),(0, 255, 0), 1, tipLength=0.3)

            # Draw projected points (red circles) and force arrows (red lines)
            if hydrosoft.hydrosoft_forces is not None:
                # Re-use forces calculated earlier
                # forces = hydrosoft.hydrosoft_forces.squeeze(0).cpu().numpy() # Already defined above
                
                force_mags = np.linalg.norm(forces, axis=1)
                mask = force_mags > 1e-6
                
                if np.any(mask):
                    active_pts = indenter_pts_in_elastomer[mask]
                    active_forces = forces[mask]
                    
                    projected_pts = active_pts + active_forces
                    # Use negative forces for arrows as in demo_viser.py
                    vis_forces = -active_forces
                    
                    for i in range(projected_pts.shape[0]):
                        x, y = projected_pts[i, :2]
                        u = int((x - min_x) * scale_x)
                        v = int((y - min_y) * scale_y)
                        
                        u = np.clip(u, 0, im_size_x - 1)
                        v = np.clip(v, 0, im_size_y - 1)
                        
                        # Red circle for projected point
                        cv2.circle(projim, (u, im_size_y - v), 5, (0, 0, 255), -1)
                        
                        # Force arrow
                        fx, fy = vis_forces[i, :2]
                        force_scale = 5000.0 * 10
                        
                        end_u = int(u + fx * force_scale)
                        end_v = int(im_size_y - (v + fy * force_scale))
                        
                        cv2.arrowedLine(projim, (end_u, end_v), (u, im_size_y - v), (0, 0, 255), 1, tipLength=0.3)

            cv2.imshow("Projection Image", projim)
        
        cv2.waitKey(1)

        global elastomer_tactile_pcd, indenter_pcd

        # Remove existing from plotter
        # Check if they exist to avoid errors, though vedo remove handles not-found gracefully usually
        viz.remove(elastomer_tactile_pcd)
        viz.remove(indenter_pcd)

        if VISUALIZE_TACTILE_PCD[0]:
            # Create new points
            elastomer_tactile_pcd = vedo.Points(elastomer_tactile_point_in_world, r=TACTILE_PCD_SIZE, c='yellow')
            viz.add(elastomer_tactile_pcd)

        if VISUALIZE_INDENTER_PCD[0]:
            indenter_pcd = vedo.Points(indenter_pts_in_world, r=INDENTER_PCD_SIZE, c='red')
            viz.add(indenter_pcd)

        viz.render()

        if save_frames:
            os.makedirs(save_folder, exist_ok=True)
            # Save shear image
            cv2.imwrite(os.path.join(save_folder, f"shear_{frame_idx:04d}.png"), sim_shear_img_uint8)
            # Save vedo screenshot
            viz.screenshot(os.path.join(save_folder, f"vedo_{frame_idx:04d}.png"), scale=1) # normal scale for frames to be fast

    def callback_fn(event):
        # translation
        step_size = step_size_container[0]
        camera_step_size = camera_step_size_container[0]
        angle_step_size = 5 # degrees
        moved = False
        if event.keypress == 'w':
            object_vedo.shift(np.array([0,step_size,0]))
            moved = True
        elif event.keypress == 's':
            object_vedo.shift(np.array([0,-step_size,0]))
            moved = True
        elif event.keypress == 'a':
            object_vedo.shift(np.array([-step_size,0,0]))
            moved = True
        elif event.keypress == 'd':
            object_vedo.shift(np.array([step_size,0,0]))
            moved = True
        elif event.keypress == 'q':
            object_vedo.shift(np.array([0,0,step_size]))
            moved = True
        elif event.keypress == 'e':
            object_vedo.shift(np.array([0,0,-step_size]))
            moved = True
        # rotation
        elif event.keypress == 'i':
            object_vedo.rotate(np.array([1,0,0]), angle_step_size)
            moved = True
        elif event.keypress == 'k':
            object_vedo.rotate(np.array([1,0,0]), -angle_step_size)
            moved = True
        elif event.keypress == 'j':
            object_vedo.rotate(np.array([0,1,0]), angle_step_size)
            moved = True
        elif event.keypress == 'l':
            object_vedo.rotate(np.array([0,1,0]), -angle_step_size)
            moved = True
        elif event.keypress == 'u':
            object_vedo.rotate(np.array([0,0,1]), angle_step_size)
            moved = True
        elif event.keypress == 'o':
            object_vedo.rotate(np.array([0,0,1]), -angle_step_size)
            moved = True
        elif event.keypress == 'z':
            object_vedo.set_mesh_pose_matrix(initial_object_pose)
            object_vedo.set_mesh_pose_matrix(initial_object_pose)
            moved = True
            print("Reset object pose")
        elif event.keypress == 'Up':
            # move camera up
            viz.camera.Azimuth(0)
            viz.camera.Elevation(camera_step_size)
            # set camera focal point to elastomer center
            viz.camera.SetFocalPoint(gsminifinger_vedo.elastomer_center[0], gsminifinger_vedo.elastomer_center[1], gsminifinger_vedo.elastomer_center[2])
            viz.camera.OrthogonalizeViewUp()
            viz.render()
        elif event.keypress == 'Down':
            # move camera down
            viz.camera.Azimuth(0)
            viz.camera.Elevation(-camera_step_size)
            viz.camera.SetFocalPoint(gsminifinger_vedo.elastomer_center[0], gsminifinger_vedo.elastomer_center[1], gsminifinger_vedo.elastomer_center[2])
            viz.camera.OrthogonalizeViewUp()
            viz.render()
        elif event.keypress == 'Left':
            # move camera left
            viz.camera.Azimuth(-camera_step_size)
            viz.camera.Elevation(0)
            viz.camera.SetFocalPoint(gsminifinger_vedo.elastomer_center[0], gsminifinger_vedo.elastomer_center[1], gsminifinger_vedo.elastomer_center[2])
            viz.camera.OrthogonalizeViewUp()
            viz.render()
        elif event.keypress == 'Right':
            # move camera right
            viz.camera.Azimuth(camera_step_size)
            viz.camera.Elevation(0)
            viz.camera.SetFocalPoint(gsminifinger_vedo.elastomer_center[0], gsminifinger_vedo.elastomer_center[1], gsminifinger_vedo.elastomer_center[2])
            viz.camera.OrthogonalizeViewUp()
            viz.render()
        elif event.keypress == 'm':
            # Handle point size scaling for high-res screenshot
            scale_factor = 10
            
            temp_actors = []
            original_actors = []
            
            # Use fixed world-space size for high-res screenshot visibility
            # 0.001 is 1mm radius
            sphere_r = 0.00035
            
            # Replace Points with Spheres temporarily
            if VISUALIZE_TACTILE_PCD[0]:
                pts = elastomer_tactile_pcd.points
                if len(pts) > 0:
                    print("Replacing tactile points with spheres for screenshot...")
                    sph = vedo.Spheres(pts, r=sphere_r, c='yellow')
                    temp_actors.append(sph)
                    original_actors.append(elastomer_tactile_pcd)
                    viz.remove(elastomer_tactile_pcd)
                    viz.add(sph)
            
            if VISUALIZE_INDENTER_PCD[0]:
                pts = indenter_pcd.points
                if len(pts) > 0:
                    print("Replacing indenter points with spheres for screenshot...")
                    sph = vedo.Spheres(pts, r=0.001, c='red')
                    temp_actors.append(sph)
                    original_actors.append(indenter_pcd)
                    viz.remove(indenter_pcd)
                    viz.add(sph)
                
            viz.render()
            
            timestamp = time.time()
            filename = f'hydroshear_demo_vedo_screenshot_{timestamp}.png'
            viz.screenshot(filename, scale=scale_factor)
            print(f"Screenshot saved to {filename}")
            
            # Save shear image if available
            if shear_image_container[0] is not None:
                filename_shear = f'hydroshear_demo_shear_image_{timestamp}.png'
                cv2.imwrite(filename_shear, shear_image_container[0])
                print(f"Shear image saved to {filename_shear}")
            
            # Restore
            for temp in temp_actors:
                viz.remove(temp)
            for orig in original_actors:
                viz.add(orig)
            
            viz.render()
            
        elif event.keypress == 'c':
            if camera_mode_container[0] == 0:
                # Switch to Top-Down
                saved_camera_pose['pos'] = viz.camera.GetPosition()
                saved_camera_pose['focal'] = viz.camera.GetFocalPoint()
                saved_camera_pose['up'] = viz.camera.GetViewUp()
                
                # Top-down view parameters
                # Elastomer center is roughly (0, -0.06, -0.01) based on code
                # But visualizer seems to center near (0, -0.03, 0) for interaction
                center = gsminifinger_vedo.elastomer_center
                
                viz.camera.SetPosition(center[0], center[1], center[2] + 0.05)
                viz.camera.SetFocalPoint(center[0], center[1], center[2])
                viz.camera.SetViewUp(0, -1, 0)
                camera_mode_container[0] = 1
                print("Switched to Top-Down view")
            else:
                # Restore previous view
                viz.camera.SetPosition(saved_camera_pose['pos'])
                viz.camera.SetFocalPoint(saved_camera_pose['focal'])
                viz.camera.SetViewUp(saved_camera_pose['up'])
                camera_mode_container[0] = 0
                print("Restored view")
            viz.render()

        # Toggle auto-rotation
        elif event.keypress == 't':
            auto_rotate_container[0] = not auto_rotate_container[0]
            status = "enabled" if auto_rotate_container[0] else "disabled"
            print(f"Auto-rotation {status}")
        
        # Recording controls
        elif event.keypress == 'r':
            recording_container[0] = not recording_container[0]
            if recording_container[0]:
                recorded_poses.clear()
                recorded_poses.append(object_vedo.get_mesh_pose_matrix())
                print("Recording started...")
            else:
                np.save("recorded_poses.npy", np.array(recorded_poses))
                print(f"Recording stopped. Saved {len(recorded_poses)} poses to recorded_poses.npy")
        
        elif event.keypress == 'p':
            try:
                print("Loading recorded_poses.npy...")
                loaded_poses = np.load("recorded_poses.npy")
                print(f"Replaying {len(loaded_poses)} poses...")
                for pose in loaded_poses:
                    object_vedo.set_mesh_pose_matrix(pose)
                    update_simulation()
                    # time.sleep(0.01) # small delay if needed, but simulation might be slow enough
                print("Replay finished.")
            except FileNotFoundError:
                print("No recorded_poses.npy found.")
            except Exception as e:
                print(f"Error during playback: {e}")

        elif event.keypress == 'v':
            # Save video
            if not os.path.exists("recorded_poses.npy"):
                print("No recorded poses to save as video.")
            else:
                print("Rendering video...")
                # Hide slider before video recording if it exists
                if VISUALIZE_SLIDER and slider_widget:
                    slider_widget.off()
                    slider_camera_widget.off()

                # use scale=5 for high resolution video as requested
                # Use ffmpeg backend directly if cv backend fails or produces warnings
                # Also ensure dimensions are even numbers which is often required by codecs
                video = vedo.Video("hydroshear_demo.mp4", fps=10, scale=3, backend='ffmpeg') 
                
                try:
                    loaded_poses = np.load("recorded_poses.npy")
                    for pose in loaded_poses:
                        object_vedo.set_mesh_pose_matrix(pose)
                        update_simulation()
                        temp_actors = []
                        original_actors = []
                        if VISUALIZE_TACTILE_PCD[0]:
                            pts = elastomer_tactile_pcd.points
                            if len(pts) > 0:
                                print("Replacing tactile points with spheres for screenshot...")
                                sph = vedo.Spheres(pts, r=0.0001, c='yellow')
                                temp_actors.append(sph)
                                original_actors.append(elastomer_tactile_pcd)
                                viz.remove(elastomer_tactile_pcd)
                                viz.add(sph)
                        
                        if VISUALIZE_INDENTER_PCD[0]:
                            pts = indenter_pcd.points
                            if len(pts) > 0:
                                print("Replacing indenter points with spheres for screenshot...")
                                sph = vedo.Spheres(pts, r=0.0001, c='red')
                                temp_actors.append(sph)
                                original_actors.append(indenter_pcd)
                                viz.remove(indenter_pcd)
                                viz.add(sph)
                        video.add_frame()
                        # clean up
                        for temp in temp_actors:
                            viz.remove(temp)
                        for orig in original_actors:
                            viz.add(orig)
                    video.close()
                    print("Video saved to hydroshear_demo.mp4")
                except Exception as e:
                    print(f"Error saving video: {e}")
                    video.close()
                
                # Show slider again after recording
                if VISUALIZE_SLIDER and slider_widget:
                    slider_widget.on()
                    slider_camera_widget.on()

        elif event.keypress == 'f':
            # Save frames
            if not os.path.exists("recorded_poses.npy"):
                print("No recorded poses to save frames.")
            else:
                print("Saving frames...")
                save_folder = "recorded_frames"
                
                # Hide slider before saving frames if it exists
                if VISUALIZE_SLIDER and slider_widget:
                    slider_widget.off()
                    slider_camera_widget.off()

                try:
                    loaded_poses = np.load("recorded_poses.npy")
                    for i, pose in enumerate(loaded_poses):
                        object_vedo.set_mesh_pose_matrix(pose)
                        update_simulation(save_frames=True, frame_idx=i, save_folder=save_folder)
                    print(f"Frames saved to {save_folder}/")
                except Exception as e:
                    print(f"Error saving frames: {e}")

                # Show slider again after saving frames
                if VISUALIZE_SLIDER and slider_widget:
                    slider_widget.on()
                    slider_camera_widget.on()

        elif event.keypress == '1':
            VISUALIZE_FORCE_ON_VEDO[0] = not VISUALIZE_FORCE_ON_VEDO[0]
            print(f"Force arrows {'enabled' if VISUALIZE_FORCE_ON_VEDO[0] else 'disabled'}")
            update_simulation()
            
        elif event.keypress == '2':
            VISUALIZE_SHEAR_ON_VEDO[0] = not VISUALIZE_SHEAR_ON_VEDO[0]
            print(f"Shear arrows {'enabled' if VISUALIZE_SHEAR_ON_VEDO[0] else 'disabled'}")
            update_simulation()
            
        elif event.keypress == '3':
            VISUALIZE_TACTILE_PCD[0] = not VISUALIZE_TACTILE_PCD[0]
            print(f"Tactile points {'enabled' if VISUALIZE_TACTILE_PCD[0] else 'disabled'}")
            update_simulation()

        elif event.keypress == '4':
            VISUALIZE_INDENTER_PCD[0] = not VISUALIZE_INDENTER_PCD[0]
            print(f"Indenter points {'enabled' if VISUALIZE_INDENTER_PCD[0] else 'disabled'}")
            update_simulation()

        elif event.keypress == '5':
            VISUALIZE_INDENTER_FRAME[0] = not VISUALIZE_INDENTER_FRAME[0]
            print(f"Indenter frame {'enabled' if VISUALIZE_INDENTER_FRAME[0] else 'disabled'}")
            update_simulation()

        elif event.keypress == '6':
            VISUALIZE_CONTACT_CIRCLE[0] = not VISUALIZE_CONTACT_CIRCLE[0]
            print(f"Contact circle {'enabled' if VISUALIZE_CONTACT_CIRCLE[0] else 'disabled'}")
            update_simulation()

        # # print mesh pose
        # print("Object mesh pose:", object_vedo.get_mesh_pose())
        # print("Finger mesh pose:\n", gsminifinger_vedo.get_mesh_pose())

        # # print camera pose
        # cam = viz.camera
        if event.keypress == 'y':
            print("Camera pos:\n", cam.GetPosition())
            print("Camera focal point:\n", cam.GetFocalPoint())
            print("Camera view up:\n", cam.GetViewUp())

        if moved:
            if recording_container[0]:
                recorded_poses.append(object_vedo.get_mesh_pose_matrix())
            update_simulation()

    viz.add_callback('on key press', callback_fn)


    # viz.at(1)
    # # red image
    # red_img = vedo.Image(np.ones((70, 70, 3), dtype=np.uint8) * np.array([255, 0, 0], dtype=np.uint8))
    # viz += red_img
    num_divs = gsminifinger_vedo.num_divs
    zero_sim_shear = np.zeros((num_divs[1], num_divs[0], 2))
    sim_shear_img = visualize_tactile_shear_image(zero_sim_shear, shear_force_threshold=5.0, resolution=70) * 255.0
    sim_shear_img_uint8 = sim_shear_img.astype(np.uint8)
    cv2.imshow("Simulated Shear Image", sim_shear_img_uint8)
    cv2.waitKey(0)

    viz.at(0).show()
    # viz.at(1).show()

    '''
    Camera pos:
    (0.2026717078204835, -0.02588413672850003, 0.01689244996347192)
    Camera focal point:
    (-1.7236867889953002e-05, -0.028585628562743243, 0.016816503663089)
    Camera view up:
    (-0.08440718641137869, 0.0021052786059308952, 0.9964291217563377)
    '''

    # vizcam0 = viz.renderers[0].MakeCamera()
    # # vizcam1 = viz.renderers[1].MakeCamera()1
    # viz.renderers[0].SetActiveCamera(vizcam0)
    # # viz.renderers[1].SetActiveCamera(vizcam1)

    # vizcam0.SetPosition(0.2026717078204835, -0.02588413672850003, 0.01689244996347192)
    # vizcam0.SetFocalPoint(-1.7236867889953002e-05, -0.028585628562743243, 0.016816503663089)
    # vizcam0.SetViewUp(-0.08440718641137869, 0.0021052786059308952, 0.9964291217563377)

    viz.interactive().close()