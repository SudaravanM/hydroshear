import open3d as o3d
import numpy as np
import cv2
import pytorch_volumetric as pv
from rl.demo_utils.utils.viz_utils import visualize_tactile_shear_image
from rl.demo_utils.hydrosoft_viser import HydroSoftSensor
import torch
import torch.nn.functional as F
import viser
import time
from scipy.spatial.transform import Rotation as R_scipy
import point_cloud_utils as pcu

def _to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32))
    return x.to(dtype=torch.float32)

if __name__ == '__main__':
    elastomer_path = './demo_assets/elastomer.obj'
    peg_path = './demo_assets/35mm_sphere.obj'
    # peg_scale = 1.5e-2
    peg_scale = 1.0
    
    # Create viser server
    server = viser.ViserServer()
    
    # Load meshes for processing
    elastomer_mesh = o3d.io.read_triangle_mesh(elastomer_path)
    # roll elastomer by 90 degrees along x axis
    R = elastomer_mesh.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))
    elastomer_mesh.rotate(R, center=elastomer_mesh.get_center())
    bbox = elastomer_mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    elastomer_mesh.translate(-center)
    bbox = elastomer_mesh.get_axis_aligned_bounding_box()
    # compute vertex normals
    elastomer_mesh.compute_vertex_normals()
    
    server.scene.add_mesh_simple(
        "elastomer_mesh_temp",
        vertices=np.asarray(elastomer_mesh.vertices),
        faces=np.asarray(elastomer_mesh.triangles),
        color=(119,136,153),
        wireframe=False,
        side="back",
        opacity=0.7
    )
    
    # create grid points on elastomer
    elastomer_dims = bbox.get_extent()
    num_divs = [7,9]
    margin = 0.003
    div_sz = (elastomer_dims[:2] - margin * 2.) / (np.array(num_divs) + 1)
    tactile_points_dx = np.amin(div_sz)
    x_pts = np.linspace(center[0] - tactile_points_dx * (num_divs[0] + 1.) / 2., center[0] + tactile_points_dx * (num_divs[0] + 1.) / 2., num_divs[0] + 2)[1:-1]
    y_pts = np.linspace(center[1] - tactile_points_dx * (num_divs[1] + 1.) / 2., center[1] + tactile_points_dx * (num_divs[1] + 1.) / 2., num_divs[1] + 2)[1:-1]
    xv, yv = np.meshgrid(x_pts, y_pts)
    zv = np.zeros_like(xv)
    pts3d = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)
    ray3d = np.array([0, 0, 1], dtype=np.float32).reshape(1, 3).repeat(pts3d.shape[0], axis=0)
    # combine pts3d and ray3d to form rays and convert to open3d tensor
    rays = np.concatenate([pts3d, ray3d], axis=-1)
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    
    elastomer_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(elastomer_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(elastomer_mesh_t)
    ans = scene.cast_rays(rays)
    zv = ans['t_hit'].numpy().reshape(-1)
    zv = np.where(np.isfinite(zv), zv, 0.0)

    tactile_pts = np.stack([xv.flatten(), yv.flatten(), zv], axis=-1)
    # done creating grid points (tactile points)
    
    peg_mesh = o3d.io.read_triangle_mesh(peg_path)
    peg_bbox = peg_mesh.get_axis_aligned_bounding_box()
    peg_mesh.scale(peg_scale, center=peg_bbox.get_center())
    peg_mesh.translate(-peg_bbox.get_center())
    peg_mesh.compute_vertex_normals()
    # print bbox
    peg_bbox = peg_mesh.get_axis_aligned_bounding_box()
    
    
    # setup pytorch-volumetric sdf stuff
    elastomer_pv = pv.MeshObjectFactory(elastomer_path)
    elastomer_pv._mesh = elastomer_mesh
    elastomer_pv._mesht = None
    elastomer_pv.precompute_sdf()
    elastomer_sdf_gt = pv.MeshSDF(elastomer_pv)
    elastomer_sdf = pv.CachedSDF(
        object_name='elastomer',
        resolution=0.001,
        range_per_dim=elastomer_pv.bounding_box(padding=0.1),
        gt_sdf=elastomer_sdf_gt,
        device='cpu',
        cache_path='elastomer_cached_sdf.pkl',
        clean_cache=False
    )

    peg_pv = pv.MeshObjectFactory(peg_path)
    peg_pv._mesh = peg_mesh
    peg_pv._mesht = None
    peg_pv.precompute_sdf()
    # peg_sdf_gt = pv.MeshSDF(peg_pv)
    # peg_sdf = pv.CachedSDF(
    #     object_name='peg',
    #     resolution=0.001,
    #     range_per_dim=peg_pv.bounding_box(padding=0.1),
    #     gt_sdf=peg_sdf_gt,
    #     device='cpu',
    #     cache_path='peg_cached_sdf.pkl',
    #     clean_cache=False
    # )
    peg_sdf = pv.MeshSDF(peg_pv)

    # peg_local_pts, peg_local_normals, _ = pv.sample_mesh_points(peg_pv, num_points=900)
    # peg_local_pts_random, peg_local_normals_random, _ = pv.sample_mesh_points(peg_pv, num_points=int(1e8), clean_cache=True)
    # poisson_disk_idx = pcu.downsample_point_cloud_poisson_disk(peg_local_pts_random.numpy(), radius=peg_scale, target_num_samples=-1)
    # peg_local_pts = peg_local_pts_random[poisson_disk_idx]
    # peg_local_normals = peg_local_normals_random[poisson_disk_idx]
    # peg_local_pts = peg_local_pts.cpu().numpy()
    
    peg_local_pts, _, _ = pv.sample_mesh_points(peg_pv, num_points=int(1e3), clean_cache=True)
    peg_local_pts = peg_local_pts.cpu().numpy()
    
    # Initialize peg pose
    peg_pose = np.eye(4)
    peg_pose[:3, 3] = np.array([0, 0, 0.02])
    
    # Add elastomer mesh to viser
    elastomer_vertices = np.asarray(elastomer_mesh.vertices)
    elastomer_faces = np.asarray(elastomer_mesh.triangles)
    # server.scene.add_mesh_simple(
    #     "elastomer",
    #     vertices=elastomer_vertices,
    #     faces=elastomer_faces,
    #     color=(200, 200, 200),
    #     wireframe=False,
    #     opacity=0.1
    # )
    
    # Add bounding box to viser
    # bbox_points = np.asarray(bbox.get_box_points())
    # bbox_lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    # for i, (start, end) in enumerate(bbox_lines):
    #     server.scene.add_spline_catmull_rom(
    #         f"bbox_line_{i}",
    #         positions=np.array([bbox_points[start], bbox_points[end]]),
    #         color=(255, 0, 0),
    #         line_width=2.0,
    #     )
    
    # Add tactile points to viser
    server.scene.add_point_cloud(
        "tactile_points",
        points=tactile_pts,
        colors=(0, 255, 0),
        point_size=0.0001,
    )
    
    # # Add origin frame
    # server.scene.add_frame(
    #     "origin",
    #     wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    #     position=np.array([0.0, 0.0, 0.0]),
    #     axes_length=0.025,
    #     axes_radius=0.002,
    # )
    
    # Initialize HydroSoft sensor
    hydrosoft = HydroSoftSensor(server)
    hydrosoft.initialize(1, peg_local_pts.shape[0])
    
    # Add transform controls for interactive peg manipulation
    peg_quat_xyzw = R_scipy.from_matrix(peg_pose[:3, :3]).as_quat()
    peg_quat_wxyz = np.array([peg_quat_xyzw[3], peg_quat_xyzw[0], peg_quat_xyzw[1], peg_quat_xyzw[2]])
    
    transform_controls = server.scene.add_transform_controls(
        "/peg_transform",
        wxyz=peg_quat_wxyz,
        position=peg_pose[:3, 3],
        scale=0.04,
    )
    
    # Create GUI controls
    with server.gui.add_folder("Peg Control"):
        step_size = server.gui.add_slider("Step Size", min=0.0001, max=0.01, step=0.0001, initial_value=0.001)
        rotation_step = server.gui.add_slider("Rotation Step (deg)", min=1, max=30, step=1, initial_value=10)
        
        go_left = server.gui.add_button("Left (A)")
        go_right = server.gui.add_button("Right (D)")
        go_forward = server.gui.add_button("Forward (W)")
        go_backward = server.gui.add_button("Backward (S)")
        go_up = server.gui.add_button("Up (E)")
        go_down = server.gui.add_button("Down (Q)")
        
        twist_left = server.gui.add_button("Twist Left (Z)")
        twist_right = server.gui.add_button("Twist Right (X)")
        rotate_x_left = server.gui.add_button("Rotate X Left (V)")
        rotate_x_right = server.gui.add_button("Rotate X Right (C)")
    
    with server.gui.add_folder("Tactile Images"):
        marker_displacement_img = server.gui.add_image(
            image=np.zeros((300, 200, 3), dtype=np.uint8),
            format="jpeg"
        )
        sdf_img = server.gui.add_image(
            image=np.zeros((300, 200, 3), dtype=np.uint8),
            format="jpeg"
        )
    with server.gui.add_folder("Projection Image"):
        projection_img = server.gui.add_image(
            image=np.zeros((800, 600, 3), dtype=np.uint8),
            format="jpeg"
        )
    
    def update_peg_visualization():
        """Update peg mesh and point cloud visualization in viser"""
        # Get current peg mesh vertices
        peg_mesh_temp = o3d.io.read_triangle_mesh(peg_path)
        # get bbox of peg_mesh
        peg_bbox = peg_mesh_temp.get_axis_aligned_bounding_box()
        peg_mesh_temp.scale(peg_scale, center=peg_bbox.get_center())
        peg_mesh_temp.translate(-peg_bbox.get_center())
        peg_mesh_temp.compute_vertex_normals()
        peg_mesh_temp.transform(peg_pose)
        peg_vertices = np.asarray(peg_mesh_temp.vertices)
        peg_faces = np.asarray(peg_mesh_temp.triangles)
        
        # Update peg mesh
        server.scene.add_mesh_simple(
            "peg_mesh",
            vertices=peg_vertices,
            faces=peg_faces,
            color=(100, 100, 255),
            material="standard",
            flat_shading=True,
            cast_shadow=False,
            receive_shadow=False,
            wireframe=False,
            side="front",
            opacity=0.5,
        )
        
        # Update peg points
        peg_world_pts = (peg_pose[:3, :3] @ peg_local_pts.T).T + peg_pose[:3, 3]
        sdf_t, _ = elastomer_sdf(_to_torch(peg_world_pts))
        sdf_np = sdf_t.detach().cpu().numpy().reshape(-1)
        colors = np.zeros((peg_world_pts.shape[0], 3), dtype=np.uint8)
        colors[sdf_np <= 0] = np.array([255, 255, 0], dtype=np.uint8)
        colors[sdf_np > 0] = np.array([0, 0, 255], dtype=np.uint8)
        server.scene.add_point_cloud(
            "peg_points",
            points=peg_world_pts[sdf_np > 0],
            colors=colors[sdf_np > 0],
            point_size=0.0001,
        )
        server.scene.add_point_cloud(
            "peg_points_in_contact",
            points=peg_world_pts[sdf_np <= 0],
            colors=colors[sdf_np <= 0],
            point_size=0.0001,
        )
        return peg_world_pts
    
    def check_markers(peg_world_pts):
        tactile_pts_elastomer = _to_torch(tactile_pts)
        peg_pose_inv = _to_torch(np.linalg.inv(peg_pose))
        tactile_pts_peg = (peg_pose_inv[:3, :3] @ tactile_pts_elastomer.T).T + peg_pose_inv[:3, 3]
        peg_pts = _to_torch(peg_world_pts)
        tactile_pts_height = F.relu(-peg_sdf(tactile_pts_peg)[0])
        elast_sdf_vals = elastomer_sdf(peg_pts)[0]
        marker_displacement = hydrosoft.get_marker_displacement(
            tactile_pts_elastomer.unsqueeze(0),
            tactile_pts_height.unsqueeze(0),
            peg_pts.unsqueeze(0),
            elast_sdf_vals.unsqueeze(0)
        )

        
        
        marker_displacement = marker_displacement.squeeze().reshape(num_divs[1], num_divs[0], 3).cpu().numpy()[..., :2][..., ::-1]
        # print(marker_displacement.shape)
        # print(np.max(marker_displacement[...,0]))
        # print(np.max(marker_displacement[...,1]))
        # print("meow")
        
        vis_image = visualize_tactile_shear_image(marker_displacement, shear_force_threshold=5.0, resolution=70).swapaxes(0, 1)
        sdf_image = tactile_pts_height.detach().cpu().numpy().reshape(num_divs[1], num_divs[0])
        normalize_sdf_image = (sdf_image - sdf_image.min()) / (sdf_image.max() - sdf_image.min() + 1e-8)
        
        # Update images in GUI
        vis_image_uint8 = (vis_image * 255).astype(np.uint8)
        sdf_image_uint8 = (normalize_sdf_image * 255).astype(np.uint8)
        # Convert grayscale to RGB for sdf image
        sdf_image_rgb = np.stack([sdf_image_uint8, sdf_image_uint8, sdf_image_uint8], axis=-1)
        
        marker_displacement_img.image = vis_image_uint8
        sdf_img.image = sdf_image_rgb
        
        sdf_t, _ = elastomer_sdf(_to_torch(peg_world_pts))
        sdf_np = sdf_t.detach().cpu().numpy().reshape(-1)
        server.scene.add_point_cloud(
            "projected_points_in_contact",
            points=peg_world_pts[sdf_np <= 0] + hydrosoft.hydrosoft_forces[sdf_t[None, :] <= 0].cpu().numpy(),
            colors=(0, 255, 0),
            point_size=0.0001,
        )
        
        # make image that visualizes the tactile points
        min_x_tactile_pts = np.min(tactile_pts[:, 0])
        max_x_tactile_pts = np.max(tactile_pts[:, 0])
        min_y_tactile_pts = np.min(tactile_pts[:, 1])
        max_y_tactile_pts = np.max(tactile_pts[:, 1])
        
        im_size_x = 600
        im_size_y = 800
        
        scale_projim_x = im_size_x / (max_x_tactile_pts - min_x_tactile_pts)
        scale_projim_y = im_size_y / (max_y_tactile_pts - min_y_tactile_pts)
        
        projim = np.zeros((800, 600, 3), dtype=np.uint8)
        for i in range(tactile_pts.shape[0]):
            x = tactile_pts[i, 0]
            y = tactile_pts[i, 1]
            z = sdf_image.flatten()[i]
            u = int((x - min_x_tactile_pts) * scale_projim_x)
            v = int((y - min_y_tactile_pts) * scale_projim_y)
            # color_val = int((z - np.min(sdf_image)) / (np.max(sdf_image) - np.min(sdf_image) + 1e-8) * 255)
            color = (0, 255, 0)
            cv2.circle(projim, (u, im_size_y - v), 2, color, -1)
            
        projected_pts = peg_world_pts[sdf_np <= 0] + hydrosoft.hydrosoft_forces[sdf_t[None, :] <= 0].cpu().numpy()
        for i in range(projected_pts.shape[0]):
            x = projected_pts[i, 0]
            y = projected_pts[i, 1]
            # z = projected_pts[i, 2]
            u = int((x - min_x_tactile_pts) * scale_projim_x)
            v = int((y - min_y_tactile_pts) * scale_projim_y)
            color = (0, 0, 255)
            cv2.circle(projim, (u, im_size_y - v), 5, color, -1)
        
        # get force arrows
        forces = -hydrosoft.hydrosoft_forces[sdf_t[None, :] <= 0].cpu().numpy()
        for i in range(projected_pts.shape[0]):
            x = projected_pts[i, 0]
            y = projected_pts[i, 1]
            u = int((x - min_x_tactile_pts) * scale_projim_x)
            v = int((y - min_y_tactile_pts) * scale_projim_y)
            fx = forces[i, 0]
            fy = forces[i, 1]
            # scale force for better visualization
            force_scale = 5000.0 * 10
            cv2.arrowedLine(projim, (u, im_size_y - v), (int(u + fx * force_scale), int(im_size_y - (v + fy * force_scale))), (0, 0, 255), 1, tipLength=0.3)
        
        # draw shear arrows
        if hydrosoft.Mshear is not None:
            shear_vectors = hydrosoft.Mshear[0].numpy()
            for i in range(tactile_pts.shape[0]):
                x = tactile_pts[i, 0]
                y = tactile_pts[i, 1]
                u = int((x - min_x_tactile_pts) * scale_projim_x)
                v = int((y - min_y_tactile_pts) * scale_projim_y)
                sx = shear_vectors[i, 0]
                sy = shear_vectors[i, 1]
                shear_scale = 100.0
                cv2.arrowedLine(projim, (u, im_size_y - v), (int(u + sx * shear_scale), int(im_size_y - (v + sy * shear_scale))), (0, 255, 0), 1, tipLength=0.3)
        
        projection_img.image = projim
                
        
        
    _updating = False

    def redraw():
        global _updating
        if _updating:
            return
        _updating = True
        try:
            peg_world = update_peg_visualization()
            check_markers(peg_world)
        finally:
            _updating = False

    # Set up button callbacks
    @go_left.on_click
    def _(_):
        global peg_pose
        peg_pose[:3, 3] += np.array([-step_size.value, 0.0, 0.0])
        transform_controls.position = peg_pose[:3, 3]
        redraw()
    
    @go_right.on_click
    def _(_):
        global peg_pose
        peg_pose[:3, 3] += np.array([step_size.value, 0.0, 0.0])
        transform_controls.position = peg_pose[:3, 3]
        redraw()
    
    @go_forward.on_click
    def _(_):
        global peg_pose
        peg_pose[:3, 3] += np.array([0.0, step_size.value, 0.0])
        transform_controls.position = peg_pose[:3, 3]
        redraw()
    
    @go_backward.on_click
    def _(_):
        global peg_pose
        peg_pose[:3, 3] += np.array([0.0, -step_size.value, 0.0])
        transform_controls.position = peg_pose[:3, 3]
        redraw()
    
    @go_up.on_click
    def _(_):
        global peg_pose
        peg_pose[:3, 3] += np.array([0.0, 0.0, step_size.value])
        transform_controls.position = peg_pose[:3, 3]
        redraw()
    
    @go_down.on_click
    def _(_):
        global peg_pose
        peg_pose[:3, 3] += np.array([0.0, 0.0, -step_size.value])
        transform_controls.position = peg_pose[:3, 3]
        redraw()
    
    @twist_left.on_click
    def _(_):
        global peg_pose
        angle = np.deg2rad(rotation_step.value)
        R = R_scipy.from_euler('xyz', [0, 0, angle]).as_matrix()
        peg_pose[:3, :3] = R @ peg_pose[:3, :3]
        quat_xyzw = R_scipy.from_matrix(peg_pose[:3, :3]).as_quat()
        transform_controls.wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        redraw()
    
    @twist_right.on_click
    def _(_):
        global peg_pose
        angle = -np.deg2rad(rotation_step.value)
        R = R_scipy.from_euler('xyz', [0, 0, angle]).as_matrix()
        peg_pose[:3, :3] = R @ peg_pose[:3, :3]
        quat_xyzw = R_scipy.from_matrix(peg_pose[:3, :3]).as_quat()
        transform_controls.wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        redraw()
    
    @rotate_x_left.on_click
    def _(_):
        global peg_pose
        angle = np.deg2rad(rotation_step.value)
        R = R_scipy.from_euler('xyz', [angle, 0, 0]).as_matrix()
        peg_pose[:3, :3] = R @ peg_pose[:3, :3]
        quat_xyzw = R_scipy.from_matrix(peg_pose[:3, :3]).as_quat()
        transform_controls.wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        redraw()
    
    @rotate_x_right.on_click
    def _(_):
        global peg_pose
        angle = -np.deg2rad(rotation_step.value)
        R = R_scipy.from_euler('xyz', [angle, 0, 0]).as_matrix()
        peg_pose[:3, :3] = R @ peg_pose[:3, :3]
        quat_xyzw = R_scipy.from_matrix(peg_pose[:3, :3]).as_quat()
        transform_controls.wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        redraw()
    
    # Add callback for transform control updates
    @transform_controls.on_update
    def _(_) -> None:
        global peg_pose
        # Update peg_pose from transform control
        wxyz = transform_controls.wxyz
        position = transform_controls.position
        
        # Convert quaternion to rotation matrix
        quat_xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
        rotation_matrix = R_scipy.from_quat(quat_xyzw).as_matrix()
        
        # Update peg pose
        peg_pose[:3, :3] = rotation_matrix
        peg_pose[:3, 3] = position
        
        # Update visualization
        redraw()
    
    # Initialize visualization
    redraw()

    
    print("Viser server is running. Open the URL shown above in your browser.")
    print("Use the GUI controls or drag the transform gizmo to manipulate the peg.")
    
    # Keep the server running
    while True:
        cv2.imshow('test', marker_displacement_img.image)
        cv2.imshow('projection', projection_img.image)
        cv2.waitKey(1)
        time.sleep(0.1)