'''
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''

import numpy as np
import cv2


def visualize_tactile_shear_image(tactile_shear_force,
                                  shear_force_threshold=0.0005,
                                  resolution=30):
    """
    Refer: https://github.com/eanswer/TactileSimulation/blob/main/utils/tactile_utils.py
    Normal forces define the color of the shear field.
    Shear forces define the direction and magnitude of the shear field.
    
    Visualize the tactile shear field.

    Args:
        tactile_normal_force (np.ndarray): Array of tactile normal forces.
        tactile_shear_force (np.ndarray): Array of tactile shear forces.
        normal_force_threshold (float): Threshold for normal force visualization.
        shear_force_threshold (float): Threshold for shear force visualization.
        resolution (int): Resolution for the visualization.

    Returns:
        np.ndarray: Image visualizing the tactile shear forces.
    """
    nrows = tactile_shear_force.shape[0]
    ncols = tactile_shear_force.shape[1]

    imgs_tactile = np.zeros((nrows * resolution, ncols * resolution, 3), dtype=float)

    # print('(min, max) tactile normal force: ', np.min(tactile_normal_force), np.max(tactile_normal_force))
    # print('(min, max) tactile shear force: ', np.min(tactile_shear_force), np.max(tactile_shear_force))
    
    # tactile_arrays = tactile_shear_force
    # lengths = np.linalg.norm(tactile_arrays, axis=-1)
        
    # max_length = np.max(lengths) + 1e-5
    # normalized_tactile_arrays = tactile_arrays / (max_length / 30.)
    
    # tactile_shear_force = normalized_tactile_arrays
    
    
    for row in range(nrows):
        for col in range(ncols):
            loc0_x = row * resolution + resolution // 2
            loc0_y = col * resolution + resolution // 2
            loc1_x = loc0_x + tactile_shear_force[row, col][0] / shear_force_threshold * resolution
            loc1_y = loc0_y + tactile_shear_force[row, col][1] / shear_force_threshold * resolution
            # color = (0.,
            #          max(0., 1. - tactile_normal_force[row][col] / normal_force_threshold),
            #          min(1., tactile_normal_force[row][col] / normal_force_threshold)
            #          )
            color = (
                0.,
                1.,
                0.,
            )

            cv2.arrowedLine(imgs_tactile,
                            (int(loc0_y), int(loc0_x)),
                            (int(loc1_y), int(loc1_x)),
                            color, 3, tipLength=0.2)

    return imgs_tactile


def visualize_penetration_depth(penetration_depth_img, resolution=5, depth_multiplier=300.):
    """
    Visualize the penetration depth.

    Args:
        penetration_depth_img (np.ndarray): Image of penetration depth.
        resolution (int): Resolution for the upsampling.
        depth_multiplier (float): Multiplier for the depth values.

    Returns:
        np.ndarray: Upsampled image visualizing the penetration depth.
    """
    # penetration_depth_img_upsampled = penetration_depth.repeat(resolution, 0).repeat(resolution, 1)
    penetration_depth_img_upsampled = np.kron(penetration_depth_img, np.ones((resolution, resolution)))
    penetration_depth_img_upsampled = np.clip(penetration_depth_img_upsampled, 0., 1.) * depth_multiplier
    return penetration_depth_img_upsampled