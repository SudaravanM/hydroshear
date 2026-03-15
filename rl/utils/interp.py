
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

def wrap_angle(angle):
    """Wrap angle to [-pi, pi] range"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def wrap_axis_angle(axis_angle):
    """Wrap axis-angle representation to ensure smooth, minimal rotation (vectorized)"""
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    
    # Avoid division by zero - extract axis
    epsilon = 1e-6
    axis = np.where(angle > epsilon, axis_angle / angle, 0.0)
    
    # Wrap angle to [-pi, pi]
    wrapped_angle = wrap_angle(angle)
    
    result = axis * wrapped_angle
    
    # Clamp each component to be safe (should already be in range, but numerical errors can occur)
    result = np.clip(result, -np.pi, np.pi)
    
    return result

def quat_shortest_path(q1, q2):
    """Ensure quaternions take shortest path by choosing sign"""
    # q and -q represent the same rotation, so choose the one with positive dot product
    # This ensures we take the shortest path
    dot = np.sum(q1 * q2, axis=-1, keepdims=True)
    q2_corrected = np.where(dot < 0, -q2, q2)
    return q2_corrected

def quat_slerp_batch(q1, q2, t):
    """Batch quaternion SLERP that works with [N, 4] shaped arrays"""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    t = np.asarray(t, dtype=float)
    
    # Ensure q1 and q2 are at least 2D
    if q1.ndim == 1:
        q1 = q1[np.newaxis, :]
        q2 = q2[np.newaxis, :]
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Compute dot product
    dot = np.sum(q1 * q2, axis=-1, keepdims=True)
    
    # Clamp dot product to avoid numerical errors
    dot = np.clip(dot, -1.0, 1.0)
    
    # If dot is very close to 1, use linear interpolation
    epsilon = 1e-6
    theta = np.arccos(np.abs(dot))
    
    # SLERP formula
    sin_theta = np.sin(theta)
    
    # When theta is small, use linear interpolation to avoid division by zero
    use_linear = sin_theta < epsilon
    
    # SLERP weights
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    # Linear interpolation weights (fallback)
    w1_linear = 1 - t
    w2_linear = t
    
    # Choose between SLERP and linear
    w1 = np.where(use_linear, w1_linear, w1)
    w2 = np.where(use_linear, w2_linear, w2)
    
    # Interpolate
    result = w1 * q1 + w2 * q2
    
    # Normalize
    result = result / np.linalg.norm(result, axis=-1, keepdims=True)
    
    if squeeze_output:
        result = result.squeeze(0)
    
    return result

def quat_to_axis_angle_slerp(q1, q2, t, order='xyz', degrees=True, w_first=False):
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    if w_first:
        q1 = np.roll(q1, -1, axis=-1)
        q2 = np.roll(q2, -1, axis=-1)
    t = np.asarray(t, dtype=float)
    
    # Use batch SLERP if we have batches
    if q1.ndim > 1:
        result_quat = quat_slerp_batch(q1, q2, t)
        return R.from_quat(result_quat).as_rotvec()
    else:
        # Original implementation for single quaternions
        key_times = [0.0, 1.0]
        key_rots = R.from_quat(np.stack([q1, q2], axis=0))
        slerp = Slerp(key_times, key_rots)
        r = slerp(t)
        return r.as_rotvec()