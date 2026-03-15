from isaacgym import torch_utils
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

euler_angle_1 = np.array([0.1, 0.2, 0.3])
euler_angle_2 = np.array([0.4, 0.5, 0.6])
euler_angle_sum = R.from_euler('xyz', euler_angle_1) * R.from_euler('xyz', euler_angle_2)

axis_angle_1 = R.from_euler('xyz', euler_angle_1).as_rotvec()
axis_angle_2 = R.from_euler('xyz', euler_angle_2).as_rotvec()

quat_1 = R.from_rotvec(axis_angle_1).as_quat()
quat_2 = R.from_rotvec(axis_angle_2).as_quat()

angle_1 = torch.tensor(np.linalg.norm(axis_angle_1))
axis_1 = torch.tensor(axis_angle_1 / angle_1)
torch_quat_1 = torch_utils.quat_from_angle_axis(angle_1, axis_1)
angle_2 = torch.tensor(np.linalg.norm(axis_angle_2))
axis_2 = torch.tensor(axis_angle_2 / angle_2)
torch_quat_2 = torch_utils.quat_from_angle_axis(angle_2, axis_2)

torch_quat = torch_utils.quat_mul(torch_quat_1, torch_quat_2)
torch_axis_angle = R.from_quat(torch_quat.numpy()).as_rotvec()
quat = R.from_quat(quat_1) * R.from_quat(quat_2)

print((R.from_rotvec(axis_angle_1) * R.from_rotvec(axis_angle_2)).as_quat())

print(f"Quat results: {quat.as_quat()} | {torch_quat.numpy()}")
print(f"Result: {quat.as_euler('xyz')} | GT: {euler_angle_sum.as_euler('xyz')} | Torch: {R.from_rotvec(torch_axis_angle).as_euler('xyz')}")


