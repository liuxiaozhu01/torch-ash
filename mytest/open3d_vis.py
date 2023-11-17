import pdb
import traceback
import open3d # 0.17.0
import numpy as np
import torch
import torch.nn as nn
import os

print ('\n =============================== ')
print (f' ======  Open3D=={open3d.__version__}  ======= ')
print (' =============================== \n')

from ash import (
    SparseDenseGrid,
    SparseDenseGridQuery,
    SparseDenseGridQueryBackward,
    SirenNet,
    UnBoundedSparseDenseGrid,
)
from pyhocon import ConfigFactory

np.random.seed(114514)
torch.manual_seed(114514)

in_dim = 3
num_embeddings = 40000
embedding_dim = 16
grid_dim = 8
device = torch.device("cuda:0")
cell_size = 0.015
pcd_path = "/root/home/workspace/torch-ash/data/dtu_scan24/preprocessed/scaled_points.ply"

# tmp to store some data
tmp_file = "./tmp"
if not os.path.exists(tmp_file):
    os.makedirs(tmp_file)


grid = UnBoundedSparseDenseGrid(
    in_dim=in_dim,
    num_embeddings=num_embeddings,
    embedding_dim=embedding_dim,
    grid_dim=grid_dim,
    device=device,
    cell_size=cell_size,
)

nn.init.uniform_(grid.embeddings, -1, 1)

import trimesh
pcd = trimesh.load(pcd_path)
pcd_tensor = torch.tensor(pcd.vertices, dtype=torch.float32, device=device).reshape(-1, 3).contiguous()

(
    grid_coords,
    cell_coords,
    grid_indices,
    cell_indices,
) = grid.spatial_init_(pcd_tensor, dilation=1, bidirectional=True)

from data_provider import Dataset

# build dataset config befor load data
dataset_conf_dict = {
    "data_dir": "/root/home/workspace/torch-ash/data/dtu_scan24/preprocessed",
    "render_cameras_name": "cameras_sphere.npz",
    "object_cameras_name": "cameras_sphere.npz",
    "camera_outside_sphere": True,
    "scale_mat_scale": 1.1,
}
# construct conf from dict
data_conf = ConfigFactory.from_dict(dataset_conf_dict)
dataset = Dataset(data_conf)

WIDTH = dataset.W
HEIGHT = dataset.H

poses = dataset.pose_all.cpu().numpy()
print(poses.shape)
# print(poses[0])
# poses to extrinsics by inverse pose
extrinsics = np.linalg.inv(poses)
print(extrinsics.shape)
# print(extrinsics[0])

intrinsics = dataset.intrinsics_all.cpu().numpy()[:, :3, :3]

# convert extrinsics and intrinsics to numpy.ndarray[numpy.float64[3, 3]] and numpy.ndarray[numpy.float64[4, 4]]
extrinsics = extrinsics.astype(np.float64)
intrinsics = intrinsics.astype(np.float64)

# store extrinsics and intrinsics and W, H to npz
# np.savez(os.path.join(tmp_file, "extrinsics.npz"), extrinsics=extrinsics)
# np.savez(os.path.join(tmp_file, "intrinsics.npz"), intrinsics=intrinsics)
camera_info = {
    "extrinsics": extrinsics,
    "intrinsics": intrinsics,
    "W": WIDTH,
    "H": HEIGHT,
}
np.savez(os.path.join(tmp_file, "camera_info.npz"), **camera_info)

# build grid
cell_positions = grid.cell_to_world(grid_coords, cell_coords).cpu().numpy()
# store cell_positions to npz
np.savez(os.path.join(tmp_file, "cell_positions.npz"), cell_positions=cell_positions)




# --- open3d visualization ---# 
## visualizer
vizualizer = open3d.visualization.Visualizer()
vizualizer.create_window()
vizualizer.create_window(width=WIDTH, height=HEIGHT)
## pointcloud
pointcloud = open3d.geometry.PointCloud()
pointcloud.points = open3d.utility.Vector3dVector(cell_positions)

## camera
# cameraLines = open3d.geometry.LineSet.create_camera_visualization(view_width_px=WIDTH, view_height_px=HEIGHT, intrinsic=intrinsics[0], extrinsic=extrinsics[0])
for intrinsic, extrinsic in zip(intrinsics, extrinsics):
    cameraLines = open3d.geometry.LineSet.create_camera_visualization(view_width_px=WIDTH, view_height_px=HEIGHT, intrinsic=intrinsic, extrinsic=extrinsic)
    vizualizer.add_geometry(cameraLines)



## add geometry
vizualizer.add_geometry(pointcloud)
# vizualizer.add_geometry(cameraLines)

## export visualization
vizualizer.run()

## debug
pdb.set_trace()
