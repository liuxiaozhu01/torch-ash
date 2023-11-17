import numpy as np
import torch
import torch.nn as nn
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

_grid_coords, _cell_coords, _grid_indices, _cell_indices = grid.items()

"""
print(grid_coords)
print(_grid_coords)

# print(cell_coords)
# print(_cell_coords)

# print(grid_indices)
# print(_grid_indices)

# print(cell_indices)
# print(_cell_indices)
"""

"""
# test whether the grid_coords and _grid_coords are the same
# assert torch.all(grid_coords == _grid_coords)

# export grid_coords to ply
grid_coords_np = grid_coords.reshape(-1, 3).cpu().numpy()
grid_coords_pcd = trimesh.PointCloud(grid_coords_np)
grid_coords_pcd.export("grid_coords.ply")
# export _grid_coords to ply
_grid_coords_np = _grid_coords.reshape(-1, 3).cpu().numpy()
_grid_coords_pcd = trimesh.PointCloud(_grid_coords_np)
_grid_coords_pcd.export("_grid_coords.ply")

# export cell_coords to ply
cell_coords_np = cell_coords.reshape(-1, 3).cpu().numpy()
cell_coords_pcd = trimesh.PointCloud(cell_coords_np)
cell_coords_pcd.export("cell_coords.ply")
# export _cell_coords to ply
_cell_coords_np = _cell_coords.reshape(-1, 3).cpu().numpy()
_cell_coords_pcd = trimesh.PointCloud(_cell_coords_np)
_cell_coords_pcd.export("_cell_coords.ply")



cell_positions = grid.cell_to_world(grid_coords, cell_coords)
# export cell_positions to ply
cell_positions_np = cell_positions.reshape(-1, 3).cpu().numpy()
cell_positions_pcd = trimesh.PointCloud(cell_positions_np)
cell_positions_pcd.export("cell_positions.ply")
"""

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

cam_origins = dataset.pose_all[:, :3, 3].reshape(-1, 3)

cell_positions = grid.cell_to_world(grid_coords, cell_coords)

# combine cells_positions and cam_origins
v = torch.cat([cell_positions, cam_origins], dim=0)
pcd = trimesh.PointCloud(v.cpu().numpy())
pcd.export("cell_positions_and_cam_origins.ply")