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

print(grid_coords)
print(_grid_coords)

# print(cell_coords)
# print(_cell_coords)

# print(grid_indices)
# print(_grid_indices)

# print(cell_indices)
# print(_cell_indices)