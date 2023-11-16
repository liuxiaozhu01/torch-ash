import numpy as np
import torch
import torch.nn as nn
from ash import (
    SparseDenseGrid,
    SparseDenseGridQuery,
    SparseDenseGridQueryBackward,
    SirenNet,
)

np.random.seed(114514)
torch.manual_seed(114514)

in_dim = 3
capacity = 1000
embedding_dim = 16
grid_dim = 8
device = torch.device("cuda:0")


grid = SparseDenseGrid(
    in_dim=in_dim,
    num_embeddings=capacity,
    embedding_dim=embedding_dim,
    grid_dim=grid_dim,
    device=device,
)

nn.init.uniform_(grid.embeddings, -1, 1)

grid_coords = grid_dim * (
    torch.arange(-1, 2, 1, dtype=torch.int, device=device)
    .view(-1, 1)
    .tile((1, in_dim))
)

(
    grid_coords,
    cell_coords,
    grid_indices,
    cell_indices,
) = grid.spatial_init_(grid_coords, dilation=1, bidirectional=True)

_grid_coords, _cell_coords, _grid_indices, _cell_indices = grid.items()


print(grid_coords)
print(_grid_coords)

# print(cell_coords)
# print(_cell_coords)

# print(grid_indices)
# print(_grid_indices)

# print(cell_indices)
# print(_cell_indices)