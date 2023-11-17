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

"""
Because remote service do not installed graphic interface, 
thus data needed to be visualized should be saved to local disk.
and do visualization locally.
"""

# read camera data from npz
data = np.load("./tmp/camera_info.npz")
extrinsics = data["extrinsics"]
intrinsics = data["intrinsics"]
WIDTH = data["W"]
HEIGHT = data["H"]

# read pcd data from npz
data = np.load("./tmp/cell_positions.npz")
cell_positions = data["cell_positions"]

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
