from data_provider import Dataset
from pyhocon import ConfigFactory
import trimesh
import torch
import numpy as np

data_conf_dict = {
    "data_dir": "/root/home/workspace/torch-ash/data/dtu_scan24/preprocessed",
    "render_cameras_name": "cameras_sphere.npz",
    "object_cameras_name": "cameras_sphere.npz",
    "camera_outside_sphere": True,
    "scale_mat_scale": 1.1,
}
# construct conf from dict
data_conf = ConfigFactory.from_dict(data_conf_dict)
dataset = Dataset(data_conf)

pcd_path = "/root/home/workspace/torch-ash/data/dtu_scan24/scaled_points.ply"
pcd = trimesh.load(pcd_path)

cam_origins = dataset.pose_all[:, :3, 3].reshape(-1, 3)
vertices = torch.from_numpy(pcd.vertices).float().to(cam_origins.device).reshape(-1, 3)
vertices = torch.cat([vertices, cam_origins], dim=0)
pcd = trimesh.PointCloud(vertices.cpu().numpy())
pcd.export("cam_origins_and_scaled-pcd.ply")
