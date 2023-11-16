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
scale_mat = np.linalg.inv(dataset.scale_mats_np[0])

pcd_path = "/root/home/workspace/torch-ash/data/dtu_scan24/sparse_points.ply"

cam_origins = dataset.pose_all[:, :3, 3].reshape(-1, 3)

# combine pcd and cam_origins, save pcd colors and set cam_origins colors to 0
pcd = trimesh.load(pcd_path)
vertices = torch.from_numpy(pcd.vertices).float().to(cam_origins.device).reshape(-1, 3)
# convert vertices to homogeneous coordinates
vertices = torch.cat([vertices, torch.ones((vertices.shape[0], 1)).to(vertices.device)], dim=1)
# scale vertices by scale_mat
vertices = vertices @ torch.from_numpy(scale_mat).float().to(vertices.device).t()
# convert vertices back to euclidean coordinates
vertices = vertices[:, :3] / vertices[:, 3:]

vertices = torch.cat([vertices, cam_origins], dim=0)
colors = torch.from_numpy(pcd.colors).float().to(cam_origins.device)
colors = torch.cat([colors, torch.zeros((cam_origins.shape[0], colors.shape[-1])).to(colors.device)], dim=0)
# save pcd
pcd = trimesh.PointCloud(vertices.cpu().numpy(), colors.cpu().numpy())
pcd.export("cam_origins_and_pcd.ply")