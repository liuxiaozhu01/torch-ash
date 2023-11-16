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

pcd_path = "/root/home/workspace/torch-ash/data/dtu_scan24/sparse_points.ply"
scale_mat = np.linalg.inv(dataset.scale_mats_np[0])

image_idx = torch.Tensor([0, 1, 2, 3]).long()
batch_size = 4
data = dataset.gen_random_rays_at(image_idx[0], batch_size=batch_size)
rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
# 100 points in each ray
dist = torch.linspace(0, 2, 100).reshape(1, -1).repeat(batch_size, 1).to(rays_o.device)
ray_samples = rays_o[:, None, :] + rays_d[:, None, :] * dist[..., None]
# print(ray_samples)
ray_samples = ray_samples.reshape(-1, 3)

# read pcd
# combine pcd and ray_samples, save pcd colors and set ray_samples colors to 0
pcd = trimesh.load(pcd_path)
vertices = torch.from_numpy(pcd.vertices).float().to(rays_o.device).reshape(-1, 3)
# convert vertices to homogeneous coordinates
vertices = torch.cat([vertices, torch.ones((vertices.shape[0], 1)).to(vertices.device)], dim=1)
# scale vertices by scale_mat
vertices = vertices @ torch.from_numpy(scale_mat).float().to(vertices.device).t()
# convert vertices back to euclidean coordinates
vertices = vertices[:, :3] / vertices[:, 3:]

vertices = torch.cat([vertices, ray_samples], dim=0)
colors = torch.from_numpy(pcd.colors).float().to(rays_o.device)
colors = torch.cat([colors, torch.zeros((ray_samples.shape[0], colors.shape[-1])).to(colors.device)], dim=0)
# save pcd
pcd = trimesh.PointCloud(vertices.cpu().numpy(), colors.cpu().numpy())
pcd.export("rays_and_pcd.ply")