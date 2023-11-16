import torch
import torch.nn as nn
import torch.nn.functional as F

from ash import UnBoundedSparseDenseGrid
import numpy as np
from tqdm import tqdm
import nerfacc
import trimesh
from pyhocon import ConfigFactory
import open3d as o3d

from torch.utils.tensorboard import SummaryWriter

from data_provider import Dataset
# from field import 

class SDFToDensity(nn.Module):
    def __init__(self, min_beta=0.01, init_beta=1):
        super(SDFToDensity, self).__init__()
        self.min_beta = min_beta
        self.beta = nn.Parameter(torch.Tensor([init_beta]))

    def forward(self, sdf):
        beta = self.min_beta + torch.abs(self.beta)

        alpha = 1 / beta
        # https://github.com/lioryariv/volsdf/blob/main/code/model/density.py
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))


class Field(nn.Module):
    def __init__(self, voxel_size, device):
        super().__init__()
        print(f"voxel_size={voxel_size}")
        self.grid = UnBoundedSparseDenseGrid(
            in_dim=3,
            num_embeddings=40000,
            embedding_dim=32,
            grid_dim=8,
            cell_size=voxel_size,
            device=device,
        )

        # sdf network
        self.sdf_net = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 16)
        )
        # initialize


        # color network
        self.color_net = nn.Sequential(
            nn.Linear(15+3, 128),   # 15: sdf_net feature, 3: normal
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )



        self.sdf_to_sigma = SDFToDensity(
            min_beta=voxel_size, init_beta=voxel_size * 2
        ).to(device)

    def parameters(self):
        return [
            {"params": self.grid.parameters()},
            {"params": self.sdf_to_sigma.parameters(), "lr": 1e-4},
        ]
    
    @torch.no_grad()
    def init_field(self, pcd_path, transform_mat, dilation=1):
        self.pcd_path = pcd_path
        pcd = trimesh.load(pcd_path)
        pcd_tensor = torch.from_numpy(pcd.vertices).float().to(self.grid.device).reshape(-1, 3)
        # convert vertices to homogeneous coordinates
        pcd_tensor = torch.cat([pcd_tensor, torch.ones((pcd_tensor.shape[0], 1)).to(pcd_tensor.device)], dim=1)
        # scale vertices by transform_mat
        pcd_tensor = pcd_tensor @ torch.from_numpy(transform_mat).float().to(pcd_tensor.device).t()
        # convert vertices back to euclidean coordinates
        pcd_tensor = pcd_tensor[:, :3] / pcd_tensor[:, 3:]
        # import trimesh
        trimesh.points.PointCloud(pcd_tensor.cpu().numpy()).export('pcd_test.ply')

        grid_coords, cell_coords, grid_indices, cell_indices = self.grid.spatial_init_(pcd_tensor, dilation=dilation)

    def forward(self, rays_o, rays_d, rays_d_norm, near, far, jitter=None):
        (rays_near, rays_far) = self.grid.ray_find_near_far(
            rays_o=rays_o,
            rays_d=rays_d,
            t_min=near,
            t_max=far,
            t_step=0.01,  # no use
        )

        (ray_indices, t_nears, t_fars, prefix_sum_ray_samples,) = self.grid.ray_sample(
            rays_o=rays_o,
            rays_d=rays_d,
            rays_near=rays_near,
            rays_far=rays_far,
            max_samples_per_ray=64,
        )
        if jitter is not None:
            t_nears += jitter[..., 0:1]
            t_fars += jitter[..., 1:2]

        t_nears = t_nears.view(-1, 1)
        t_fars = t_fars.view(-1, 1)
        t_mid = 0.5 * (t_nears + t_fars)
        x = rays_o[ray_indices] + t_mid * rays_d[ray_indices]

        x.requires_grad_(True)
        embeddings, masks = self.grid(x, interpolation="linear")

        # import ipdb; ipdb.set_trace()

        # Used for filtering out empty voxels

        # Could optimize a bit
        # embeddings = embeddings[masks]
        masks = masks.view(-1, 1)
        sdfs = embeddings[..., 0:1].contiguous().view(-1, 1)
        rgbs = embeddings[..., 1:4].contiguous().view(-1, 3)
        sdf_grads = torch.autograd.grad(
            outputs=sdfs,
            inputs=x,
            grad_outputs=torch.ones_like(sdfs, requires_grad=False),
            create_graph=True,
        )[0]

        normals = F.normalize(sdf_grads, dim=-1)
        # print(f'normals.shape={normals.shape}, {normals}')
        sigmas = self.sdf_to_sigma(sdfs) * masks.float()

        weights = nerfacc.render_weight_from_density(
            t_starts=t_nears,
            t_ends=t_fars,
            sigmas=sigmas,
            ray_indices=ray_indices,
            n_rays=len(rays_o),
        )

        # TODO: could use concatenated rendering in one pass and dispatch
        # TODO: also can reuse the packed info
        rendered_rgb = nerfacc.accumulate_along_rays(
            weights=weights,
            ray_indices=ray_indices,
            values=rgbs,
            n_rays=len(rays_o),
        )

        rendered_depth = nerfacc.accumulate_along_rays(
            weights=weights,
            ray_indices=ray_indices,
            values=t_mid,
            n_rays=len(rays_o),
        )
        rays_near = rays_near.view(-1, 1) / rays_d_norm
        rays_far = rays_far.view(-1, 1) / rays_d_norm
        rendered_depth = rendered_depth / rays_d_norm

        rendered_normals = nerfacc.accumulate_along_rays(
            weights=weights,
            ray_indices=ray_indices,
            values=normals,
            n_rays=len(rays_o),
        )

        accumulated_weights = nerfacc.accumulate_along_rays(
            weights=weights,
            ray_indices=ray_indices,
            values=None,
            n_rays=len(rays_o),
        )
        # print(f'rendered_normals.shape={rendered_normals.shape}: {rendered_normals}')

        return {
            "rgb": rendered_rgb,
            "depth": rendered_depth,
            "normal": rendered_normals,
            "weights": accumulated_weights,
            "sdf_grads": sdf_grads,
            "near": rays_near,
            "far": rays_far,
        }
    
    def marching_cubes(self):
        sdf = self.grid.embeddings[..., 0].contiguous()
        weight = self.grid.embeddings[..., 4].contiguous()
        mesh = self.grid.marching_cubes(
            sdf,
            weight,
            vertices_only=False,
            color_fn=self.color_fn,
            normal_fn=self.normal_fn,
            iso_value=0.0,
            weight_thr=1,
        )
        return mesh
    
    def occupancy_lineset(self, color=[0, 0, 1], scale=1.0):
        xyz000, _, _, _ = self.grid.items()
        xyz000 = xyz000.view(-1, 3)

        block_len = self.grid.cell_size * self.grid.grid_dim

        xyz000 = xyz000.cpu().numpy() * block_len
        xyz001 = xyz000 + np.array([[block_len * scale, 0, 0]])
        xyz010 = xyz000 + np.array([[0, block_len * scale, 0]])
        xyz100 = xyz000 + np.array([[0, 0, block_len * scale]])
        xyz = np.concatenate((xyz000, xyz001, xyz010, xyz100), axis=0).astype(
            np.float32
        )

        lineset = o3d.t.geometry.LineSet()
        lineset.point["positions"] = o3d.core.Tensor(xyz)

        n = len(xyz000)
        lineset000 = np.arange(0, n)
        lineset001 = np.arange(n, 2 * n)
        lineset010 = np.arange(2 * n, 3 * n)
        lineset100 = np.arange(3 * n, 4 * n)

        indices001 = np.stack((lineset000, lineset001), axis=1)
        indices010 = np.stack((lineset000, lineset010), axis=1)
        indices100 = np.stack((lineset000, lineset100), axis=1)
        indices = np.concatenate((indices001, indices010, indices100), axis=0)

        lineset.line["indices"] = o3d.core.Tensor(indices.astype(np.int32))
        colors = np.tile(color, (3 * n, 1))
        lineset.line["colors"] = o3d.core.Tensor(colors.astype(np.float32))
        return lineset

    def color_fn(self, x):
        embeddings, masks = self.grid(x, interpolation="linear")
        return embeddings[..., 1:4].contiguous()

    def weight_fn(self, x):
        embeddings, masks = self.grid(x, interpolation="linear")
        return embeddings[..., 4:5].contiguous()

    def grad_fn(self, x):
        x.requires_grad_(True)
        embeddings, masks = self.grid(x, interpolation="linear")

        grad_x = torch.autograd.grad(
            outputs=embeddings[..., 0],
            inputs=x,
            grad_outputs=torch.ones_like(embeddings[..., 0], requires_grad=False),
            create_graph=True,
        )[0]
        return grad_x * masks.float().view(-1, 1)

    def normal_fn(self, x):
        return F.normalize(self.grad_fn(x), dim=-1).contiguous()
    


if __name__ == "__main__":
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--voxel_size", type=float, default=0.015)
    parser.add_argument("--depth_type", type=str, default="sensor", choices=["sensor", "learned"])
    parser.add_argument("--depth_max", type=float, default=5.0, help="max depth value to truncate in meters")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--train_batch_size", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    args = parser.parse_args()
    # fmt: on

    import datetime

    path = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(path)

    # build dataset config befor load data
    dataset_conf_dict = {
        "data_dir": args.path,  # "/root/home/workspace/torch-ash/data/dtu_scan24/preprocessed",
        "render_cameras_name": "cameras_sphere.npz",
        "object_cameras_name": "cameras_sphere.npz",
        "camera_outside_sphere": True,
        "scale_mat_scale": 1.1,
    }
    # construct conf from dict
    data_conf = ConfigFactory.from_dict(dataset_conf_dict)
    dataset = Dataset(data_conf)
    # pcd and transform matrix for init field
    pcd_path = args.path + "/../sparse_points.ply" # "/root/home/workspace/torch-ash/data/dtu_scan24/sparse_points.ply"
    transform_mat = np.linalg.inv(dataset.scale_mats_np[0])

    model = Field(args.voxel_size, device=torch.device("cuda:0"))
    dilation = 2
    model.init_field(pcd_path, transform_mat, dilation=dilation)
    model.grid.gaussian_filter_(7, 1)   # ?
    # mesh = model.marching_cubes()

    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size

    pixel_count = dataset.H * dataset.W
    eval_batches_per_image = pixel_count // eval_batch_size
    # assert pixel_count % eval_batch_size == 0

    # eval_dataloader = Dataloader(
    #     dataset,
    #     batch_size=eval_batch_size,
    #     shuffle=False,
    #     device=torch.device("cuda:0"),
    # )

    # train_dataloader = Dataloader(
    #     dataset, batch_size=train_batch_size, shuffle=True, device=torch.device("cuda:0")
    # )

    optim = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.9)

    # Training
    # depth_loss_fn = ScaleAndShiftInvariantLoss()
    pbar = tqdm(range(args.steps + 1))
    image_perm = torch.randperm(dataset.n_images)

    jitter = None

    def reset_jitter_():
        jitter = (
            (torch.rand(train_batch_size, 2, device=torch.device("cuda:0")) * 2 - 1)
            * 0.5
            * args.voxel_size
        )

    for step in pbar:
        data = dataset.gen_random_rays_at(image_perm[step % len(image_perm)], train_batch_size)

        rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
        # near, far = dataset.near_far_from_sphere(rays_o, rays_d)
        result = model(rays_o, rays_d, torch.norm(rays_d, dim=-1, keepdim=True), near=0.1, far=5.0, jitter=jitter)

        rgb_loss = F.mse_loss(result["rgb"], true_rgb)
        eikonal_loss_ray = (torch.norm(result["sdf_grads"], dim=-1) - 1).abs().mean()

        uniform_samples = model.grid.uniform_sample(train_batch_size)
        uniform_sdf_grads = model.grad_fn(uniform_samples)
        eikonal_loss_uniform = (torch.norm(uniform_sdf_grads, dim=-1) - 1).abs().mean()

        loss = (
            rgb_loss
            + 0.1 * eikonal_loss_ray 
            + 0.1 * eikonal_loss_uniform
        )

        loss.backward()
        pbar.set_description(
            f"loss: {loss.item():.4f},"
            f"rgb: {rgb_loss.item():.4f},"
            f"eikonal_ray: {eikonal_loss_ray.item():.4f}",
            f"eikonal_uniform: {eikonal_loss_uniform.item():.4f}",
        )
        optim.step()
        writer.add_scalar("loss/total", loss.item(), step)
        writer.add_scalar("loss/rgb", rgb_loss.item(), step)
        writer.add_scalar("loss/eikonal_ray", eikonal_loss_ray.item(), step)
        writer.add_scalar("loss/eikonal_uniform", eikonal_loss_uniform.item(), step)
        
        if step % 100 == 0:
            # mesh = model.marching_cubes()
            # o3d.io.write_triangle_mesh(f"{path}/mesh_{step}.ply", mesh.to_legacy())

            # validate image
            idx = np.random.randint(dataset.n_images)
            rays_o, rays_d = dataset.gen_rays_at(idx, resolution_level=1)
            H, W, _ = rays_o.shape
            rays_o = rays_o.reshape(-1, 3).split(eval_batch_size)
            rays_d = rays_d.reshape(-1, 3).split(eval_batch_size)

            im_rgbs = []
            im_weights = []
            im_depths = []
            im_normals = []
            im_near = []
            im_far = []

            for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
                result = model(
                    rays_o=rays_o_batch,
                    rays_d=rays_d_batch,
                    rays_d_norm=torch.norm(rays_d_batch, dim=-1, keepdim=True),
                    near=0.1,
                    far=5.0,
                )
                im_rgbs.append(result["rgb"].detach().cpu().numpy())
                im_weights.append(result["weights"].detach().cpu().numpy())
                im_depths.append(result["depth"].detach().cpu().numpy())
                im_normals.append(result["normal"].detach().cpu().numpy())
                im_near.append(result["near"].detach().cpu().numpy())
                im_far.append(result["far"].detach().cpu().numpy())
            
            im_rgbs = np.concatenate(im_rgbs, axis=0).reshape(dataset.H, dataset.W, 3)
            im_weights = np.concatenate(im_weights, axis=0).reshape(
                dataset.H, dataset.W, 1
            )
            im_depths = np.concatenate(im_depths, axis=0).reshape(
                dataset.H, dataset.W, 1
            )
            im_normals = (
                np.concatenate(im_normals, axis=0).reshape(dataset.H, dataset.W, 3)
                + 1.0
            ) * 0.5
            im_near = np.concatenate(im_near, axis=0).reshape(dataset.H, dataset.W, 1)
            im_far = np.concatenate(im_far, axis=0).reshape(dataset.H, dataset.W, 1)

            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3)
            axes[0, 0].imshow(im_rgbs)
            axes[0, 1].imshow(im_weights)
            axes[0, 2].imshow(im_near)
            axes[1, 0].imshow((im_depths))
            axes[1, 1].imshow((im_normals))
            axes[1, 2].imshow((im_far))
            for i in range(2):
                for j in range(3):
                    axes[i, j].set_axis_off()
            fig.tight_layout()
            writer.add_figure("eval", fig, step)
            plt.close(fig)

            if step > 0:
                scheduler.step()
                reset_jitter_()