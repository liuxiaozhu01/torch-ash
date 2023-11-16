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
from field import ImplicitNetwork

from omegaconf import OmegaConf

sdf_config = {
    "output_activation": "none",
    "n_neurons": 64,
    "n_hidden_layers": 1,
    "sphere_init": True,
    "sphere_init_radius": 0.0,
    "weight_norm": True,
}
color_config = {
    "output_activation": "sigmoid",
    "n_neurons": 64,
    "n_hidden_layers": 1,
}
# convert dict to OmegaConf
sdf_config = OmegaConf.create(sdf_config)
color_config = OmegaConf.create(color_config)

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

    device = torch.device("cuda:0")
    model = ImplicitNetwork(
        voxel_size=args.voxel_size,
        device=device,
        scaled_pcd_path="/root/home/workspace/torch-ash/data/dtu_scan24/preprocessed/scaled_points.ply",
        sdf_config=sdf_config,
        color_config=color_config,
        dilation=2,
    )
    # dilation = 2
    # model.init_field(pcd_path, transform_mat, dilation=dilation)
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