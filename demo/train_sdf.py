from typing import Tuple

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch.utils.dlpack import from_dlpack, to_dlpack

from ash import (
    UnBoundedSparseDenseGrid,
    BoundedSparseDenseGrid,
    DotDict,
    BoundedMultiResGrid,
    SirenNet
)
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

import open3d as o3d
import open3d.core as o3c

from mesh_util import create_mesh

from data_provider import PointCloudDataset, Dataloader

np.random.seed(15213)
torch.manual_seed(15213)

class NeuralSDF(nn.Module):
    def __init__(
        self,
        embedding_dim,
        resolution=16,
        encoder="ash",
        num_layers=2,
        hidden_dim=128,
        device=torch.device("cuda:0"),
    ):
        super().__init__()

        self.encoder_type = encoder

        if encoder == "ash":
            sparse_grid_dim = resolution // 4
            grid_dim = 4
            self.encoder = BoundedSparseDenseGrid(
                in_dim=3,
                num_embeddings=sparse_grid_dim ** 3,
                embedding_dim=embedding_dim,
                grid_dim=grid_dim,
                sparse_grid_dim=sparse_grid_dim,
                bbox_min=torch.zeros(3, dtype=torch.float32, device=device),
                bbox_max=torch.ones(3, dtype=torch.float32, device=device),
                device=device,
            )
            nn.init.uniform_(self.encoder.embeddings, -1e-4, 1e-4)

            # Active all entries
            self.encoder.dense_init_()
            print(self.encoder.engine.size())

        elif encoder == "ngp":
            import tinycudann as tcnn

            self.encoder = tcnn.Encoding(
                3,
                {
                    "otype": "DenseGrid",
                    "n_levels": 1,
                    "n_features_per_level": embedding_dim,
                    "log2_hashmap_size": 19,
                    "base_resolution": resolution,
                    "per_level_scale": 2,
                    "interpolation": "Linear",
                },
                dtype=torch.float32,
            ).to(device)
            # print(self.encoder.params.min(), self.encoder.params.max(), self.encoder.params.mean(), self.encoder.params.std(), self.encoder.params.median())
            nn.init.uniform_(self.encoder.params, -1e-4, 1e-4)

            print(self.encoder.native_tcnn_module.param_precision())

        self.mlp = SirenNet(
            dim_in=embedding_dim + 3,
            dim_hidden=hidden_dim,
            dim_out=1,
            num_layers=num_layers,
            w0=30.0,
        ).to(device)

    def sample(self, num_samples):
        eps = 1e-2
        sample_coords = np.random.uniform(-1.0 + eps, 1.0 - eps, size=(num_samples, 3))
        return torch.from_numpy(sample_coords).float().contiguous().cuda()

    @torch.no_grad()
    def visualize_occupied_cells(self):
        if self.encoder_type == "ash":
            grid_coords, cell_coords, grid_indices, cell_indices = self.encoder.items()
            cell_positions = self.encoder.cell_to_world(grid_coords, cell_coords)
            pcd = o3d.t.geometry.PointCloud(cell_positions.cpu().numpy())
            pcd.paint_uniform_color([0.0, 0.0, 1.0])
            return [pcd]
        elif self.encoder_type == "ngp":
            return []

    def forward(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positions.requires_grad_(True)

        positions = positions * 0.5 + 0.5
        if self.encoder_type == "ash":
            embedding, mask = self.encoder(positions, interpolation="linear")
            assert mask.all()
        elif self.encoder_type == "ngp":

            # print(positions.min(), positions.max())
            # print(positions)

            embedding = self.encoder(positions)
            # print(embedding.shape, embedding.dtype)

        sdf = self.mlp(torch.cat((embedding, positions), dim=-1))

        grad_x = torch.autograd.grad(
            outputs=sdf[..., 0],
            inputs=positions,
            grad_outputs=torch.ones_like(sdf[..., 0], requires_grad=False),
            create_graph=True,
        )[0]

        return sdf, grad_x

    def marching_cubes(self, fname):
        def sdf_fn(x):
            sdf, grad_x = self.forward(x)
            return sdf.detach()

        create_mesh(sdf_fn, fname, N=256, max_batch=64**3, offset=None, scale=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--path", type=str)
    parser.add_argument("--model", type=str, default="plain", choices=["ash", "ngp"])
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    # fmt: on
    args = parser.parse_args()

    logs = Path('logs') / args.model
    logs.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=logs, comment=f"{args.model}_{args.resolution}_{args.embedding_dim}")

    dataset = PointCloudDataset(args.path)
    dataloader = Dataloader(dataset, batch_size=25000, shuffle=True)

    bbox_lineset = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox([-1, -1, -1], [1, 1, 1])
    )

    model = NeuralSDF(
        embedding_dim=args.embedding_dim, resolution=args.resolution, encoder=args.model
    )

    occupied_cells = model.visualize_occupied_cells()
    # o3d.visualization.draw(occupied_cells + [dataset.pcd, bbox_lineset])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    pbar = tqdm(range(3001))
    for step in pbar:
        batch = next(iter(dataloader))

        optimizer.zero_grad()

        positions = batch["position"].cuda()
        normals = batch["normal"].cuda()

        sdf, grad_x = model(positions)

        loss_surface = sdf.abs().mean()
        loss_surface_normal = (1 - F.cosine_similarity(grad_x, normals, dim=-1)).mean()
        loss_surface_eikonal = (torch.norm(grad_x, dim=-1) - 1).abs().mean()

        rand_positions = model.sample(num_samples=int(len(positions)))
        sdf_rand, grad_x_rand = model(rand_positions)

        loss_rand_sdf = torch.exp(-1e2 * sdf_rand.abs()).mean()
        # loss_rand_sdf = torch.exp(-sdf_rand.abs()).mean()
        loss_rand_eikonal = (torch.norm(grad_x_rand, dim=-1) - 1).abs().mean()

        loss = (
            loss_surface * 3e3
            + loss_surface_normal * 1e3
            + loss_surface_eikonal * 5e1
            + loss_rand_sdf * 1e2
            + loss_rand_eikonal * 5e1
        )
        loss.backward()

        # import torchviz
        # torchviz.make_dot(
        #     (positions, rand_positions, grad_x, grad_x_rand, loss),
        #     params={"x": positions, "x_rand": rand_positions, "grad_x": grad_x, "grad_x_rand": grad_x_rand, "loss": loss},
        #     show_saved=False,
        # ).render("graph", format="png")
        # exit()


        pbar.set_description(
            f"Total={loss.item():.4f},"
            f"sdf={loss_surface.item():.4f},"
            f"normal={loss_surface_normal.item():.4f},"
            f"eikonal={loss_surface_eikonal.item():.4f},"
            f"rand_sdf={loss_rand_sdf.item():.4f},"
            f"rand_eikonal={loss_rand_eikonal.item():.4f}"
        )
        writer.add_scalar("loss/loss", loss.item(), step)
        writer.add_scalar("loss/loss_surface", loss_surface.item(), step)
        writer.add_scalar("loss/loss_surface_normal", loss_surface_normal.item(), step)
        writer.add_scalar("loss/loss_surface_eikonal", loss_surface_eikonal.item(), step)
        writer.add_scalar("loss/loss_rand_sdf", loss_rand_sdf.item(), step)
        writer.add_scalar("loss/loss_rand_eikonal", loss_rand_eikonal.item(), step)

        optimizer.step()

        if step % 1000 == 0 and step > 0:
            # scheduler.step()
            model.marching_cubes(f"mesh_{step:03d}_{args.model}.ply")
            mesh = o3d.io.read_triangle_mesh(f"mesh_{step:03d}_{args.model}.ply")
            mesh.compute_vertex_normals()
            # o3d.visualization.draw(occupied_cells + [mesh, dataset.pcd, bbox_lineset])
            #exit()