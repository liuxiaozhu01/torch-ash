import torch
import torch.nn as nn
import torch.nn.functional as F

from ash import UnBoundedSparseDenseGrid
import numpy as np
from tqdm import tqdm
import nerfacc
import trimesh

import open3d as o3d
from torch.utils.tensorboard import SummaryWriter
import math

def get_activation(name):
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == 'none':
        return lambda x: x
    elif name.startswith('scale'):
        scale_factor = float(name[5:])
        return lambda x: x.clamp(0., scale_factor) / scale_factor
    elif name.startswith('clamp'):
        clamp_max = float(name[5:])
        return lambda x: x.clamp(0., clamp_max)
    elif name.startswith('mul'):
        mul_factor = float(name[3:])
        return lambda x: x * mul_factor
    elif name == 'lin2srgb':
        return lambda x: torch.where(x > 0.0031308, torch.pow(torch.clamp(x, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*x).clamp(0., 1.)
    elif name.startswith('+') or name.startswith('-'):
        return lambda x: x + float(name)
    elif name == 'sigmoid':
        return lambda x: torch.sigmoid(x)
    elif name == 'tanh':
        return lambda x: torch.tanh(x)
    else:
        return getattr(F, name)

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

class VanillaMLP(nn.Module):
    def __init__(self, dim_in, dim_out, config):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = config['n_neurons'], config['n_hidden_layers']
        self.sphere_init, self.weight_norm = config.get('sphere_init', False), config.get('weight_norm', False)
        self.sphere_init_radius = config.get('sphere_init_radius', 0.5)
        self.layers = [self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False), self.make_activation()]
        for i in range(self.n_hidden_layers - 1):
            self.layers += [self.make_linear(self.n_neurons, self.n_neurons, is_first=False, is_last=False), self.make_activation()]
        self.layers += [self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)]
        self.layers = nn.Sequential(*self.layers)
        self.output_activation = get_activation(config['output_activation'])
    
    @torch.cuda.amp.autocast(False)
    def forward(self, x):
        x = self.layers(x.float())
        x = self.output_activation(x)
        return x
    
    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=True) # network without bias will degrade quality
        if self.sphere_init:
            if is_last:
                torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                torch.nn.init.normal_(layer.weight, mean=math.sqrt(math.pi) / math.sqrt(dim_in), std=0.0001)
            elif is_first:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                torch.nn.init.normal_(layer.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(dim_out))
            else:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.normal_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(dim_out))
        else:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        
        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        return layer   

    def make_activation(self):
        if self.sphere_init:
            return nn.Softplus(beta=100)
        else:
            return nn.ReLU(inplace=True)

class ImplicitNetwork(nn.Module):
    def __init__(
        self,
        voxel_size,
        device,
        scaled_pcd_path,
        sdf_config,
        color_config,
        dilation=1,
    ):
        super().__init__()
        print(f"voxel_size={voxel_size}")
        self.grid = UnBoundedSparseDenseGrid(
            in_dim=3,
            num_embeddings=40000,
            embedding_dim=16,
            grid_dim=8,
            cell_size=voxel_size,
            device=device,
        )
        self.init_field(scaled_pcd_path, dilation=dilation)

        self.sdf = VanillaMLP(self.grid.embedding_dim, 16, sdf_config).to(device)
        self.color = VanillaMLP(16-1 + 3, 3, color_config).to(device)

        self.sdf_to_sigma = SDFToDensity(
            min_beta=voxel_size,
            init_beta=voxel_size*2,
        ).to(device)

        
    @torch.no_grad()
    def init_field(self, scaled_pcd_path, dilation=1):
        self.pcd_path = scaled_pcd_path
        pcd = trimesh.load(scaled_pcd_path)
        pcd_tensor = torch.from_numpy(pcd.vertices).float().to(self.grid.device).reshape(-1, 3).contiguous()

        grid_coords, cell_coords, grid_indices, cell_indices = self.grid.spatial_init_(pcd_tensor, dilation=dilation, bidirectional=True)
        nn.init.uniform_(self.grid.embeddings, -1, 1)

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

        masks = masks.view(-1, 1)

        out = self.sdf(embeddings)
        sdfs = out[..., 0:1].contiguous().view(-1, 1)
        feat = out[..., 1:]
        sdf_grads = torch.autograd.grad(
            outputs=sdfs, 
            inputs=x, 
            grad_outputs=torch.ones_like(sdfs), 
            create_graph=True
        )[0]
        normals = F.normalize(sdf_grads, dim=-1)

        rgbs = self.color(torch.cat([feat, normals], dim=-1)).contiguous().view(-1, 3)
        # print(min(rgbs), max(rgbs))
        # test the data range of rgbs
        # print(rgbs.min(), rgbs.max())
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
    
    def color_fn(self, x):
        embeddings, masks = self.grid(x, interpolation="linear")
        out = self.sdf(embeddings)
        sdfs = out[..., 0:1].contiguous().view(-1, 1)
        feat = out[..., 1:]
        sdf_grads = torch.autograd.grad(
            outputs=sdfs, 
            inputs=x, 
            grad_outputs=torch.ones_like(sdfs), 
            create_graph=True
        )[0]
        normals = F.normalize(sdf_grads, dim=-1)
        rgbs = self.color(torch.cat([feat, normals], dim=-1)).contiguous().view(-1, 3)
        return rgbs
    
    def grad_fn(self, x):
        x.requires_grad_(True)
        embeddings, masks = self.grid(x, interpolation="linear")

        out = self.sdf(embeddings)
        sdfs = out[..., 0:1].contiguous().view(-1, 1)
        # feat = out[..., 1:]
        grad_x = torch.autograd.grad(
            outputs=sdfs, 
            inputs=x, 
            grad_outputs=torch.ones_like(sdfs), 
            create_graph=True
        )[0]
        return grad_x * masks.float().view(-1, 1)
    
    def normal_fn(self, x):
        return F.normalize(self.grad_fn(x), dim=-1).contiguous()
    

if __name__ == "__main__":
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
        "output_activation": "none",
        "n_neurons": 64,
        "n_hidden_layers": 1,
    }
    # convert dict to OmegaConf
    sdf_config = OmegaConf.create(sdf_config)
    color_config = OmegaConf.create(color_config)

    device = torch.device("cuda:0")
    model = ImplicitNetwork(
        voxel_size=0.01,
        device=device,
        scaled_pcd_path="/root/home/workspace/torch-ash/data/dtu_scan24/preprocessed/scaled_points.ply",
        sdf_config=sdf_config,
        color_config=color_config,
    )

