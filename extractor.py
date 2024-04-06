import torch
import mrcfile
import numpy as np
from tqdm import tqdm
from math import ceil
import _gaussian_density_field as _C

__all__ = [
    "extract_densify_field_cuda", 
    "extract_density_field_native", 
    "export_mrc_file"
]

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_scaling_rotation_inv(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = 1. / (s[:,0] + 1E-8)
    L[:,1,1] = 1. / (s[:,1] + 1E-8)
    L[:,2,2] = 1. / (s[:,2] + 1E-8)

    L = R @ L
    return L

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def recover_symmetric(uncertainty):
    sym = torch.zeros((uncertainty.shape[0], 3, 3), dtype=torch.float, device="cuda")
    sym[:, 0, 0] = uncertainty[:, 0]
    sym[:, 0, 1] = sym[:, 1, 0] = uncertainty[:, 1]
    sym[:, 0, 2] = sym[:, 2, 0] = uncertainty[:, 2]
    sym[:, 1, 1] = uncertainty[:, 3]
    sym[:, 1, 2] = sym[:, 2, 1] = uncertainty[:, 4]
    sym[:, 2, 2] = uncertainty[:, 5]
    return sym

def build_covariance_from_scaling_rotation(scaling, rotation):
    L = build_scaling_rotation(scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm

def build_covariance_inv_from_scaling_rotation(scaling, rotation):
    L = build_scaling_rotation_inv(scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm

class DensityExtractor(object):
    @torch.no_grad()
    def __init__(self, 
        xyzs,           # (N, 3)
        opacities,      # (N, 1)
        covariances,    # (N, 6)
        covariances_inv # (N, 6)
    ) -> None:
        super().__init__()
        coefficients = torch.nan_to_num(torch.rsqrt(torch.linalg.det(recover_symmetric(covariances)).clamp_min(0.) + 1E-8).reshape(-1, 1)) # (N, 1)
        radius = 3.0 * torch.sqrt(torch.real(torch.linalg.eig(recover_symmetric(covariances)).eigenvalues).max(dim=-1).values).reshape(-1)
        front_coordinate = xyzs[:, 0] - radius
        back_coordinate = xyzs[:, 0] + radius

        indices = torch.sort(back_coordinate).indices.squeeze(-1)

        self.xyzs = xyzs[indices]
        self.opacities = opacities[indices]
        self.covariances = covariances[indices]
        self.covariances_inv = covariances_inv[indices]
        self.front_coordinate = front_coordinate[indices]
        self.back_coordinate = back_coordinate[indices]
        self.coefficients = coefficients[indices]
    def __call__(self, coordinates, batch_size: int = 10000):
        sigmas_s = []
        for batch_idx in tqdm(range(ceil(len(coordinates) / batch_size))):
            start_idx = batch_idx * batch_size
            stop_idx = (batch_idx + 1) * batch_size
            sorted_sigmas = _C.estimate_density(
                coordinates[start_idx:stop_idx].to(self.xyzs.device), 
                self.xyzs, 
                self.front_coordinate, 
                self.back_coordinate, 
                self.opacities, 
                self.coefficients, 
                self.covariances_inv, 
            )
            sigmas_s.append(sorted_sigmas.cpu())
        return torch.cat(sigmas_s)

def _extract_density_field(
    gaussians, 
    resolution: int, 
    max_batch_size: int, 
    deterministic=True, 
    cuda_accelerate=False
):
    # Centralize the Gaussians to estimate the length of grid
    center = gaussians.get_xyz.mean(dim=0, keepdim=True)
    min_coordinate = (gaussians.get_xyz - center).min().item()
    max_coordinate = (gaussians.get_xyz - center).max().item()
    print(f"Extract geometry by constructing a cubic density field ({min_coordinate}, {max_coordinate}) with resolution {resolution} and center {center.cpu().numpy()}.")

    # Construct the coordinates to query the densities
    # Coordinates are stored in the CPU first, and moved into CUDA only when evaluating for avoiding OOM.
    sampled_coordinates_1D = torch.linspace(min_coordinate, max_coordinate, steps=resolution) # (M, )
    step = (max_coordinate - min_coordinate) / resolution
    X, Y, Z = torch.meshgrid(sampled_coordinates_1D, sampled_coordinates_1D, sampled_coordinates_1D, indexing='ij')
    assert len(X.shape) == 3 and len(Y.shape) == 3 and len(Z.shape) == 3
    coordinates = torch.stack((X, Y, Z), dim=-1).reshape(-1, 3) + center.cpu() # (M ** 3, 3)

    # Perturb the coordinates for sampling inside the voxel
    if not deterministic:
        coordinates = coordinates + torch.rand_like(coordinates) * step
    
    if cuda_accelerate:
        extractor = DensityExtractor(
            gaussians.get_xyz, 
            gaussians.get_opacity, 
            build_covariance_from_scaling_rotation(gaussians.get_scaling, gaussians._rotation), 
            build_covariance_inv_from_scaling_rotation(gaussians.get_scaling, gaussians._rotation)
        )
        return extractor(coordinates, max_batch_size).reshape(resolution, resolution, resolution)
    else:
        sigmas = torch.ones_like(coordinates[:, 0]) * (-1000.0) # (M ** 3)

        xyzs        = gaussians.get_xyz                              # (N, 3)
        opacities   = gaussians.get_opacity                          # (N, 1)
        covariances = recover_symmetric(build_covariance_from_scaling_rotation(gaussians.get_scaling, gaussians._rotation))  # (N, 3, 3)
        covariances_inv = recover_symmetric(build_covariance_inv_from_scaling_rotation(gaussians.get_scaling, gaussians._rotation))  # (N, 3, 3)
        rec_covariances_det = torch.rsqrt(torch.linalg.det(covariances[None, ...]))[..., None]

        for _ in tqdm(range(int(np.ceil(len(sigmas) / max_batch_size)))):
            start_idx = _ * max_batch_size
            stop_idx = min((_ + 1) * max_batch_size, len(sigmas))
            if start_idx == stop_idx: break

            batch_coordinates = coordinates[start_idx:stop_idx].to(xyzs.device) # (m, 3)

            delta = batch_coordinates[:, None, :] - xyzs[None, :, :] # (m, N, 3)
            # Evaluate Gaussian Densities
            batch_gaussian_sigmas = opacities[None, ...] * 0.0634 * rec_covariances_det * torch.exp(- 0.5 * (delta[:, :, None, :] @ covariances_inv[None, ...] @ delta[:, :, :, None])).squeeze(-1) # (m, N, 1)
            batch_gaussian_sigmas = batch_gaussian_sigmas.squeeze(-1) # (m, N)

            sigmas[start_idx:stop_idx] = batch_gaussian_sigmas.sum(dim=-1).cpu()
        return sigmas.reshape(resolution, resolution, resolution)

@torch.no_grad()
def extract_densify_field_cuda(gaussians, 
    resolution: int, 
    max_batch_size: int = 10000000, 
    deterministic: bool = True
):
    return _extract_density_field(gaussians, resolution, max_batch_size, deterministic, True)

@torch.no_grad()
def extract_density_field_native(gaussians, 
    resolution: int, 
    max_batch_size: int = 10, 
    deterministic: bool = True
):
    return _extract_density_field(gaussians, resolution, max_batch_size, deterministic, False)

@torch.no_grad()
def export_mrc_file(fname, gaussians, 
    resolution: int, 
    max_batch_size: int = 10000000, 
    samples_per_voxel: int = 8, 
    cuda_acceleration: bool = True, 
):
    fn = extract_densify_field_cuda if cuda_acceleration else extract_density_field_native
    assert samples_per_voxel > 0
    if samples_per_voxel == 1:
        sigmas = fn(gaussians, resolution, max_batch_size, True)
    else:
        sigmas = torch.stack([ fn(gaussians, resolution, max_batch_size, False) for _ in range(samples_per_voxel) ]).mean(dim=0)
    
    with mrcfile.new_mmap(fname, overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
        mrc.data[:] = np.flip(sigmas, 0)