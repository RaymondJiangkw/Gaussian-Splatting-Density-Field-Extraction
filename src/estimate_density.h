#ifndef CUDA_EXTRACTOR_ESTIMATE_DENSITY_H_INCLUDED
#define CUDA_EXTRACTOR_ESTIMATE_DENSITY_H_INCLUDED

#include <cstdio>
#include <torch/torch.h>
#include <torch/extension.h>

torch::Tensor EstimateDensityCUDA(
    const torch::Tensor& coordinates, 
    const torch::Tensor& means3D, 
    const torch::Tensor& front_coordinate, 
    const torch::Tensor& back_coordinate, 
    const torch::Tensor& opacities, 
    const torch::Tensor& coefficients, 
    const torch::Tensor& cov3D_invs
);

#endif