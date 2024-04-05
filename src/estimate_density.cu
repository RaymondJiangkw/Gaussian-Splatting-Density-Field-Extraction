#include <torch/torch.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#define BLOCK_SIZE 256

__global__ void estimateDensityCUDA(
    const int N, const int M, 
    const float* __restrict__ coordinates, 
    const float* __restrict__ means3D, 
    const float* front_coordinate, 
    const float* back_coordinate, 
    const float* opacities, 
    const float* coefficients, 
    const float* cov3D_invs, 
    float* out_density
) {
    const int coordinate_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( coordinate_idx >= M ) return;
    glm::vec3 c = {
        coordinates[3 * coordinate_idx + 0], 
        coordinates[3 * coordinate_idx + 1], 
        coordinates[3 * coordinate_idx + 2]
    };
    float sigma = 0.0f;
    for ( int kernel_idx = 0; kernel_idx < N; ++kernel_idx ) {
        if ( back_coordinate[kernel_idx] < c.x ) continue;
        if ( front_coordinate[kernel_idx] > c.x ) continue;

        float opacity = opacities[kernel_idx];
        float det = coefficients[kernel_idx];
        glm::mat3 cov3D_inv = {
            cov3D_invs[6 * kernel_idx + 0], 
            cov3D_invs[6 * kernel_idx + 1], 
            cov3D_invs[6 * kernel_idx + 2], 
            cov3D_invs[6 * kernel_idx + 1], 
            cov3D_invs[6 * kernel_idx + 3], 
            cov3D_invs[6 * kernel_idx + 4], 
            cov3D_invs[6 * kernel_idx + 2], 
            cov3D_invs[6 * kernel_idx + 4], 
            cov3D_invs[6 * kernel_idx + 5]
        };

        glm::vec3 kernel_c = {
            means3D[3 * kernel_idx + 0], 
            means3D[3 * kernel_idx + 1], 
            means3D[3 * kernel_idx + 2]
        };
        auto delta = c - kernel_c;
        // Gaussian Distribution
        float G = min(-0.5f * glm::dot(delta, cov3D_inv * delta), 0.0f);
        float exponend = max(glm::dot(delta, cov3D_inv * delta), 0.0f);
        float gaussian_sigma = 0.0634f * opacity * det * exp(-0.5f * exponend);

        sigma += gaussian_sigma;
    }
    out_density[coordinate_idx] = sigma;
}

void estimateDensity(
    const int N, const int M, 
    const float* coordinates, 
    const float* means3D, 
    const float* front_coordinate, 
    const float* back_coordinate, 
    const float* opacities, 
    const float* coefficients, 
    const float* cov3D_invs, 
    float* out_density
) {
    estimateDensityCUDA <<< (M + BLOCK_SIZE) / BLOCK_SIZE, BLOCK_SIZE >>> (
        N, M, 
        coordinates, 
        means3D, 
        front_coordinate, 
        back_coordinate, 
        opacities, 
        coefficients, 
        cov3D_invs, 
        out_density
    );
}

torch::Tensor EstimateDensityCUDA(
    const torch::Tensor& coordinates, 
    const torch::Tensor& means3D, 
    const torch::Tensor& front_coordinate, 
    const torch::Tensor& back_coordinate, 
    const torch::Tensor& opacities, 
    const torch::Tensor& coefficients, 
    const torch::Tensor& cov3D_invs
) {
    const int M = coordinates.size(0);
    const int N = means3D.size(0);
    auto float_opts = means3D.options().dtype(torch::kFloat32);
    torch::Tensor out_density = torch::full({M}, -1000.0f, float_opts);
    estimateDensity(
        N, M, 
        coordinates.contiguous().data<float>(), 
        means3D.contiguous().data<float>(), 
        front_coordinate.contiguous().data<float>(), 
        back_coordinate.contiguous().data<float>(), 
        opacities.contiguous().data<float>(), 
        coefficients.contiguous().data<float>(), 
        cov3D_invs.contiguous().data<float>(), 
        out_density.contiguous().data<float>()
    );
    return out_density;
}