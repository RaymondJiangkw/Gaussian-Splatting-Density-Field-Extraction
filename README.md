# Density Field Extractor for Gaussian Splatting

## Overview

Here is an open-source CUDA-accelerated algorithm for extracting the density field from a Gaussian splatting model. It is intended for helping extract the geometry (mesh etc.) using marching cube algorithm (or something tailored for the volume representation).

## Set Up

The environment is the same with that in [3DGS](https://github.com/graphdeco-inria/gaussian-splatting/tree/main) repository, except that you also need to install `mrcfile` if you want to visualize the density field using [UCSF ChimeraX](https://www.cgl.ucsf.edu/chimerax/). A small guide for using UCSF ChimeraX can be found [here](https://github.com/NVlabs/eg3d/tree/main?tab=readme-ov-file#generating-media), except that you probably need to adjust the level set to get the best geometry.

Besides, you need to install the cuda extension for acceleration by
```bash
$ python setup.py install
```

The code is tested on Ubuntu 22.04 with a NVIDIA RTX 3080. Other platforms and hardwares should be fine.

## Usage

A code snippet for extracting the density field or directly exporting the `.mrc` file is demonstrated here:
```python
# Loading or Training the model.
# ...
sigmas_cuda = extract_densify_field_cuda(gaussians, 512, max_batch_size=10000000) # (512, 512, 512) Density field extracted with CUDA
sigmas_native = extract_density_field_native(gaussians, 512, max_batch_size=10) # (512, 512, 512) Density field extracted with native PyTorch
export_mrc_file("geometry.mrc", gaussians, 512, samples_per_voxel=8) # (512, 512, 512). Typically, sampling more points (8 in this case) in the voxel (and then take the average) produce better geometry.
```

## Performance
### Time Complexity
A `(512, 512, 512)` evaluation with $\sim10^5$ Gaussians takes $\sim1$ minute with the CUDA implementation, and takes $\sim1$ day with the native PyTorch implementation.

### Qualitative Results
Extracted Mesh (level set = $15$) for `Chair` in the NeRF-Synthetic Dataset with the official Gaussian Splatting (took $\sim8$ minutes on a NVIDIA RTX 3080 for $(512, 512, 512)$ with $8$ samples per voxel):
![chair](https://github.com/RaymondJiangkw/Gaussian-Splatting-Density-Field-Extraction/blob/main/demo/chair.png)

## Method Explained
For people who are interested, the scheme of algorithm is to first identify the boundary of the scene, and then divide the scene into `(resolution, resolution, resolution)` cubic grid. For each voxel, a point $x$ is sampled inside and therefore the density of voxel is given by $$\sum_{i=1}^{N}o_i\mathcal{G}_i(x).$$ In this case, we are using the density value at a single point to approximate the density distribution of the whole voxel, which can be sub-optimal. Therefore, a better choice is to put multiple different points inside the voxel, estimate their density values, and take the average as the density value of the voxel. However, the trade-off is that it will elongate the time consumption.

The cuda implementation incorporates a small optimization for speed up. It estimates a valid range for each Gaussian kernel along an axis (the choice of axis doesn't matter. I choose `x`-axis in the implementation), which is similar to truncate the Gaussian into finite-support kernel in the 3DGS paper. Therefore, during the evaluation, for the queried point, if it does not fall into the valid range of a Gaussian kernel, the kernel can be directly skipped. This could result in slight differences with the estimated densities by the native PyTorch. The algorithm further sorts the Gaussian kernels based on the larger value of the valid range of the Gaussian kernel, so the evaluation for a point starts at the first Gaussian kernel, whose larger value of valid range is larger than the `x`-coordinate of the point.

A even more efficient way is to incorporate some data structures, e.g., BVH, to further speed up the evaluation, which is left for future. :)

## Acknowledgement
Credits to `3D Gaussian Splatting for Real-Time Radiance Field Rendering`.
<div class="container is-max-desktop content">
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
</div>