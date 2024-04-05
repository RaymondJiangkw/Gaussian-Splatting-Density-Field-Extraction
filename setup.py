from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

setup(
    name="gaussian_density_field",
    ext_modules=[
        CUDAExtension(
            name="_gaussian_density_field",
            sources=[
            "src/estimate_density.cu",
            "src/ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)