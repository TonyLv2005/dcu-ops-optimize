from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# Paths relative to this script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The dispatch source files must be compiled with -DWARP_LIB to exclude their
# stand-alone main() entry points.
common_flags = ['-DWARP_LIB', '-O3', '--offload-arch=gfx936']

ext_module = CUDAExtension(
    name='fp32gemm',
    sources=[
        'warp_gemm.cu',
        os.path.join(ROOT, 'kernels/gemm_dispatch.cu'),
        os.path.join(ROOT, 'kernels/gemm_ABT_dispatch.cu'),
    ],
    extra_compile_args={
        'cxx': common_flags,
        'nvcc': common_flags,
    },
)

setup(
    name='fp32gemm',
    version='1.0.0',
    description='BF16×FP32→FP32 GEMM dispatch for DCU gfx936',
    ext_modules=[ext_module],
    cmdclass={'build_ext': BuildExtension},
)
