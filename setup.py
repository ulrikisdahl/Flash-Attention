from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flash_attention_extension',
    ext_modules=[
        CUDAExtension(
            name='flash_attention_extension',
            sources=['src/main.cpp', 'src/flash_attention_forward.cu', 'src/flash_attention_backward.cu'],
            include_dirs=['/ulrik/home/libtorch/include', '/ulrik/home/libtorch/include/torch/csrc/api/include', "/usr/local/cuda/include"],
            library_dirs=['/ulrik/home/libtorch/lib'],
            libraries=['c10', 'torch', 'torch_cpu', 'torch_cuda']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)