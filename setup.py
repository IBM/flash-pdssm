import os

from setuptools import setup

ext_modules, cmdclass = [], {}
if os.environ.get("PDSSM_SKIP_CUDA_BUILD") != "1":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    src = "pdssm/csrc"
    ext_modules = [CUDAExtension(
        name="pdssm._C",
        sources=[f"{src}/bindings.cpp", f"{src}/scan_kernel_compact.cu", f"{src}/scan_kernel_grad_compact.cu"],
        include_dirs=[os.path.abspath(src)],
        extra_compile_args={
            "cxx": ["-O3", "-DNDEBUG"],
            "nvcc": ["-O3", "--use_fast_math", "-U__CUDA_NO_HALF_OPERATORS__",
                     "-U__CUDA_NO_HALF_CONVERSIONS__", "--expt-relaxed-constexpr", "--expt-extended-lambda"],
        },
    )]
    cmdclass = {"build_ext": BuildExtension}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
