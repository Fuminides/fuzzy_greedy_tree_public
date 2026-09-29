"""Build script for the ferl_fast compiled kernels.

Usage (from the repo root)::

    python ferl_fast/setup.py build_ext --inplace

We intentionally avoid -ffast-math: the split criterion compares candidate
scores and breaks ties, so we keep IEEE float semantics to stay numerically
close to the pure-Python reference.
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# OpenMP: GCC/Clang-with-libgomp use -fopenmp for both compile and link. The
# parallel kernels degrade gracefully to serial when called with num_threads=1,
# so a build without OpenMP only needs these flags removed (packaging TODO:
# detect OpenMP availability, e.g. macOS clang needs libomp).
_OPENMP = ["-fopenmp"]

extensions = [
    Extension(
        "ferl_fast._kernels",
        [os.path.join(HERE, "_kernels.pyx")],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3"] + _OPENMP,
        extra_link_args=_OPENMP,
    )
]

setup(
    name="ferl_fast",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
        build_dir=os.path.join(HERE, "_cython_build"),
    ),
    zip_safe=False,
)
