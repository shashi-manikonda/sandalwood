# This file is used to build the C++ and C extensions for the mtflib library.
# It is required by setuptools, which is specified as the build backend in
# pyproject.toml. For more details on how setuptools handles extensions, see:
# https://setuptools.pypa.io/en/latest/userguide/ext_modules.html

import sys

import numpy
import pybind11
from setuptools import Extension, setup

# Set compiler arguments based on the operating system
if sys.platform == "win32":
    # MSVC compiler arguments
    cpp_args = ["/std:c++17", "/openmp"]
    link_args = []
else:
    # GCC/Clang compiler arguments
    cpp_args = ["-std=c++17", "-fopenmp", "-O3", "-march=native"]
    link_args = ["-fopenmp"]

extensions = [
    Extension(
        "mtflib.backends.cpp.mtf_cpp",
        [
            "src/mtflib/backends/cpp/mtf_data.cpp",
            "src/mtflib/backends/cpp/pybind_wrapper.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            numpy.get_include(),
            "src/mtflib/backends/cpp",
        ],
        language="c++",
        extra_compile_args=cpp_args,
        extra_link_args=link_args,
    ),
    Extension(
        "mtflib.backends.c.mtf_c_backend",
        [
            "src/mtflib/backends/c/c_backend.cpp",
            "src/mtflib/backends/c/c_pybind_wrapper.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            numpy.get_include(),
            "src/mtflib/backends/c",
        ],
        language="c++",
        extra_compile_args=cpp_args,
        extra_link_args=link_args,
    ),
]

setup(ext_modules=extensions)
