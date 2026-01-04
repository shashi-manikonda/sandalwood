# This file is used to build the C++ and C extensions for the sandalwood library.
# It is required by setuptools, which is specified as the build backend in
# pyproject.toml. For more details on how setuptools handles extensions, see:
# https://setuptools.pypa.io/en/latest/userguide/ext_modules.html

import os
import shutil
import subprocess
import sys

import numpy
import pybind11
from setuptools import Command, Extension, setup
from setuptools.command.build_ext import build_ext

class BuildCosy(Command):
    """Custom command to build the COSY backend."""
    description = "build COSY shared library"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Runs the COSY compilation script or equivalent logic."""
        dir_path = os.path.abspath(os.path.dirname(__file__))
        backend_dir = os.path.join(dir_path, "src", "sandalwood", "backends", "cosy")
        cosy_src = os.path.join(backend_dir, "cosy_src")
        
        # Platform-specific output
        if sys.platform == "win32":
            lib_name = "libcosy.dll"
        elif sys.platform == "darwin":
            lib_name = "libcosy.dylib"
        else:
            lib_name = "libcosy.so"
        
        output_path = os.path.join(backend_dir, lib_name)
        
        # Find gfortran
        gfortran = shutil.which("gfortran")
        if not gfortran:
            print("Warning: gfortran not found. COSY backend will not be built.")
            return

        # Compilation arguments matching C++ backend performance levels
        cmd = [
            gfortran, "-shared", "-fPIC", "-std=legacy", "-g", "-O3",
            "-march=native",
            os.path.join(cosy_src, "dafox.f"),
            os.path.join(cosy_src, "foxfit.f"),
            os.path.join(cosy_src, "foxgraf.f"),
            os.path.join(backend_dir, "wrapper.f"),
            "-o", output_path
        ]
        
        print(f"Building COSY library: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully built {lib_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error building COSY library: {e}")
            raise

class CustomBuildExt(build_ext):
    """Custom build_ext to ensure COSY is built."""
    def run(self):
        self.run_command("build_cosy")
        super().run()

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
        "sandalwood.backends.cpp.mtf_cpp",
        [
            "src/sandalwood/backends/cpp/mtf_data.cpp",
            "src/sandalwood/backends/cpp/pybind_wrapper.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            numpy.get_include(),
            "src/sandalwood/backends/cpp",
        ],
        language="c++",
        extra_compile_args=cpp_args,
        extra_link_args=link_args,
    ),
    Extension(
        "sandalwood.backends.c.mtf_c_backend",
        [
            "src/sandalwood/backends/c/c_backend.cpp",
            "src/sandalwood/backends/c/c_pybind_wrapper.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            numpy.get_include(),
            "src/sandalwood/backends/c",
        ],
        language="c++",
        extra_compile_args=cpp_args,
        extra_link_args=link_args,
    ),
]

setup(
    ext_modules=extensions,
    cmdclass={
        "build_cosy": BuildCosy,
        "build_ext": CustomBuildExt,
    }
)
