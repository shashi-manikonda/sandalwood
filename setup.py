# This file is used to build the C++ and C extensions for the sandalwood library.
# It is required by setuptools, which is specified as the build backend in
# pyproject.toml. For more details on how setuptools handles extensions, see:
# https://setuptools.pypa.io/en/latest/userguide/ext_modules.html

import os
import shutil
import subprocess
import sys

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

        # Compilation arguments for robust legacy Fortran support
        cmd = [
            gfortran, "-shared", "-fPIC", "-fcommon", "-std=legacy", "-g", "-O3",
            "-march=native", "-ffixed-form",
            os.path.join(cosy_src, "dafox.f"),
            os.path.join(cosy_src, "foxfit.f"),
            os.path.join(cosy_src, "foxgraf.f"),
            os.path.join(cosy_src, "helper.f"),
            os.path.join(backend_dir, "wrapper.f"),
            "-fopenmp",
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
        # Only run if we are actually building or if we want to force it
        self.run_command("build_cosy")
        # If we have real extensions, super().run() will build them.
        # If we only have our dummy, it will just ensure build_ext was called.
        super().run()

# Set compiler arguments (not used for extensions anymore but keeping for potential future use or reference)
# Actually, since we are removing all standard extensions, we can simplify this.

setup(
    ext_modules=[
        Extension("sandalwood.backends.cosy._dummy", sources=[])
    ],
    cmdclass={
        "build_cosy": BuildCosy,
        "build_ext": CustomBuildExt,
    }
)
