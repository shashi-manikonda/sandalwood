# This file is used to build the C++ and C extensions for the sandalwood library.
# It is required by setuptools, which is specified as the build backend in
# pyproject.toml. For more details on how setuptools handles extensions, see:
# https://setuptools.pypa.io/en/latest/userguide/ext_modules.html

import os
import shutil
import subprocess
import sys

from setuptools import Command, Extension, setup
# from setuptools.command.build_ext import build_ext  # No longer needed



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

        # Find compiler (ifx or gfortran)
        compiler = None
        compiler_type = None

        # 1. Prefer Intel ifx on Windows
        if sys.platform == "win32":
            ifx_path = r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\ifx.exe"
            if os.path.exists(ifx_path):
                compiler = ifx_path
                compiler_type = "ifx"

        # 2. Fallback to gfortran
        if not compiler:
            compiler = shutil.which("gfortran")
            compiler_type = "gfortran"
        
        # Windows-specific search for gfortran (e.g., MinGW) if not in PATH and not using ifx
        if sys.platform == "win32" and not compiler:
            possible_paths = [
                r"C:\ProgramData\chocolatey\bin\gfortran.exe",
                r"C:\msys64\mingw64\bin\gfortran.exe",
                r"C:\MinGW\bin\gfortran.exe",
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    compiler = p
                    compiler_type = "gfortran"
                    break

        if not compiler:
            print("Warning: No Fortran compiler (ifx or gfortran) found. COSY backend will not be built.")
            return

        cmd = []
        if compiler_type == "ifx":
            print(f"Found Intel Fortran Compiler: {compiler}")
            cmd = [
                compiler,
                "/nologo",
                "/dll",
                "/O3",
                "/Qopenmp",
                "/fixed",
                f"/Fe{output_path}",
                os.path.join(cosy_src, "dafox.f"),
                os.path.join(cosy_src, "foxfit.f"),
                os.path.join(cosy_src, "foxgraf.f"),
                os.path.join(cosy_src, "helper.f"),
                os.path.join(backend_dir, "wrapper.f"),
                "/link",
                f"/DEF:{os.path.join(backend_dir, 'cosy.def')}",
            ]
        else:
            # gfortran
            print(f"Found GNU Fortran Compiler: {compiler}")
            cmd = [
                compiler,
                "-shared",
            ]
            
            if sys.platform != "win32":
                cmd.append("-fPIC")

            cmd.extend([
                "-fcommon",
                "-std=legacy",
                "-g",
                "-O3",
                "-march=native",
                "-ffixed-form",
                os.path.join(cosy_src, "dafox.f"),
                os.path.join(cosy_src, "foxfit.f"),
                os.path.join(cosy_src, "foxgraf.f"),
                os.path.join(cosy_src, "helper.f"),
                os.path.join(backend_dir, "wrapper.f"),
                "-fopenmp",
                "-o",
                output_path,
            ])

        print(f"Building COSY library: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully built {lib_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error building COSY library: {e}")
            # Do not raise so installation succeeds without backend
            print("Continuing installation without COSY backend...")



from setuptools.command.build_py import build_py

class CustomBuildPy(build_py):
    """Custom build_py to ensure COSY is built before packaging."""

    def run(self):
        # Trigger COSY build
        self.run_command("build_cosy")
        super().run()


setup(
    ext_modules=[],
    cmdclass={
        "build_cosy": BuildCosy,
        "build_py": CustomBuildPy,
    },
)


