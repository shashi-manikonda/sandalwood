# This file is used to build the C++ and C extensions for the sandalwood library.
import os
import re
import shutil
import subprocess
import sys
from setuptools import Command, setup
from setuptools.command.build_py import build_py

class BuildCosy(Command):
    """Custom command to build the COSY backend with source patching."""

    description = "build COSY shared library with dynamic patching"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Runs the COSY compilation logic."""
        # 1. Setup Paths
        dir_path = os.path.abspath(os.path.dirname(__file__))
        backend_dir = os.path.join(dir_path, "src", "sandalwood", "backends", "cosy")
        cosy_src_orig = os.path.join(backend_dir, "cosy_src")
        build_temp = os.path.join(backend_dir, "build_tmp")

        # Determine Output Name
        if sys.platform == "win32":
            lib_name = "libcosy.dll"
        elif sys.platform == "darwin":
            lib_name = "libcosy.dylib"
        else:
            lib_name = "libcosy.so"

        output_path = os.path.join(backend_dir, lib_name)

        # 2. Compiler Detection Strategy
        compiler = None
        compiler_type = None

        # A. Priority: Check PATH (Critical for 'setvars.bat' users)
        if sys.platform == "win32":
            if shutil.which("ifx"):
                compiler = "ifx"
                compiler_type = "ifx"
        
        # B. Fallback: Default Intel Install Path
        if not compiler and sys.platform == "win32":
            default_ifx = r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\ifx.exe"
            if os.path.exists(default_ifx):
                compiler = default_ifx
                compiler_type = "ifx"

        # C. Fallback: GFortran
        if not compiler:
            compiler = shutil.which("gfortran")
            if compiler:
                compiler_type = "gfortran"
            # Windows fallback search for MinGW/Chocolatey
            elif sys.platform == "win32":
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

        print(f"Using Compiler: {compiler} ({compiler_type})")

        # 3. Source Isolation (Copy to build_tmp)
        print(f"Preparing build directory: {build_temp}")
        if os.path.exists(build_temp):
            shutil.rmtree(build_temp)
        os.makedirs(build_temp)

        src_files = ["dafox.f", "foxfit.f", "foxgraf.f", "helper.f"]
        
        # Copy core COSY files
        for f in src_files:
            shutil.copy2(os.path.join(cosy_src_orig, f), build_temp)
        
        # Copy wrapper.f (it lives one level up)
        shutil.copy2(os.path.join(backend_dir, "wrapper.f"), build_temp)

        # 4. Dynamic Patching
        if compiler_type == "ifx":
            print("Applying Intel Fortran patches (SLEEP -> SLEEPQQ)...")
            dafox_path = os.path.join(build_temp, "dafox.f")
            
            with open(dafox_path, "r") as f:
                content = f.read()
            
            # Regex: Finds 'CALL SLEEP(X)' and replaces with 'CALL SLEEPQQ(INT(X*1000))'
            # Captures the argument inside parentheses group(1)
            # Use case-insensitive flag because Fortran is case-insensitive
            new_content = re.sub(
                r"CALL\s+SLEEP\s*\(([^)]+)\)", 
                r"CALL SLEEPQQ(INT(\1*1000))", 
                content, 
                flags=re.IGNORECASE
            )
            
            with open(dafox_path, "w") as f:
                f.write(new_content)

        # 5. Compilation
        cmd = []
        # Define paths to the COPIED files in build_temp
        # Note: We must compile the files in the temp dir
        files_to_compile = [
            os.path.join(build_temp, "dafox.f"),
            os.path.join(build_temp, "foxfit.f"),
            os.path.join(build_temp, "foxgraf.f"),
            os.path.join(build_temp, "helper.f"),
            os.path.join(build_temp, "wrapper.f"),
        ]

        if compiler_type == "ifx":
            cmd = [
                compiler,
                "/nologo",
                "/dll",
                "/O3",
                "/Qopenmp",
                "/fixed",
                f"/Fe{output_path}",
                *files_to_compile, # Unpack list
                "/link",
                f"/DEF:{os.path.join(backend_dir, 'cosy.def')}",
            ]
        else:
            # gfortran
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
                "-fopenmp",
                *files_to_compile, # Unpack list
                "-o",
                output_path,
            ])

        print(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully built {lib_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error building COSY library: {e}")
            print("Continuing installation without COSY backend...")
        
        # Cleanup (Optional - keep commented for debugging)
        # shutil.rmtree(build_temp)

class CustomBuildPy(build_py):
    """Custom build_py to ensure COSY is built before packaging."""
    def run(self):
        self.run_command("build_cosy")
        super().run()

setup(
    ext_modules=[],
    cmdclass={
        "build_cosy": BuildCosy,
        "build_py": CustomBuildPy,
    },
)
