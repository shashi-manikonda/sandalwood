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

    def _ensure_win_environment(self):
        """Ensures Visual Studio environment (link.exe, LIB, INCLUDE) is set."""
        if sys.platform != "win32":
            return

        # Check if link.exe is functional (basic check)
        if shutil.which("link") and "LIB" in os.environ:
            return

        print("Configuring Visual Studio environment...")
        possible_roots = [
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise",
        ]

        vs_dev_cmd = None
        for root in possible_roots:
            candidate = os.path.join(root, "Common7", "Tools", "VsDevCmd.bat")
            if os.path.exists(candidate):
                vs_dev_cmd = candidate
                break

        if not vs_dev_cmd:
            print("Warning: Could not locate VsDevCmd.bat. Linking may fail.")
            return

        print(f"Loading environment from {vs_dev_cmd}")
        # Run VsDevCmd.bat and dump environment
        cmd = f'"{vs_dev_cmd}" -arch=x64 -no_logo && set'
        try:
            output = subprocess.check_output(cmd, shell=True, text=True)
            for line in output.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    # Update PATH, LIB, INCLUDE, LIBPATH
                    if key.upper() in ["PATH", "LIB", "INCLUDE", "LIBPATH"]:
                        os.environ[key] = value
            
            # Explicitly verify link.exe again
            if not shutil.which("link"):
                 print("Warning: link.exe still not found in PATH after loading VsDevCmd.")
                 
        except subprocess.CalledProcessError as e:
            print(f"Error loading Visual Studio environment: {e}")

        # Ensure Intel libraries are in LIB
        intel_lib_found = False
        if "LIB" in os.environ:
             for path in os.environ["LIB"].split(os.pathsep):
                 if os.path.join(path, "libiomp5md.lib") and os.path.exists(os.path.join(path, "libiomp5md.lib")):
                      intel_lib_found = True
                      break
        
        if not intel_lib_found:
             # Try standard path
             intel_lib_path = r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib"
             
             # Also check relative to ifx if available
             ifx_path = shutil.which("ifx")
             if ifx_path:
                  # Expected: .../bin/ifx.exe -> .../lib or .../windows/compiler/lib/intel64_win
                  root = os.path.dirname(os.path.dirname(ifx_path))
                  candidates = [
                      os.path.join(root, "lib"),
                      os.path.join(root, "windows", "compiler", "lib", "intel64_win"),
                  ]
                  for c in candidates:
                      if os.path.exists(os.path.join(c, "libiomp5md.lib")):
                          intel_lib_path = c
                          break
             
             if os.path.exists(os.path.join(intel_lib_path, "libiomp5md.lib")):
                  print(f"Adding Intel library path: {intel_lib_path}")
                  if "LIB" in os.environ:
                       os.environ["LIB"] += os.pathsep + intel_lib_path
                  else:
                       os.environ["LIB"] = intel_lib_path
             else:
                  print("Warning: Could not locate libiomp5md.lib. Linking may fail.")


    def run(self):
        """Runs the COSY compilation logic."""
        self._ensure_win_environment()
        # 1. Setup Paths
        dir_path = os.path.abspath(os.path.dirname(__file__))
        backend_dir = os.path.join(dir_path, "src", "sandalwood", "backends", "cosy")
        cosy_src_orig = os.path.join(backend_dir, "cosy_src")
        build_temp = os.path.join(backend_dir, "build_tmp")
        config_path = os.path.join(backend_dir, "cosy_config.env")

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

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                for line in f:
                    if line.startswith("export COSY_COMPILER="):
                        val = line.split("=")[1].strip().lower().strip("'").strip('"')
                        if shutil.which(val):
                            compiler = val
                            compiler_type = "ifx" if "ifx" in val else "gfortran"
                            break
        
        if not compiler:
            compiler = os.environ.get("COSY_COMPILER")
            if compiler:
                compiler_type = "ifx" if "ifx" in compiler else "gfortran"

        # B. Fallback: Search PATH
        if not compiler:
            if shutil.which("ifx"):
                compiler = "ifx"
                compiler_type = "ifx"
            elif shutil.which("gfortran"):
                compiler = "gfortran"
                compiler_type = "gfortran"
        
        # C. Windows fallback search for MinGW/Chocolatey
        if not compiler and sys.platform == "win32":
            possible_paths = [
                r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\ifx.exe",
                r"C:\ProgramData\chocolatey\bin\gfortran.exe",
                r"C:\msys64\mingw64\bin\gfortran.exe",
                r"C:\MinGW\bin\gfortran.exe",
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    compiler = p
                    compiler_type = "ifx" if "ifx" in p else "gfortran"
                    break

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

        # 4. Version Switching using version.f
        print(f"Switching code versions using {compiler_type}...")
        version_src = os.path.join(cosy_src_orig, "version.f")
        version_bin = os.path.join(build_temp, "version.exe" if sys.platform == "win32" else "version")
        
        try:
            # Compile version utility
            v_cmd = [compiler]
            if compiler_type == "ifx" and sys.platform == "win32":
                v_cmd.extend(["/nologo", f"/Fe{version_bin}", version_src])
            else:
                v_cmd.extend(["-O3", version_src, "-o", version_bin])
            
            print(f"Compiling version utility: {' '.join(v_cmd)}")
            subprocess.run(v_cmd, check=True, env=build_env)
            
            # Determine markers
            if compiler_type == "ifx":
                old_m, new_m = "*GFOR", "*IFOR"
            else:
                old_m, new_m = "*IFOR", "*GFOR"
            
            print(f"Applying version markers: {old_m} -> {new_m}")
            
            # Run on all copied .f files
            for f_name in os.listdir(build_temp):
                if f_name.endswith(".f") and f_name != "version.f":
                    f_path = os.path.join(build_temp, f_name)
                    f_tmp = f_path + ".tmp"
                    
                    # Pass 1: Platform switching (GFOR <-> IFOR)
                    p1 = subprocess.Popen([version_bin], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    p1.communicate(input=f"{f_path}\n{f_tmp}\n{old_m}\n{new_m}\n")
                    os.replace(f_tmp, f_path)
                    
                    # Pass 2: Serial switching (MPI -> NORM)
                    p2 = subprocess.Popen([version_bin], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    p2.communicate(input=f"{f_path}\n{f_tmp}\n*MPI\n*NORM\n")
                    os.replace(f_tmp, f_path)
            
            print("Version switching complete.")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not use version.f utility: {e}")
        except Exception as e:
            print(f"Warning: Error during version switching: {e}")

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
            if sys.platform == "win32":
                cmd = [
                    compiler,
                    "/nologo",
                    "/dll",
                    "/O3",
                    "/Qopenmp",
                    "/fixed",
                    f"/Fo{build_temp}\\",
                    f"/Fe{output_path}",
                    *files_to_compile,
                    "/link",
                    f"/DEF:{os.path.join(backend_dir, 'cosy.def')}",
                ]
            else:
                # Linux ifx
                cmd = [
                    compiler,
                    "-shared",
                    "-fPIC",
                    "-O3",
                    "-march=native",
                    "-fixed",
                    "-qopenmp",
                    "-diag-disable=10448",
                    *files_to_compile,
                    "-o",
                    output_path,
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
