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
        has_link = shutil.which("link") and "LIB" in os.environ
        
        if not has_link:
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
                # Don't return, try finding Intel libs anyway
            else:
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

        # Try standard path
        intel_lib_dir = r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib"
        
        # Also check relative to ifx if available
        ifx_path = shutil.which("ifx")
        if ifx_path:
             root = os.path.dirname(os.path.dirname(ifx_path))
             c1 = os.path.join(root, "lib")
             c2 = os.path.join(root, "windows", "compiler", "lib", "intel64_win")
             if os.path.exists(os.path.join(c1, "libiomp5md.lib")):
                 intel_lib_dir = c1
             elif os.path.exists(os.path.join(c2, "libiomp5md.lib")):
                 intel_lib_dir = c2
        
        if os.path.exists(os.path.join(intel_lib_dir, "libiomp5md.lib")):
             print(f"Found Intel libraries at: {intel_lib_dir}")
             # Always add it if not clearly present
             if "LIB" not in os.environ:
                  os.environ["LIB"] = intel_lib_dir
             elif intel_lib_dir.lower() not in os.environ["LIB"].lower():
                  print(f"Adding {intel_lib_dir} to LIB")
                  os.environ["LIB"] += os.pathsep + intel_lib_dir
        else:
             print(f"Warning: Could not locate libiomp5md.lib at {intel_lib_dir}")

        print(f"LIB environment variable length: {len(os.environ.get('LIB', ''))}")


    def run(self):
        """Runs the COSY compilation logic."""
        self._ensure_win_environment()
        # 1. Setup Paths
        dir_path = os.path.abspath(os.path.dirname(__file__))
        backend_dir = os.path.join(dir_path, "src", "sandalwood", "backends", "cosy")
        
        # Determine Source Path: Priority: ENV > Local cosy_src
        cosy_src_env = os.environ.get("SANDALWOOD_COSY_SRC")
        if cosy_src_env and os.path.exists(cosy_src_env):
            cosy_src_orig = os.path.abspath(cosy_src_env)
            print(f"Using COSY source from environment variable: {cosy_src_orig}")
        else:
            cosy_src_orig = os.path.join(backend_dir, "cosy_src")
            print(f"Checking for COSY source in local directory: {cosy_src_orig}")

        # Verify source exists
        required_files = ["dafox.f", "foxfit.f", "foxgraf.f", "version.f"]
        missing = [f for f in required_files if not os.path.exists(os.path.join(cosy_src_orig, f))]
        
        if missing:
            print(f"Warning: COSY source files missing in {cosy_src_orig}: {missing}")
            print("COSY backend will not be built. Please set SANDALWOOD_COSY_SRC if you have a COSY license.")
            return

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

        # A. Core COSY files (from external source)
        cosy_core_files = ["dafox.f", "foxfit.f", "foxgraf.f"]
        for f in cosy_core_files:
            shutil.copy2(os.path.join(cosy_src_orig, f), build_temp)
        
        # B. Sandalwood bridge files (from local directory)
        sandalwood_bridge_files = ["wrapper.f", "helper.f"]
        for f in sandalwood_bridge_files:
            src_path = os.path.join(backend_dir, f)
            if os.path.exists(src_path):
                shutil.copy2(src_path, build_temp)
            else:
                print(f"Warning: Sandalwood bridge file missing: {src_path}")

        # 3.5 Load and Apply Memory Patches
        config_map = {}
        if os.path.exists(config_path):
            print(f"Loading COSY memory configuration from {config_path}...")
            raw_config = {}
            with open(config_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[7:].strip()
                    if "=" in line:
                        key, val = line.split("=", 1)
                        if key.startswith("COSY_"):
                            root_key = key[5:]
                            raw_config[root_key] = val.strip()

            # Resolve OS-specific overrides
            # Priority: VAR_PLATFORM > VAR
            target_suffix = ""
            if sys.platform == "win32":
                target_suffix = "_WIN32"
            elif sys.platform.startswith("linux"):
                target_suffix = "_LINUX"
            elif sys.platform == "darwin":
                target_suffix = "_DARWIN"

            # First pass: collect bases
            bases = set()
            for k in raw_config:
                if k.endswith("_WIN32"): bases.add(k[:-6])
                elif k.endswith("_LINUX"): bases.add(k[:-6])
                elif k.endswith("_DARWIN"): bases.add(k[:-7])
                else: bases.add(k)
            
            for base in bases:
                # Check for specific override first
                override_key = base + target_suffix
                if override_key in raw_config:
                    config_map[base] = raw_config[override_key]
                elif base in raw_config:
                    config_map[base] = raw_config[base]

        if config_map:
            print(f"Applying memory patches: {config_map}")
            target_patch_files = [os.path.join(build_temp, f) for f in os.listdir(build_temp) if f.endswith(".f")]
            for file_path in target_patch_files:
                with open(file_path, "r") as f:
                    content = f.read()
                
                modified = False
                for param, value in config_map.items():
                    # Pattern matches "PARAM = NUMBER"
                    pattern = fr"({param}\s*=\s*)\d+"
                    if re.search(pattern, content):
                        content = re.sub(pattern, fr"\g<1>{value}", content)
                        modified = True
                
                if modified:
                    with open(file_path, "w") as f:
                        f.write(content)

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
            subprocess.run(v_cmd, check=True)
            
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
