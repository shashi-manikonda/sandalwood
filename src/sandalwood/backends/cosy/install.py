#!/usr/bin/env python3
"""
Post-installation compiler utility for the Sandalwood COSY backend.
This script compiles the proprietary COSY Infinity Fortran files along with the
Sandalwood wrapper bridge into a shared library, placed in the installed package.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys


class CosyInstaller:
    def __init__(self, cosy_src=None, compiler=None, config_path=None, verbose=False):
        # 1. Determine active backend directory (where this script is installed)
        self.backend_dir = os.path.abspath(os.path.dirname(__file__))
        self.verbose = verbose

        # 2. Determine COSY source path: Priority: CLI arg > ENV > Local directory
        if cosy_src:
            self.cosy_src_orig = os.path.abspath(cosy_src)
        else:
            cosy_src_env = os.environ.get("SANDALWOOD_COSY_SRC")
            if cosy_src_env and os.path.exists(cosy_src_env):
                self.cosy_src_orig = os.path.abspath(cosy_src_env)
            else:
                self.cosy_src_orig = os.path.join(self.backend_dir, "cosy_src")

        # 3. Determine Output Shared Library path and name
        if sys.platform == "win32":
            self.lib_name = "libcosy.dll"
        elif sys.platform == "darwin":
            self.lib_name = "libcosy.dylib"
        else:
            self.lib_name = "libcosy.so"

        self.output_path = os.path.join(self.backend_dir, self.lib_name)
        self.build_temp = os.path.join(self.backend_dir, "build_tmp")

        # 4. Determine Config Path: Priority: CLI arg > Local installed config
        if config_path:
            self.config_path = os.path.abspath(config_path)
        else:
            self.config_path = os.path.join(self.backend_dir, "cosy_config.env")

        # 5. Determine compiler selection
        self.compiler = compiler
        self.compiler_type = None

    def _ensure_win_environment(self):
        """Ensures Visual Studio environment (link.exe, LIB, INCLUDE) is set."""
        if sys.platform != "win32":
            return

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
                if self.compiler_type == "ifx":
                    print("Warning: Could not locate VsDevCmd.bat. Linking may fail.")
            else:
                print(f"Loading environment from {vs_dev_cmd}")
                cmd = f'"{vs_dev_cmd}" -arch=x64 -no_logo && set'
                try:
                    output = subprocess.check_output(cmd, shell=True, text=True)
                    for line in output.splitlines():
                        if "=" in line:
                            key, value = line.split("=", 1)
                            os.environ[key] = value

                    if not shutil.which("link"):
                        print(
                            "Warning: link.exe still not found in PATH after loading VsDevCmd."
                        )

                except subprocess.CalledProcessError as e:
                    print(f"Error loading Visual Studio environment: {e}")

        # Standard Intel compilation libraries path configuration
        intel_lib_dir = None
        ifx_bin = self.compiler
        if ifx_bin and os.path.exists(ifx_bin):
            ifx_real = os.path.realpath(ifx_bin)
            curr = os.path.dirname(ifx_real)
            for _ in range(4):
                lib_candidate = os.path.join(curr, "lib")
                if os.path.exists(lib_candidate):
                    # Check for key Intel Fortran libraries in the candidate directory
                    if (
                        os.path.exists(os.path.join(lib_candidate, "ifconsol.lib"))
                        or os.path.exists(os.path.join(lib_candidate, "libiomp5md.lib"))
                        or os.path.exists(
                            os.path.join(lib_candidate, "intel64", "libiomp5md.lib")
                        )
                        or os.path.exists(
                            os.path.join(lib_candidate, "intel64_win", "libiomp5md.lib")
                        )
                    ):
                        intel_lib_dir = lib_candidate
                        break
                curr = os.path.dirname(curr)

        # Fallback to standard path if not resolved
        if not intel_lib_dir:
            intel_lib_dir = r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib"

        if os.path.exists(intel_lib_dir):
            if self.verbose:
                print(f"Found Intel libraries directory at: {intel_lib_dir}")

            # Add both root lib and subfolders (intel64, intel64_win) to search path if present
            lib_paths = [intel_lib_dir]
            for sub in ["intel64", "intel64_win", "windows/compiler/lib/intel64_win"]:
                sub_path = os.path.join(intel_lib_dir, sub)
                if os.path.exists(sub_path):
                    lib_paths.append(sub_path)

            for path in lib_paths:
                if "LIB" not in os.environ:
                    os.environ["LIB"] = path
                elif path.lower() not in os.environ["LIB"].lower():
                    os.environ["LIB"] += os.pathsep + path

            # Add sibling bin directory to PATH for runtime DLL resolution of compiled binaries (e.g. version.exe)
            intel_bin_dir = os.path.join(os.path.dirname(intel_lib_dir), "bin")
            if os.path.exists(intel_bin_dir):
                if "PATH" not in os.environ:
                    os.environ["PATH"] = intel_bin_dir
                elif intel_bin_dir.lower() not in os.environ["PATH"].lower():
                    os.environ["PATH"] += os.pathsep + intel_bin_dir

    def verify_source(self):
        """Verifies that both COSY source files and bridge wrapper files exist."""
        print(f"Checking for COSY source in: {self.cosy_src_orig}")

        required_cosy = ["dafox.f", "foxfit.f", "foxgraf.f", "version.f"]
        missing_cosy = [
            f
            for f in required_cosy
            if not os.path.exists(os.path.join(self.cosy_src_orig, f))
        ]

        if missing_cosy:
            print(
                f"Error: COSY core files missing in '{self.cosy_src_orig}': {missing_cosy}"
            )
            print(
                "Please ensure your proprietary COSY Infinity source files are placed in that directory,"
            )
            print("or set the environment variable: SANDALWOOD_COSY_SRC")
            return False

        print("COSY source verified successfully.")

        required_bridge = [
            "wrapper.f",
            "helper.f",
            "cosy.def" if sys.platform == "win32" else None,
        ]
        required_bridge = [f for f in required_bridge if f is not None]
        missing_bridge = [
            f
            for f in required_bridge
            if not os.path.exists(os.path.join(self.backend_dir, f))
        ]

        if missing_bridge:
            print(
                f"Error: Sandalwood package bridge files missing in '{self.backend_dir}': {missing_bridge}"
            )
            print("Please ensure the package was correctly installed.")
            return False

        return True

    def detect_compiler(self):
        """Detects a working Fortran compiler (ifx or gfortran)."""
        # If user explicitly provided a compiler via CLI
        if self.compiler:
            if shutil.which(self.compiler):
                self.compiler_type = "ifx" if "ifx" in self.compiler else "gfortran"
                return True
            else:
                print(
                    f"Error: Provided compiler '{self.compiler}' is not executable or not in PATH."
                )
                return False

        # Attempt to parse from cosy_config.env
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                for line in f:
                    if line.startswith("export COSY_COMPILER="):
                        val = line.split("=")[1].strip().lower().strip("'").strip('"')
                        if shutil.which(val):
                            self.compiler = val
                            self.compiler_type = "ifx" if "ifx" in val else "gfortran"
                            break

        # Search PATH
        if not self.compiler:
            if shutil.which("ifx"):
                self.compiler = "ifx"
                self.compiler_type = "ifx"
            elif shutil.which("gfortran"):
                self.compiler = "gfortran"
                self.compiler_type = "gfortran"

        # Windows Fallback Searches
        if not self.compiler and sys.platform == "win32":
            possible_paths = [
                r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\ifx.exe",
                r"C:\ProgramData\chocolatey\bin\gfortran.exe",
                r"C:\msys64\mingw64\bin\gfortran.exe",
                r"C:\MinGW\bin\gfortran.exe",
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    self.compiler = p
                    self.compiler_type = "ifx" if "ifx" in p else "gfortran"
                    break

        # Linux/macOS Fallback Searches
        if not self.compiler and sys.platform != "win32":
            possible_paths = [
                "/opt/intel/oneapi/compiler/latest/linux/bin/intel64/ifx",
                "/opt/intel/oneapi/compiler/latest/bin/ifx",
                "/usr/bin/gfortran",
                "/usr/local/bin/gfortran",
                "/opt/homebrew/bin/gfortran",
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    self.compiler = p
                    self.compiler_type = "ifx" if "ifx" in p else "gfortran"
                    break

        if not self.compiler:
            print("\n[INFO] No Fortran compiler detected on this system.")
            print(
                "Sandalwood will run natively using its accelerated Python/Numba backend."
            )
            print(
                "If you wish to enable the optional COSY Infinity HPC layer, please install gfortran or ifx."
            )
            return False

        print(f"Using compiler: {self.compiler} ({self.compiler_type})")
        return True

    def build(self):
        """Runs the complete source-patching, versioning, and compilation pipeline."""
        if not self.verify_source() or not self.detect_compiler():
            return False

        self._ensure_win_environment()

        # 1. Isolation: Prepare temporary build directory
        print(f"Preparing temporary build directory: {self.build_temp}")
        if os.path.exists(self.build_temp):
            shutil.rmtree(self.build_temp)
        os.makedirs(self.build_temp)

        try:
            # Copy core COSY files
            for f in ["dafox.f", "foxfit.f", "foxgraf.f"]:
                shutil.copy2(os.path.join(self.cosy_src_orig, f), self.build_temp)

            # Copy Sandalwood bridge files
            for f in ["wrapper.f", "helper.f"]:
                shutil.copy2(os.path.join(self.backend_dir, f), self.build_temp)

            # 2. Patch memory parameter configurations
            config_map = {}
            if os.path.exists(self.config_path):
                print(f"Loading memory limits configuration from {self.config_path}...")
                raw_config = {}
                with open(self.config_path, "r") as f:
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
                target_suffix = ""
                if sys.platform == "win32":
                    target_suffix = "_WIN32"
                elif sys.platform.startswith("linux"):
                    target_suffix = "_LINUX"
                elif sys.platform == "darwin":
                    target_suffix = "_DARWIN"

                bases = set()
                for k in raw_config:
                    if k.endswith("_WIN32"):
                        bases.add(k[:-6])
                    elif k.endswith("_LINUX"):
                        bases.add(k[:-6])
                    elif k.endswith("_DARWIN"):
                        bases.add(k[:-7])
                    else:
                        bases.add(k)

                for base in bases:
                    override_key = base + target_suffix
                    if override_key in raw_config:
                        config_map[base] = raw_config[override_key]
                    elif base in raw_config:
                        config_map[base] = raw_config[base]

            if config_map:
                print(f"Applying memory configurations to source code: {config_map}")
                for f_name in os.listdir(self.build_temp):
                    if f_name.endswith(".f"):
                        f_path = os.path.join(self.build_temp, f_name)
                        with open(f_path, "r") as f:
                            content = f.read()

                        modified = False
                        for param, value in config_map.items():
                            pattern = rf"({param}\s*=\s*)\d+"
                            if re.search(pattern, content):
                                content = re.sub(pattern, rf"\g<1>{value}", content)
                                modified = True

                        if modified:
                            with open(f_path, "w") as f:
                                f.write(content)

            # 3. Version switching
            print("Running COSY code compiler/platform adaptation...")
            version_src = os.path.join(self.cosy_src_orig, "version.f")
            version_bin = os.path.join(
                self.build_temp, "version.exe" if sys.platform == "win32" else "version"
            )

            v_cmd = [self.compiler]
            if self.compiler_type == "ifx" and sys.platform == "win32":
                v_cmd.extend(["/nologo", f"/Fe{version_bin}", version_src])
            else:
                v_cmd.extend(["-O3", version_src, "-o", version_bin])

            if self.verbose:
                print(f"Compiling version utility: {' '.join(v_cmd)}")
                res = subprocess.run(v_cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    raise subprocess.CalledProcessError(
                        res.returncode, v_cmd, output=res.stdout, stderr=res.stderr
                    )
            else:
                subprocess.run(
                    v_cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            old_m, new_m = (
                ("*GFOR", "*IFOR")
                if self.compiler_type == "ifx"
                else ("*IFOR", "*GFOR")
            )

            for f_name in os.listdir(self.build_temp):
                if f_name.endswith(".f") and f_name != "version.f":
                    # Use relative names to avoid COSY's hardcoded 20-character filename limit (version.f)
                    f_tmp_name = f_name + ".tmp"

                    # Platform versioning pass
                    p1 = subprocess.Popen(
                        [version_bin],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=self.build_temp,
                        text=True,
                    )
                    stdout, stderr = p1.communicate(
                        input=f"{f_name}\n{f_tmp_name}\n{old_m}\n{new_m}\n"
                    )
                    if p1.returncode != 0:
                        raise RuntimeError(
                            f"Platform versioning failed for {f_name}: {stderr}"
                        )
                    os.replace(
                        os.path.join(self.build_temp, f_tmp_name),
                        os.path.join(self.build_temp, f_name),
                    )

                    # Serial MPI versioning pass
                    p2 = subprocess.Popen(
                        [version_bin],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=self.build_temp,
                        text=True,
                    )
                    stdout, stderr = p2.communicate(
                        input=f"{f_name}\n{f_tmp_name}\n*MPI\n*NORM\n"
                    )
                    if p2.returncode != 0:
                        raise RuntimeError(
                            f"MPI versioning failed for {f_name}: {stderr}"
                        )
                    os.replace(
                        os.path.join(self.build_temp, f_tmp_name),
                        os.path.join(self.build_temp, f_name),
                    )

            print("COSY code adaptation complete.")

            # 4. Link & Compile shared library
            print(
                "Compiling shared bridge library directly into installed library path..."
            )
            files_to_compile = [
                os.path.join(self.build_temp, "dafox.f"),
                os.path.join(self.build_temp, "foxfit.f"),
                os.path.join(self.build_temp, "foxgraf.f"),
                os.path.join(self.build_temp, "helper.f"),
                os.path.join(self.build_temp, "wrapper.f"),
            ]

            cmd = []
            if self.compiler_type == "ifx":
                if sys.platform == "win32":
                    cmd = [
                        self.compiler,
                        "/nologo",
                        "/dll",
                        "/O3",
                        "/Qopenmp",
                        "/fixed",
                        f"/Fo{self.build_temp}\\",
                        f"/Fe{self.output_path}",
                        *files_to_compile,
                        "/link",
                        f"/DEF:{os.path.join(self.backend_dir, 'cosy.def')}",
                    ]
                else:
                    cmd = [
                        self.compiler,
                        "-shared",
                        "-fPIC",
                        "-O3",
                        "-march=native",
                        "-fixed",
                        "-qopenmp",
                        "-diag-disable=10448",
                        *files_to_compile,
                        "-o",
                        self.output_path,
                    ]
            else:
                cmd = [self.compiler, "-shared"]
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
                    *files_to_compile,
                    "-o",
                    self.output_path,
                ])

            if self.verbose:
                print(f"Executing: {' '.join(cmd)}")
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    raise subprocess.CalledProcessError(
                        res.returncode, cmd, output=res.stdout, stderr=res.stderr
                    )
            else:
                print("Executing compilation command...")
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            print("\n[SUCCESS] Successfully compiled COSY Backend Shared Library!")
            print(f"Library saved at: {self.output_path}\n")
            return True

        except subprocess.CalledProcessError as e:
            print(
                f"\n[ERROR] Compilation command failed with exit status {e.returncode}: {e}"
            )
            if e.stdout:
                print(f"Compiler stdout:\n{e.stdout}")
            if e.stderr:
                print(f"Compiler stderr:\n{e.stderr}")
            return False
        except Exception as e:
            print(f"\n[ERROR] Compilation failed: {e}")
            import traceback

            traceback.print_exc()
            return False
        finally:
            # Clean up build directory
            if os.path.exists(self.build_temp):
                shutil.rmtree(self.build_temp)


def main():
    parser = argparse.ArgumentParser(
        description="Sandalwood COSY post-installation shared library builder utility."
    )
    parser.add_argument(
        "--src",
        help="Path to directory containing COSY Infinity source files (dafox.f, etc.). "
        "Defaults to environment variable SANDALWOOD_COSY_SRC.",
        default=None,
    )
    parser.add_argument(
        "--compiler",
        help="Path or name of compiler executable (e.g. ifx, gfortran).",
        default=None,
    )
    parser.add_argument(
        "--config", help="Path to custom cosy_config.env file.", default=None
    )
    parser.add_argument(
        "--verbose",
        "-v",
        help="Enable full verbose compiler stdout reporting.",
        action="store_true",
    )

    args = parser.parse_args()

    installer = CosyInstaller(
        cosy_src=args.src,
        compiler=args.compiler,
        config_path=args.config,
        verbose=args.verbose,
    )

    success = installer.build()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
