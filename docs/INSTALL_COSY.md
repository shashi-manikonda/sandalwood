# Installing and Side-Loading the COSY Infinity Backend

Sandalwood supports a high-performance backend based on **COSY Infinity**. Due to licensing restrictions, the COSY Fortran source files are not distributed with the Sandalwood library. 

Licensed users of COSY Infinity can enable this backend on **Windows** and **Linux** using either **Intel oneAPI (`ifx`)** or **GNU Fortran (`gfortran`)** compilers.

---

## 📋 System Requirements

To compile the COSY backend, your system must have the following dependencies:

### Windows:
*   **Fortran Compiler**: Either **Intel oneAPI Fortran Compiler (`ifx`)** or **GNU Fortran (`gfortran`)**.
*   **C Linker & SDK**: **Visual Studio Build Tools 2022** (specifically the MSVC C++ build tools and Windows SDK) is required to link the shared library when compiling with `ifx`.
*   *Note*: If using `ifx`, Sandalwood will automatically locate Visual Studio's developer command prompt (`VsDevCmd.bat`) and map all necessary linker and runtime libraries.

### Linux:
*   **Fortran Compiler**: Either **Intel oneAPI Fortran Compiler (`ifx`)** or **GNU Fortran (`gfortran`)**.
*   **Development Tools**: Standard build tools (`make`, `gcc`, `glibc-devel`).

---

## 📂 1. Obtain COSY Infinity Source
You must have a valid license for COSY Infinity. You can request it from the official [COSY Infinity website](https://cosyinfinity.org/).

The following files are required from the COSY distribution:
*   `dafox.f`
*   `foxfit.f`
*   `foxgraf.f`
*   `version.f`

---

## 🛠️ 2. Integration Strategies

Sandalwood offers two ways to integrate and compile your licensed COSY source:

### Strategy A: Post-Installation Side-Load (Recommended for Pip Users)
If you installed Sandalwood directly from PyPI (`pip install sandalwood`), you can use the built-in CLI tool to compile and hook the COSY backend into your existing installation.

#### Step 1: Install Sandalwood
```bash
pip install sandalwood
```

#### Step 2: Run the Setup CLI Tool
Run the `sandalwood-setup-cosy` script, passing the path to the directory containing your COSY source `.f` files:

**Windows (PowerShell):**
```powershell
sandalwood-setup-cosy --src "C:\Path\To\COSY10p2\src" -v
```

**Linux:**
```bash
sandalwood-setup-cosy --src "/path/to/COSY10p2/src" -v
```

#### CLI Options:
*   `--src PATH`: The directory containing your licensed `dafox.f`, etc. (Can also be set via the `SANDALWOOD_COSY_SRC` environment variable).
*   `--compiler COMPILER`: Manually choose a compiler executable (e.g. `ifx`, `gfortran`). If omitted, Sandalwood will auto-detect standard compilers.
*   `--config CONFIG`: Provide a custom `cosy_config.env` containing custom memory and dimension parameters.
*   `-v, --verbose`: Enable verbose reporting of compiler command lines and outputs for easy debugging.

---

### Strategy B: Build-on-Install (For Developers & Source Installs)
If you are developing Sandalwood or building the package from source, you can instruct the installer to compile the COSY backend during the installation process itself.

#### Option 1: Environment Variable (Recommended)
Set the `SANDALWOOD_COSY_SRC` environment variable to point to the directory containing your COSY `.f` files, then install Sandalwood in editable mode.

**Windows (PowerShell):**
```powershell
$env:SANDALWOOD_BUILD_COSY_ON_INSTALL = "1"
$env:SANDALWOOD_COSY_SRC = "C:\Path\To\COSY\Source"
pip install -e .[dev]
```

**Linux/macOS:**
```bash
export SANDALWOOD_BUILD_COSY_ON_INSTALL=1
export SANDALWOOD_COSY_SRC=/path/to/cosy/source
pip install -e .[dev]
```

#### Option 2: Local Drop-in
Copy the required `.f` files directly into the repository folder:
`src/sandalwood/backends/cosy/cosy_src/`

Then install:
```bash
$env:SANDALWOOD_BUILD_COSY_ON_INSTALL = "1"  # On Windows PowerShell
# OR export SANDALWOOD_BUILD_COSY_ON_INSTALL=1  # On Linux
pip install -e .[dev]
```

---

## ⚙️ 3. Customizing Memory Limits
You can customize the memory limits and dimensions of the COSY backend by editing `src/sandalwood/backends/cosy/cosy_config.env`. 

The compilation utility will automatically patch the COSY Fortran source code with these values during compilation (e.g. `LMEM`, `LDIM`, `LNV`, `LVAR`, `LNO`).

---

## 🔍 4. Verification
After compiling, you can verify that the COSY backend is active by running:

```python
import sandalwood
print(sandalwood.taylor_function._COSY_BACKEND_AVAILABLE)
```

If it returns **`True`**, the backend was successfully built, linked, and loaded!

If it returns **`False`**, you can run:
```bash
sandalwood-setup-cosy -v
```
to view detailed diagnostics of the compiler checks and identify any missing paths or libraries on your system.
