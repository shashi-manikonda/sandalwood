# Sandalwood Local Setup Guide

This guide provides step-by-step instructions for setting up a local development environment for Sandalwood, installing the package in editable mode, compiling the COSY Infinity HPC backend using either Intel oneAPI (`ifx`) or GNU Fortran (`gfortran`), and running tests to verify the installation.

---

## 📦 1. Create a Virtual Environment

You can set up a Python virtual environment using either the fast `uv` package manager or the standard `venv` module.

### Option A: Using `uv` (Recommended)
`uv` is extremely fast and integrates with the `uv.lock` file in this repository.

```bash
# Create the virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate
```

### Option B: Using standard Python `venv`
```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

---

## 🛠️ 2. Install Sandalwood in Editable Mode

Once your virtual environment is active, install Sandalwood in editable mode with development dependencies:

### Using `uv` (Recommended)
```bash
uv pip install -e .[dev]
```

### Using `pip`
```bash
pip install -e .[dev]
```

---

## ⚙️ 3. Compile the COSY Backend

To enable the accelerated COSY Infinity backend, run the `sandalwood-setup-cosy` CLI tool. Specify the COSY source directory (where `dafox.f`, etc., are located) and your chosen compiler.

### Option A: Using Intel oneAPI (`ifx`)

1. **Initialize the Intel Environment**: Sourcing the compiler environment configuration script is necessary so that `ifx` and its dependencies are available in your shell's `PATH` and `LD_LIBRARY_PATH`.
   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```

2. **Run Compilation**:
   ```bash
   sandalwood-setup-cosy --src /home/mls/work/cosy10p2/src --compiler ifx -v
   ```

### Option B: Using GNU Fortran (`gfortran`)

1. **Ensure `gfortran` is installed**:
   On Ubuntu/Debian:
   ```bash
   sudo apt-get update && sudo apt-get install gfortran build-essential
   ```

2. **Run Compilation**:
   ```bash
   sandalwood-setup-cosy --src /home/mls/work/cosy10p2/src --compiler gfortran -v
   ```

---

## 🔍 4. Verification

After compiling the backend, verify that everything is configured and linked correctly.

### 1. Check Shared Library Linking
Verify that all shared library dependencies can be resolved. (If you compiled with `ifx`, make sure to run this inside a shell where `setvars.sh` has been sourced).

```bash
# (Optional) Source Intel environment if compiled with ifx
# source /opt/intel/oneapi/setvars.sh

ldd src/sandalwood/backends/cosy/libcosy.so
```

### 2. Verify in Python
Run Python and verify that the COSY backend is loaded:

```python
import sandalwood
# Should print True if compiled and linked successfully
print(sandalwood.backends.cosy.cosy_backend._COSY_BACKEND_AVAILABLE)
```

### 3. Run the Test Suite
Ensure the compiled backend passes all tests. (Ensure `setvars.sh` is sourced if using `ifx`).

```bash
# Run backend-specific tests
pytest tests/test_backend.py

# Run the full test suite
pytest
```
