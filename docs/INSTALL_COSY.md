# Installing the COSY Infinity Backend

Sandalwood supports a high-performance backend based on **COSY Infinity**. Due to licensing restrictions, the COSY Fortran source files are not distributed with the Sandalwood library. 

Licensed users of COSY Infinity can enable this backend by following the steps below.

## 1. Obtain COSY Infinity Source
You must have a valid license for COSY Infinity. You can request it from the official [COSY Infinity website](https://cosyinfinity.org/).

The following files are required from the COSY distribution:
- `dafox.f`
- `foxfit.f`
- `foxgraf.f`
- `foxy.f`
- `version.f`

## 2. Integration Strategies

There are two ways to provide the COSY source to Sandalwood during installation:

### Option A: Environment Variable (Recommended)
Set the `SANDALWOOD_COSY_SRC` environment variable to point to the directory containing the `.f` files before installing Sandalwood.

**Windows (PowerShell):**
```powershell
$env:SANDALWOOD_COSY_SRC = "C:\Path\To\COSY\Source"
pip install -e .[dev]
```

**Linux/macOS:**
```bash
export SANDALWOOD_COSY_SRC=/path/to/cosy/source
pip install -e .[dev]
```

### Option B: Local Drop-in
Copy the required `.f` files into the following directory within the Sandalwood source tree:
`src/sandalwood/backends/cosy/cosy_src/`

Then run the installation:
```bash
pip install -e .[dev]
```

## 3. Customizing Memory Limits
You can customize the memory limits and dimensions of the COSY backend by editing `src/sandalwood/backends/cosy/cosy_config.env`. The build system will automatically patch the Fortran source code with these values during compilation.

## 4. Verification
After installation, you can verify that the COSY backend is active by running:
```python
import sandalwood
print(sandalwood.taylor_function._COSY_BACKEND_AVAILABLE)
```
If it returns `True`, the backend was successfully built and linked.
