---
description: Rebuild COSY Infinity Backend
---

Use this when modifying `wrapper.f` or `cosy_config.env`.

1.  // turbo
    **Rebuild Shared Library**:
    - **Linux/macOS**:
      ```bash
      bash src/sandalwood/backends/cosy/compile_cosy.sh
      ```
    - **Windows**:
      ```bash
      python setup.py build_cosy
      ```
2.  **Verify Linking**:
    ```bash
    ldd src/sandalwood/backends/cosy/libcosy.so
    ```
3.  **Verify Bindings**:
    Run the solver tests to confirm Python can call the new Fortran symbols:
    ```bash
    pytest tests/test_cosy_backend.py
    ```
