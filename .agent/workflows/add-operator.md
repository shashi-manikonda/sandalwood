---
description: Adding a new mathematical operator or elementary function
---

Procedure for implementing new math logic ensuring Backend Parity.

1.  **Define Interface**: Add the operator method to `MultivariateTaylorFunction` in `src/sandalwood/mtf.py`.
2.  **Implement Python Backend**: 
    - Add the corresponding logic to `MtfData` in `src/sandalwood/backends/python/mtf_data.py`.
    - Use `@numba.njit` for performance.
3.  **Implement COSY Backend**:
    - Add the logic to `CosyMtfData` in `src/sandalwood/backends/cosy/cosy_mtf_data.py`.
    - Use `CosyBackend` bindings to invoke the Fortran kernel.
4.  **Verify Parity**:
    - // turbo
      Run parity tests:
      ```bash
      pytest tests/test_elementary_functions.py
      ```
5.  **Documentation**: Update `API_REFERENCE.md` with the new operator.
