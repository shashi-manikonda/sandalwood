# Agent Rules: Sandalwood Core

These rules govern the development of the **Sandalwood** MTF library to ensure performance and memory safety within the COSY backend.

## 🧠 Memory & Safety

*   **Memory Pooling**: Strictly enforce the use of `CosyIndexPool` when acquiring indices for temporary COSY variables. Avoid direct index manipulation outside the pool logic.
*   **Scope Management**: Use `CosyScope` (context manager) for any complex sequence of COSY operations. This ensures predictable O(1) "rewinding" of the Fortran stack and prevents leaks.
*   **Windows Memory Limits**: The Windows PE/NTCOFF format has a **2GB limit** for static sections. When modifying `cosy_config.env`, ensure that `LVAR` and `LMEM` values do not exceed this limit collectively. Use `_WIN32` suffixes for platform-specific overrides to maintain high limits on Linux.
*   **Thread Safety**: The COSY backend is **single-threaded**. Never invoke `CosyDA` or `CosyCDA` operations within a `threading.Thread` or `multiprocessing` worker without a global lock.

## 🚀 Performance & Backends

*   **Backend Parity**: Any new mathematical operator or elementary function implemented in the Python/Numba backend (`MtfData`) MUST have a corresponding implementation in the COSY backend (`CosyMtfData`).
*   **Vectorization**: When implementing Python kernels, always use `@numba.njit` with `fastmath=True` and `parallel=True` where appropriate. Use the established **Map-Reduce** patterns for multiplications.
*   **Dispatch Hygiene**: Hot arithmetic loops (Add, Mul) must use the class-level method binding established in `initialize_mtf` to avoid repeated implementation checks.

## 🧹 Quality Control

*   **Numerical Stability**: New elementary functions must be verified against high-precision analytical values (or `mpmath`).
*   **Numba JIT Signatures**: Prefer using explicit type signatures in `@numba.njit` for core kernels. This improves compile-time error reporting and ensures SIMD alignment.
*   **Public API Hygiene**: Every public-facing method in the `mtf` module MUST have a Google-style docstring including at least one `Example` block showing standard usage.
*   **Versioning**: Modifying the Fortran `wrapper.f` or `cosy_config.env` requires a version bump and specialized verification of binary compatibility across platforms.
