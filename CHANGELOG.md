# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.4] - 2026-05-24

### Added
- **Thread-Local Index Pools**: Implemented thread-local index pools to enhance concurrent performance and stability.
- **Dense Multiplication Tables**: Added multi-dimensional dense multiplication tables for performance optimizations.

### Changed
- **COSY Backend Thread-Safety**: Refactored the COSY backend to be strictly thread-safe.
- **Directory Swapping Removal**: Eliminated `os.chdir` directory swapping logic to prevent global state side effects.
- **Map Operations Optimization**: Optimized map inversion and composition operations.
- **Logging Strategy**: Implemented a seamless fallback logging strategy.
- **Code Quality**: Applied standard `ruff` formatting fixes and added comprehensive `mypy` type annotations for enhanced maintainability.

### Fixed
- **DAINI.DAT PID Lock**: Fixed the PID locking issue associated with `DAINI.DAT` which previously caused resource conflicts.

## [0.1.3] - 2026-05-24

### Added
- **`CosyDA.transfer_ownership()`**: A new method on the Fortran DA wrapper that atomically marks the object as non-owning and returns its raw index, replacing the error-prone manual `owned = False` pattern throughout `cosy_backend.py`.
- **`_INIT_LOCK` (module-level `threading.RLock`)** in `taylor_function.py`: Guards all class-state mutations in `initialize_mtf` against concurrent calls from multiple threads.
- **`_KERNEL_ETOL`** constant in `numba_kernels.py`: A named, module-level tolerance (`1e-16`) that documents and centralises the early-exit threshold used in the `dense_mul` Numba kernel.
- **`Array`, `Shape`, `DType` type aliases** in `backend.py`: Formal `Union`-based type aliases used across all backend method signatures.
- **`@overload` signatures for `get_backend`**: Enables `mypy` to narrow the return type (`type[NumpyBackend]` or `type[TorchBackend]`) at each call site.

### Changed
- **`TorchBackend.to_numpy`**: Now safely handles CUDA/MPS tensors and autograd-tracked tensors (both previously caused `RuntimeError`). The method now detaches from the computation graph and moves the tensor to CPU before conversion, emitting `UserWarning` at each step.
- **`TorchBackend.prod(axis=None)`**: Fixed `TypeError` crash when `axis=None`. Now dispatches to `torch.prod(a)` (no `dim` argument) for a global reduction, matching `np.prod` semantics.
- **`TorchBackend.atleast_2d`**: Fixed incorrect shape for 0-D scalar tensors. A 0-D tensor now produces shape `(1, 1)` (matching `np.atleast_2d`) instead of the previous `(1,)`.
- **`from_numpy` semantics**: Both `NumpyBackend` and `TorchBackend` now accept an explicit `copy: bool = True` parameter. The default (`copy=True`) prevents silent buffer sharing — `TorchBackend.from_numpy` previously returned a zero-copy view via `torch.from_numpy`, which could silently corrupt the source NumPy array on in-place mutations.
- **`get_backend` return type**: Now returns the backend **class** (a singleton type) rather than a freshly constructed instance, eliminating per-call object allocation overhead. All backend methods are `@staticmethod`, so the class is the correct callable.
- **`CosyIndexPool`**: `acquire` and `release` are now wrapped in a `threading.RLock`, preventing duplicate-index allocation race conditions when multiple threads access the pool concurrently.
- **`CosyMtfData._create_res`**: Uses `transfer_ownership()` instead of `res_da.owned = False`.
- **`CosyMtfData.inverse()` and `divide()`**: Zero-constant checks changed from exact `== 0` to tolerant `abs(c0) < 1e-14` comparisons.
- **`initialize_mtf` operator dispatch**: Removed the previous class-level operator monkey-patching (`cls.__add__ = cls._add_cosy`, etc.) that mutated the global class at initialization time. Dispatch is now done via an explicit `if self._IMPLEMENTATION == "cosy"` inside each dunder method — semantically equivalent but safe from concurrent class mutation.
- **`neval` Numba fast-path**: Eliminated two unnecessary `backend.from_numpy()` calls that allocated tensor copies of `self.coeffs` and `self.exponents` before the Numba path, even though Numba reads those arrays directly.
- **`neval` exponent indexing**: Replaced the `exponents[np.newaxis, :, d]` compound index with a portable `exponents[:, d].reshape(1, -1)` that works identically for both NumPy and PyTorch tensors.
- **`backend.py`**: Fully rewritten with PEP 484 type annotations on every method.

### Fixed
- **Bare `except:` in `_truediv_python`**: Changed to `except (TypeError, ValueError)` so `KeyboardInterrupt`, `SystemExit`, and `MemoryError` are no longer silently swallowed.
- **Dead `pass` blocks in `_mul_cosy`**: Removed three consecutive empty branches that were left over from an incomplete refactor.
- **Dead `prange` loop in `compose_dense_kernel`**: Removed the empty `for i in prange(n_outer_terms): pass` loop that preceded the actual chunked parallel loop, eliminating spurious parallel scheduling overhead.
- **Missing `chunk_size` in `compose_dense_kernel`**: Restored the `chunk_size` calculation that was accidentally omitted.

## [0.1.2] - 2026-05-23


### Added
- **Local Pre-commit hooks**: Quality checks using `pre-commit` (Ruff and MyPy).
- **Read the Docs integration**: Integrated automated Sphinx builds with Jupyter notebook execution.
- **Continuous Integration**: Added GitHub Actions workflow (`test.yml`) verifying the package on Windows/Ubuntu for Python 3.9-3.13.
- **Continuous Delivery**: Added GitHub Actions publish workflow (`python-publish.yml`) to build wheels using `cibuildwheel` and publish to PyPI.

### Fixed
- **Mypy Type Checking**: Resolved numpy typing overload issue in `np.append` for `TaylorMap`.
- **WSL/Linux Memory Limits**: Added Linux-specific `COSY_LMEM_LINUX` environment override to prevent page-mapping errors under the WSL VM.

## [0.2.0] - 2026-01-21

### Added
- **Robust Memory Pooling**: Implemented `CosyIndexPool` (Object Pool Pattern) for O(1) variable allocation.
- **Map Composition Optimization**: Implemented Numba JIT kernels for map composition, resulting in 3x-5x speedup.
- **Architectural Documentation**: New sections in `cosy_backend.rst` and `optimization.rst` detailing memory management and composition strategies.
- **Performance Configuration**: Support for `SANDALWOOD_COSY_POOL_SIZE` for fine-tuning memory usage.

### Fixed
- **Memory Leaks**: Resolved "Split Brain" leaks in COSY batch operations by implementing strict conditional allocation.
- **Numerical Hygiene**: Implemented `DA_RESET` (Hard Reset) to ensure numerical stability across recycled indices.
- **Mixed-Mode Arithmetic**: Fixed backend crashes by ensuring strict type promotion (Real -> Complex) in variable wrappers.

## [0.1.1] - 2026-01-18

### Fixed
- **Numpy Compatibility**: Implemented `__array_ufunc__` fallback for `MultivariateTaylorFunction` to support element-wise operations on object arrays (e.g., `np.vectorize` path) when COSY batch optimization is not applicable.

## [0.1.0] - 2026-01-09

### Added
- **High-Performance Python Backend**: Implemented "Dense Mode" with $O(1)$ lookup tables for monomial multiplication.
- **Numba Acceleration**: JIT-compiled kernels for multiplication (Map-Reduce) and evaluation (Power Caching), achieving 5-10x speedup.
- **Documentation**: New `docs/optimization.rst` detailing the architecture.
- **Benchmarking Suite**: Added benchmarks confirming sub-50ms multiplication times for order 12.

### Changed
- **COSY Backend**: Robustified complex number serialization with fallback logic.
- **Testing**: Fixed all skipped tests in linear algebra and serialization. Optimized demo tests to run in <3s.
- **Warnings**: Suppressed benign deprecation warnings.

## [0.0.1] - 2026-01-05

### Changed
- Re-versioned project to 0.0.1.
- Updated build system to ensure COSY backend is rebuilt on installation.

## [1.5.2] - 2025-12-14

### Added
- Added `py.typed` marker to package distribution to support type checkers (mypy).

### Changed
- C++ Backend Optimization: Implemented memory pooling and Structure-of-Arrays (SoA) layout for Biot-Savart calculations, resulting in ~25% performance improvement.
- Implemented `subtract_inplace` in `MtfData` C++ class.

## [1.5.1] - 2025-09-26

### Changed
- Refactored `taylor_function.py` to inline several internal helper functions, improving code clarity.
- Generalized the `__pow__` method in `MultivariateTaylorFunction` to support all negative integer exponents.
- Refactored `elementary_functions.py` to use a new helper function, `_create_composed_taylor_from_coeffs`, significantly reducing code duplication.
- Made the `_integrate` function thread-safe by removing the modification of the global `_MAX_ORDER` state.

### Fixed
- Corrected a test for the `__pow__` method to align with its new, more general behavior for negative exponents.

## [1.5.0] - 2025-09-13

### BREAKING CHANGES
- This is a major release with significant improvements and breaking changes. It is NOT backward compatible with previous versions.

### Added
- New features and major performance improvements.

### Changed
- General bug fixes.

## [1.4.3] - 2025-09-12

### Changed
- General bug fixes and performance improvements.

## [1.4.0] - 2025-09-11

### Changed
- **Note:** This is a version downgrade from 2.0.1, as requested.
- General bug fixes and performance improvements.

## [2.0.1] - 2024-08-31

### Added
- `pyproject.toml` with full project metadata for modern packaging.
- `LICENSE` file with the MIT License.
- `CHANGELOG.md` to track changes.

### Changed
- Updated `README.md` with clear installation, usage, and testing instructions.
- Set project version to `2.0.1`.
