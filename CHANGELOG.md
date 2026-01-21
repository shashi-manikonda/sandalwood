# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
