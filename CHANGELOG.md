# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
