# mtflib: Multivariate Taylor Function Library

[![Documentation Status](https://readthedocs.org/projects/mtflibrary/badge/?version=latest)](https://mtflibrary.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for creating, manipulating, and composing Multivariate Taylor Functions (MTF/mtf), with a C++ backend for performance-critical applications.

## Installation

The recommended way to install `mtflib` is from PyPI:

```bash
uv pip install mtflib
```

### Installation from Source

Alternatively, you can install `mtflib` directly from the source repository using uv (recommended) or pip. Ensure you have a C++17 compliant compiler (e.g., GCC, Clang, MSVC) for building the backend extensions.

```bash
uv pip install .
```

## Quick Start

Here's a simple example to get you started with `mtflib`:

```python
import numpy as np
from mtflib import mtf
from IPython.display import display

# 1. Initialize global settings (optional but recommended for non-default values)
# If skipped, defaults to max_order=4, max_dimension=3.
mtf.initialize_mtf(max_order=5, max_dimension=2)

# 2. Define symbolic variables
# var(1) corresponds to x, var(2) to y
x = mtf.var(1)
y = mtf.var(2)

# 3. Create a Taylor series expression
# This creates a Taylor series for sin(x) + y^2
f = mtf.sin(x) + y**2

# 4. Evaluate the result at a point
# Let's evaluate f at (x=0.5, y=2.0)
eval_point = np.array([0.5, 2.0])
result = f.eval(eval_point)

print(f"\nf(x, y) = sin(x) + y^2")
print(f"Result of f(0.5, 2.0): {result[0]}")

# For comparison, the exact value is sin(0.5) + 2.0^2
exact_value = np.sin(0.5) + 4.0
print(f"Exact value: {exact_value}")

# You can also view the Taylor series coefficients
print("\nTaylor Series Representation:")
print(f)

print("Symbolic representation of the function:")
display(f.symprint())  # This will print the series in a human-readable format
```
output:
```
Initializing MTF globals with: _MAX_ORDER=5, _MAX_DIMENSION=2
Loading/Precomputing Taylor coefficients up to order 5
Global precomputed coefficients loading/generation complete.
Size of precomputed_coefficients dictionary in memory: 464 bytes, 0.45 KB, 0.00 MB
MTF globals initialized: _MAX_ORDER=5, _MAX_DIMENSION=2, _INITIALIZED=True
Max coefficient count (order=5, nvars=2): 21
Precomputed coefficients loaded and ready for use.

f(x, y) = sin(x) + y^2
Result of f(0.5, 2.0): 4.479427083333333
Exact value: 4.479425538604203

Taylor Series Representation:
          Coefficient  Order Exponents
0  1.000000000000e+00      1    (1, 0)
1  1.000000000000e+00      2    (0, 2)
2 -1.666666666667e-01      3    (3, 0)
3  8.333333333333e-03      5    (5, 0)

Symbolic representation of the function:
```
$\displaystyle 0.00833333 x^{5} - 0.166667 x^{3} + 1.0 x + 1.0 y^{2}$


## JSON Serialization

`mtflib` supports serializing MTF objects to JSON format, preserving all coefficients and properties (including complex values).

```python
# Serialize to JSON string
json_str = f.to_json()

# Deserialize back to object
f_loaded = mtf.from_json(json_str)
```

## Running Tests

The project uses `pytest` for testing. First, install the test dependencies:

```bash
uv pip install -e .[test]
```

Then, run the test suite from the root of the repository:

```bash
pytest
```
