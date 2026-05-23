# sandalwood/__init__.py
"""
sandalwood: A Python Library for Multivariate Taylor Functions
===========================================================

This library provides a robust framework for working with multivariate
Taylor series expansions, based on the principles of Differential Algebra (DA).
It allows for the creation, manipulation, and analysis of functions
represented by their Taylor series, supporting both real and complex
coefficients.

Core Features:
- **MultivariateTaylorFunction**: The fundamental class for representing
  a function as a DA vector of its Taylor coefficients.
- **ComplexMultivariateTaylorFunction**: An extension for functions with
  complex coefficients.
- **TaylorMap**: Represents vector-valued functions (maps from R^n to R^m)
  for coordinate transformations and systems of equations.
- **Elementary Functions**: A rich set of elementary functions (sin, cos, exp,
  log, etc.) defined for Taylor series objects.
- **Differential Operators**: `derivative` and `integrate` functions that
  act as the derivation operators of the Differential Algebra.

Example:
    >>> from sandalwood import mtf
    >>>
    >>> # It is crucial to initialize the library's global settings first.
    >>> mtf.initialize_mtf(max_order=4, max_dimension=2)
    >>>
    >>> # Create variables x and y
    >>> x = mtf.var(1)
    >>> y = mtf.var(2)
    >>>
    >>> # Create a function f(x, y) = sin(x + y)
    >>> f = mtf.sin(x + y)
    >>>
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      1    (0, 1)
    1  1.000000e+00      1    (1, 0)
    2 -1.666667e-01      3    (0, 3)
    3 -5.000000e-01      3    (1, 2)
    4 -5.000000e-01      3    (2, 1)
    5 -1.666667e-01      3    (3, 0)

"""

from importlib.metadata import PackageNotFoundError, version

from .complex_taylor_function import (
    ComplexMultivariateTaylorFunction,
    cmtf,
)
from .taylor_function import (
    MultivariateTaylorFunction,
    mtf,
)
from .taylor_map import TaylorMap

try:
    __version__ = version("sandalwood")
except PackageNotFoundError:
    __version__ = "unknown"


# Defines the public API for the library.
__all__ = [
    # Core Classes
    "MultivariateTaylorFunction",
    "ComplexMultivariateTaylorFunction",
    "TaylorMap",
    "mtf",
    "cmtf",
    # Utility Functions
]
