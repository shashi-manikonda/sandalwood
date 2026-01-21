# sandalwood/taylor_function.py
"""
Core implementation of the MultivariateTaylorFunction class.

This module defines the `MultivariateTaylorFunction` class, which is the
fundamental data structure for representing and manipulating multivariate
Taylor series expansions. It serves as an implementation of a Differential
Algebra (DA) vector, and the operations defined on it form a Truncated
Power Series Algebra (TPSA).

See Also
--------
:ref:`optimization` : For details on the internal Dense Mode architecture and JIT optimizations.
"""

import cmath
import json
import math
import numbers
from collections import defaultdict
from functools import reduce
from typing import Optional

import numpy as np
import pandas as pd

from .backend import get_backend

# COSY Backend availability
try:
    from .backends.cosy import cosy_backend

    if hasattr(cosy_backend, "COSY_AVAILABLE"):
        _COSY_BACKEND_AVAILABLE = cosy_backend.COSY_AVAILABLE
    else:
        # Fallback for legacy behavior
        _COSY_BACKEND_AVAILABLE = (
            cosy_backend.libcosy.__class__.__name__ != "DummyLib"
        )
except Exception:
    _COSY_BACKEND_AVAILABLE = False

# Numba availability
try:
    from . import numba_kernels
    if numba_kernels._NUMBA_AVAILABLE:
        _NUMBA_AVAILABLE = True
    else:
        _NUMBA_AVAILABLE = False
except ImportError:
    _NUMBA_AVAILABLE = False


def _generate_exponent(order, var_index, dimension):
    """
    Generates an exponent tuple for a single-variable monomial term.

    This helper function is used to create a standard basis exponent vector,
    representing a term like `x_i^k`.

    Parameters
    ----------
    order : int
        The order of the monomial (the exponent value).
    var_index : int
        The 0-based index of the variable.
    dimension : int
        The total dimension of the exponent space.

    Returns
    -------
    tuple
        An exponent tuple, e.g., `(0, k, 0, ...)` for `var_index=1`.
    """
    exponent = [0] * dimension
    exponent[var_index] = order
    return tuple(exponent)


class MultivariateTaylorFunction:
    """
    A class to represent and manipulate multivariate Taylor series expansions.

    This class provides an object-oriented implementation of a Differential
    Algebra (DA) vector. It stores the Taylor coefficients of a function
    up to a specified order and provides a rich set of methods for
    arithmetic, composition, and analysis. The overloaded operators
    (+, *, etc.) form a Truncated Power Series Algebra (TPSA).

    Attributes
    ----------
    exponents : np.ndarray
        A 2D numpy array of shape `(n_terms, dimension)`, where each row
        represents the multi-index exponent of a term.
    coeffs : np.ndarray
        A 1D numpy array of shape `(n_terms,)` containing the coefficient
        for each corresponding exponent.
    dimension : int
        The number of variables in the Taylor series.
    var_name : str, optional
        An optional name for the function, often used for debugging or
        representation purposes.

    Class Attributes
    ----------------
    _MAX_ORDER : int
        Global setting for the maximum order of Taylor expansions. Set via
        `initialize_mtf`.
    _MAX_DIMENSION : int
        Global setting for the maximum number of variables. Set via
        `initialize_mtf`.
    _ETOL : float
        Global tolerance for floating-point comparisons. Coefficients
        smaller than this value are considered zero.
    _TRUNCATE_AFTER_OPERATION : bool
        If True, coefficients smaller than `_ETOL` are automatically
        removed after each operation.

    Examples
    --------
    >>> import numpy as np
    >>> from sandalwood import mtf
    >>>
    >>> # Initialize global settings for mtf
    >>> mtf.initialize_mtf(max_order=5, max_dimension=2)
    >>>
    >>> # Create a constant function f(x1, x2) = 2.0
    >>> f = mtf.from_constant(2.0, dimension=2)
    >>>
    >>> # Create a variable x1
    >>> x1 = mtf.var(1, 2)
    >>>
    >>> # Create a function g(x1, x2) = 2.0 + x1
    >>> g = f + x1
    >>>
    >>> # Evaluate g at (x1, x2) = (3.0, 4.0)
    >>> result = g.eval([3.0, 4.0])
    >>> print(result)
    [5.]
    """

    _MAX_ORDER = None
    _MAX_DIMENSION = None
    _INITIALIZED = False
    _ETOL = 1e-16
    _TRUNCATE_AFTER_OPERATION = True
    _IMPLEMENTATION = "python"
    _EXP_TO_IDX = None
    _IDX_TO_EXP = None
    _MULT_TABLE = None

    @classmethod
    def initialize_mtf(cls, max_order=None, max_dimension=None, implementation="cosy"):
        """
        Initializes global settings for the sandalwood library.

        This method must be called once at the beginning of a program before
        creating or manipulating any `MultivariateTaylorFunction` objects. It
        sets the global maximum order and dimension for all subsequent
        Taylor series operations.

        If the library is already initialized, this method will do nothing if
        called with the same `max_order` and `max_dimension`. However, attempting
        to re-initialize with different settings will raise a RuntimeError.

        Parameters
        ----------
        max_order : int, optional
            The default maximum order for Taylor series expansions.
        max_dimension : int, optional
            The default maximum number of variables for functions.
        implementation : {'python', 'cosy'}, optional
            The backend implementation to use for core operations. Defaults
            to 'cosy' if available, otherwise falls back to 'python'.

        Examples
        --------
        >>> from sandalwood import mtf
        >>> # Initialize for problems up to order 10 in 5 variables.
        >>> mtf.initialize_mtf(max_order=10, max_dimension=5)
        Initializing mtf globals with: _MAX_ORDER=10, _MAX_DIMENSION=5
        Loading/Precomputing Taylor coefficients up to order 10
        Global precomputed coefficients loading/generation complete.
        Size of precomputed_coefficients dictionary in memory: ...
        mtf globals initialized: _MAX_ORDER=10, _MAX_DIMENSION=5, _INITIALIZED=True
        Max coefficient count (order=10, nvars=5): 3003
        Precomputed coefficients loaded and ready for use.

        >>> # This will raise an error because initialization with new settings is
        >>> # not allowed
        >>> mtf.initialize_mtf(max_order=12, max_dimension=5)
        Traceback (most recent call last):
        ...
        RuntimeError: MTF Globals are already initialized with different settings.
        Re-initialization with different max_order or max_dimension is not allowed.
        """
        if (not cls._INITIALIZED) or (
            cls._INITIALIZED
            and cls._MAX_ORDER == max_order
            and cls._MAX_DIMENSION == max_dimension
        ):
            if max_order is not None:
                if not isinstance(max_order, int) or max_order <= 0:
                    raise ValueError("max_order must be a positive integer.")
                cls._MAX_ORDER = max_order
            if max_dimension is not None:
                if not isinstance(max_dimension, int) or max_dimension <= 0:
                    raise ValueError("max_dimension must be a positive integer.")
                cls._MAX_DIMENSION = max_dimension

            # Default to proposed implementation, but handle "auto" behavior if needed
            # User passed 'cosy' explicitly or implicit default?
            # The signature says default="cpp" (wait, actually default was "python" in previous).
            # We want default="cosy".

            # Logic:
            # 1. If implementation is explicitly "python", use python.
            # 2. If implementation is "cosy" (default), try cosy.
            #    If not available, warn and fallback to python.

            if implementation == "python":
                cls._IMPLEMENTATION = "python"
            else:
                # Default is now COSY or user asked for COSY
                if _COSY_BACKEND_AVAILABLE:
                    cls._IMPLEMENTATION = "cosy"
                else:
                    if implementation == "cosy":
                        print(
                            "Warning: COSY backend requested but not available. Falling back to Python."
                        )
                    cls._IMPLEMENTATION = "python"

            print(
                f"Initializing MTF globals with: _MAX_ORDER={cls._MAX_ORDER}, "
                f"_MAX_DIMENSION={cls._MAX_DIMENSION} using {cls._IMPLEMENTATION} backend"
            )

            if cls._IMPLEMENTATION == "cosy":
                print("Initializing COSY backend...")
                cosy_backend.CosyBackendManager.initialize(
                    cls._MAX_ORDER, cls._MAX_DIMENSION
                )

            cls._INITIALIZED = True
            print(
                f"MTF globals initialized: _MAX_ORDER={cls._MAX_ORDER}, "
                f"_MAX_DIMENSION={cls._MAX_DIMENSION}, _INITIALIZED={cls._INITIALIZED}"
            )
            print(
                f"Max coefficient count (order={cls._MAX_ORDER}, "
                f"nvars={cls._MAX_DIMENSION}): {cls.get_max_coefficient_count()}"
            )
        else:
            raise RuntimeError(
                "Re-initialization with different max_order or max_dimension is "
            )
        
        if cls._IMPLEMENTATION == "python":
            cls._precompute_tables()
            
        # --- Fast-Path Method Binding ---
        # Bind the correct implementation of arithmetic operators to the dunder methods
        # to avoid if/else checks in the hot path.
        if cls._IMPLEMENTATION == "cosy":
            cls.__add__ = cls._add_cosy
            cls.__sub__ = cls._sub_cosy
            cls.__mul__ = cls._mul_cosy
            cls.__truediv__ = cls._truediv_cosy
            # Also bind reverse operators if needed, or rely on them calling forward
            # __radd__ usually calls self + other, so it inherits the bound __add__
        else:
            cls.__add__ = cls._add_python
            cls.__sub__ = cls._sub_python
            cls.__mul__ = cls._mul_python
            cls.__truediv__ = cls._truediv_python

    @classmethod
    def _batch_add(cls, mtfs):
        """
        Efficiently adds a list of MultivariateTaylorFunction objects.
        
        Args:
            mtfs: A list of MultivariateTaylorFunction objects.
            
        Returns:
            MultivariateTaylorFunction: The sum.
        """
        if not mtfs:
            return cls.from_constant(0.0)
        
        if len(mtfs) == 1:
            return mtfs[0]
            
        # Dispatch to implementation
        if cls._IMPLEMENTATION == "cosy":
            return cls._batch_add_cosy(mtfs)
        else:
            return cls._batch_add_python(mtfs)

    @classmethod
    def _batch_add_cosy(cls, mtfs):
        """
        Efficiently adds a list of MultivariateTaylorFunction objects using COSY batch kernel.
        """
        if not mtfs:
            return cls.from_constant(0.0)

        da_list = []
        for i, m in enumerate(mtfs):
            if isinstance(m, cls) and hasattr(m, "mtf_data") and m.mtf_data is not None:
                 da_list.append(m.mtf_data)
            elif hasattr(m, "mtf_data") and m.mtf_data is not None:
                 # Handle case where isinstance fails (e.g. reload) but supports protocol
                 da_list.append(m.mtf_data)
            elif isinstance(m, cls):
                 # Has class but no data? Likely Python backend MTF mixed in.
                 # Convert to COSY using constructor which handles backend sync
                 # Use efficient tuple constructor
                 new_m = cls(
                     coefficients=(m.exponents, m.coeffs), 
                     dimension=m.dimension
                 )
                 if new_m.mtf_data is None:
                     # Should not happen if cls is COSY
                     raise RuntimeError("Failed to create COSY MTF data during conversion.")
                 da_list.append(new_m.mtf_data)
            else:
                 # Scalar or compatible
                 da_list.append(m)

        from .backends.cosy.cosy_backend import CosyBackend
        
        # Use linear combination with all 1.0 coeffs
        coeffs = [1.0] * len(da_list)
        res_da = CosyBackend.linear_combination(coeffs, da_list)
        
        # Create result MTF wrapper
        from .backends.cosy.cosy_backend import CosyMtfData
        
        # Check complexity
        is_complex = getattr(res_da, "is_complex", False)
        
        # Get dimension from first MTF
        ref_dim = cls.get_max_dimension()
        if hasattr(mtfs[0], "dimension"):
            ref_dim = mtfs[0].dimension
            
        # Transfer ownership to prevent double-free
        if hasattr(res_da, "owned"):
            res_da.owned = False

        data = CosyMtfData(idx=res_da.idx, owned=True, is_complex=is_complex, dimension=ref_dim)
        
        new_mtf = cls.__new__(cls)
        new_mtf.dimension = ref_dim
        new_mtf.mtf_data = data
        new_mtf._IMPLEMENTATION = "cosy"
        
        # Initialize storage as None (lazy loading)
        new_mtf.var_name = None
        new_mtf._exponents = None
        new_mtf._coeffs = None
        new_mtf._indices = None
        new_mtf._dense_coeffs = None
        
        return new_mtf

    @classmethod
    def _batch_add_python(cls, mtfs):
        # Collect all dictionaries
        is_complex = False
        sample_mtf = mtfs[0]
        dimension = sample_mtf.dimension
        
        # Check complexity
        for m in mtfs:
             if m.coeffs.dtype == np.complex128 or np.iscomplexobj(m.coeffs):
                 is_complex = True
                 break
        
        # Merge using dictionary optimization
        summed_coeffs_dict = defaultdict(complex if is_complex else float)
        
        for m in mtfs:
            # We can iterate over exponents and coeffs
            # Using direct array access is faster
            n_terms = len(m.coeffs)
            if n_terms == 0:
                continue
                
            exps = m.exponents
            cs = m.coeffs
            
            # Map exponents to tuple for dictionary key
            # This loop is still Python overhead but avoids object creation overhead of pairwise add
            for i in range(n_terms):
                exp_tuple = tuple(exps[i])
                summed_coeffs_dict[exp_tuple] += cs[i]

        if not summed_coeffs_dict:
             return cls(coefficients={}, dimension=dimension)

        unique_exponents = np.array(list(summed_coeffs_dict.keys()), dtype=np.int32)
        summed_coeffs = np.array(list(summed_coeffs_dict.values()), dtype=np.complex128 if is_complex else np.float64)
        
        res = cls((unique_exponents, summed_coeffs), dimension=dimension)
        if cls._TRUNCATE_AFTER_OPERATION:
             res._cleanup_after_operation()
        return res


    @classmethod
    def _precompute_tables(cls):
        """
        Precomputes exponent mapping and multiplication tables for dense mode.
        """
        if cls._MAX_ORDER is None or cls._MAX_DIMENSION is None:
            return

        print("Precomputing multiplication tables for Dense Mode...")

        # 1. Generate all valid exponents
        exponents = []
        
        # Helper to generate terms with sum <= max_order
        def generate_exponents(dim, current_order, current_exp):
            if dim == 0:
                exponents.append(tuple(current_exp))
                return

            if dim == 1:
                # Can range from 0 to (MAX_ORDER - current_order)
                for i in range(cls._MAX_ORDER - current_order + 1):
                    exponents.append(tuple(current_exp + [i]))
                return

            for i in range(cls._MAX_ORDER - current_order + 1):
                generate_exponents(dim - 1, current_order + i, current_exp + [i])

        generate_exponents(cls._MAX_DIMENSION, 0, [])
        
        # Sort exponents (total order, then lex)
        exponents.sort(key=lambda x: (sum(x), x))
        
        cls._IDX_TO_EXP = np.array(exponents, dtype=np.int32)
        cls._EXP_TO_IDX = {exp: i for i, exp in enumerate(exponents)}
        
        n_terms = len(exponents)
        
        # 2. Build Multiplication Table
        # shape: (n_terms, n_terms)
        # value: index of result, or -1 if truncated
        cls._MULT_TABLE = np.full((n_terms, n_terms), -1, dtype=np.int32)
        
        exps_arr = cls._IDX_TO_EXP # (N, D)
        
        # Sum of exponents: (N, 1, D) + (1, N, D) -> (N, N, D)
        sum_exps = exps_arr[:, np.newaxis, :] + exps_arr[np.newaxis, :, :]
        
        # Check orders: (N, N)
        orders = np.sum(sum_exps, axis=2)
        valid_mask = orders <= cls._MAX_ORDER
        
        # Fill table
        for i in range(n_terms):
            for j in range(n_terms):
                if valid_mask[i, j]:
                     tup = tuple(sum_exps[i, j])
                     cls._MULT_TABLE[i, j] = cls._EXP_TO_IDX[tup]
        
        print(f"Dense Mode tables ready. {n_terms} terms.")

    @classmethod
    def _auto_initialize(cls):
        """Auto-initializes the library with defaults if not already initialized."""
        if not cls._INITIALIZED:
            print(
                "Warning: sandalwood not initialized. Auto-initializing with defaults "
                "(Order=4, Dimension=3)."
            )
            cls.initialize_mtf(max_order=4, max_dimension=3)

    @classmethod
    def get_max_coefficient_count(cls, max_order=None, max_dimension=None):
        """Calculates max coefficient count for given order/dimension."""
        if not cls._INITIALIZED and max_order is None and max_dimension is None:
            cls._auto_initialize()

        effective_max_order = max_order if max_order is not None else cls._MAX_ORDER
        effective_max_dimension = (
            max_dimension if max_dimension is not None else cls._MAX_DIMENSION
        )
        if effective_max_order is None or effective_max_dimension is None:
            raise ValueError(
                "Global max_order or max_dimension not initialized and no defaults "
                "provided."
            )
        return math.comb(
            effective_max_order + effective_max_dimension,
            effective_max_dimension,
        )

    @classmethod
    def get_precomputed_coefficients(cls):
        """Returns the precomputed Taylor coefficients for elementary functions."""
        if not cls._INITIALIZED:
            cls._auto_initialize()
        return cls._PRECOMPUTED_COEFFICIENTS

    @classmethod
    def get_mtf_initialized_status(cls):
        """Returns initialization status of MTF globals."""
        return cls._INITIALIZED

    @classmethod
    def set_max_order(cls, order):
        """Sets the global maximum order for Taylor series."""
        if not cls._INITIALIZED:
            cls._auto_initialize()
        if not isinstance(order, int) or order < 0:
            raise ValueError("Order must be a non-negative integer.")
        cls._MAX_ORDER = order

    @classmethod
    def get_max_order(cls):
        """Returns the global maximum order for Taylor series."""
        if not cls._INITIALIZED:
            cls._auto_initialize()
        return cls._MAX_ORDER

    @classmethod
    def get_max_dimension(cls):
        """Returns the global maximum dimension (number of variables)."""
        if not cls._INITIALIZED:
            cls._auto_initialize()
        return cls._MAX_DIMENSION

    @classmethod
    def set_etol(cls, etol):
        """Sets the global error tolerance (etol) for `sandalwood`."""
        if not cls._INITIALIZED:
            cls._auto_initialize()
        if not isinstance(etol, float) or etol <= 0:
            raise ValueError("Error tolerance (etol) must be a positive float.")
        cls._ETOL = etol

    @classmethod
    def get_etol(cls):
        """Returns the global error tolerance (etol)."""
        if not cls._INITIALIZED:
            cls._auto_initialize()
        return cls._ETOL

    @classmethod
    def set_truncate_after_operation(cls, enable: bool):
        """
        Sets the global flag to enable or disable automatic coefficient cleanup.
        """
        if not isinstance(enable, bool):
            raise ValueError("Input 'enable' must be a boolean value (True or False).")
        cls._TRUNCATE_AFTER_OPERATION = enable

    @classmethod
    def reset_mtf(cls):
        """
        Resets the MTF backend. For COSY, this clears all memory.
        """
        if cls._INITIALIZED and cls._IMPLEMENTATION == "cosy":
            cosy_backend.CosyBackend.reset()

    def __init__(self, coefficients=None, dimension=None, var_name=None, mtf_data=None):
        """
        Initializes a MultivariateTaylorFunction object.

        This constructor is flexible and accepts coefficients in two main
        formats: a dictionary mapping exponent tuples to coefficient values, or
        a tuple containing two NumPy arrays (exponents and coefficients).
        It is generally recommended to use the factory methods
        `from_constant` or `from_variable` for creating new instances.

        Parameters
        ----------
        coefficients : dict or tuple
            The coefficients of the Taylor series. This can be either:
            - A dictionary mapping exponent tuples to coefficient values,
              e.g., `{(0, 0): 1.0, (1, 0): 2.0}`.
            - A tuple `(exponents, coeffs)`, where `exponents` is a 2D
              `np.ndarray` of term exponents and `coeffs` is a 1D `np.ndarray`
              of the corresponding coefficients.
        dimension : int, optional
            The number of variables in the function. If not provided, it is
            inferred from the `coefficients` data. It must be provided if
            `coefficients` is an empty dictionary.
        var_name : str, optional
            An optional name for the variable, by default None.
        mtf_data : object, optional
            Internal data object for C++ backend acceleration. Users should
            not set this directly.
        """
        self.var_name = var_name
        self.mtf_data = mtf_data

        # Initialize storage as None (lazy loading)
        self._exponents = None
        self._coeffs = None
        self._indices = None
        self._dense_coeffs = None

        if self.mtf_data:
            # Lazy initialization: don't call to_dict() yet
            if dimension is None:
                if hasattr(self.mtf_data, "dimension"):
                    self.dimension = self.mtf_data.dimension
                else:
                    # Force sync if we can't get dimension otherwise
                    self._ensure_synced()
                    # Infer dimension from synced data
                    if self._exponents.size > 0:
                        self.dimension = self._exponents.shape[1]
                    else:
                        self.dimension = self.get_max_dimension()
            else:
                self.dimension = dimension
            return

        # Fast path for tuple of (exponents, coeffs)
        if (
            isinstance(coefficients, tuple)
            and len(coefficients) == 2
            and isinstance(coefficients[0], np.ndarray)
            and isinstance(coefficients[1], np.ndarray)
        ):
            self._exponents, self._coeffs = coefficients
            if dimension is None:
                self.dimension = (
                    self._exponents.shape[1]
                    if self._exponents.size > 0
                    else self.get_max_dimension()
                )
            else:
                self.dimension = dimension

            if self._exponents.size > 0 and self._exponents.shape[1] != self.dimension:
                raise ValueError(
                    f"Provided dimension {self.dimension} does not match exponent "
                    f"dimension {self._exponents.shape[1]}."
                )
            if (
                self._coeffs.ndim != 1
                or self._coeffs.shape[0] != self._exponents.shape[0]
            ):
                raise ValueError("Coefficients array has incorrect shape.")

        # Path for dictionary
        elif isinstance(coefficients, dict):
            if not coefficients:
                self.dimension = (
                    dimension if dimension is not None else self.get_max_dimension()
                )
                self._exponents = np.empty((0, self.dimension), dtype=np.int32)
                self._coeffs = np.empty((0,), dtype=np.float64)
            else:
                first_exp = next(iter(coefficients.keys()))
                inferred_dim = len(first_exp)
                if dimension is None:
                    self.dimension = inferred_dim
                elif dimension != inferred_dim:
                    raise ValueError(
                        f"Provided dimension {dimension} does not match inferred "
                        f"dimension {inferred_dim} from coefficients."
                    )
                else:
                    self.dimension = dimension

                # Optimized dict conversion
                num_items = len(coefficients)
                self._exponents = np.empty((num_items, self.dimension), dtype=np.int32)
                is_complex = any(np.iscomplexobj(v) for v in coefficients.values())
                dtype = np.complex128 if is_complex else np.float64
                self._coeffs = np.empty(num_items, dtype=dtype)

                for i, (exp, coeff) in enumerate(coefficients.items()):
                    self._exponents[i] = exp
                    if isinstance(coeff, np.ndarray):
                        self._coeffs[i] = coeff.item()
                    else:
                        self._coeffs[i] = coeff

                # Sort both arrays based on exponents
                sorted_indices = np.lexsort(self._exponents.T)
                self._exponents = self._exponents[sorted_indices]
                self._coeffs = self._coeffs[sorted_indices]
        else:
            raise TypeError(
                "Unsupported type for 'coefficients'. Must be a dict or a tuple of "
                "(exponents, coeffs) arrays."
            )

        if _COSY_BACKEND_AVAILABLE and self._IMPLEMENTATION == "cosy":
            is_complex = np.iscomplexobj(self._coeffs)
            self.mtf_data = cosy_backend.CosyMtfData(
                self.dimension, is_complex=is_complex
            )
            self.mtf_data.from_numpy(self._exponents, self._coeffs)

    def _ensure_synced(self):
        """Synchronize Python-side coefficients from backend if needed."""
        if self._exponents is not None:
            return

        if self.mtf_data is None:
            # Should not happen if initialized correctly
            raise RuntimeError(
                "MTF data is missing and coefficients are not initialized."
            )

        data_dict = self.mtf_data.to_dict()
        self._exponents = data_dict["exponents"]
        self._coeffs = data_dict["coeffs"]

        # Apply truncation if enabled, to match Python behavior
        if self._TRUNCATE_AFTER_OPERATION and self._coeffs.size > 0:
            etol = self.get_etol()
            keep_mask = np.abs(self._coeffs) > etol
            if not np.all(keep_mask):
                self._exponents = self._exponents[keep_mask]
                self._coeffs = self._coeffs[keep_mask]

        # Ensure dimensions match (sanity check)
        if hasattr(self, "dimension") and self.dimension is not None:
            if self._exponents.size > 0 and self._exponents.shape[1] != self.dimension:
                # This might happen if backend truncates or changes dimension unexpectedly?
                # Or if self.dimension was set incorrectly.
                # Or if self.dimension was set incorrectly.
                pass

    def _ensure_backend(self):
        """Attempts to reconstruct backend data if it is missing."""
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is None:
            if (
                self._coeffs is not None
                and self._exponents is not None
                and _COSY_BACKEND_AVAILABLE
            ):
                is_complex = np.iscomplexobj(self._coeffs)
                self.mtf_data = cosy_backend.CosyMtfData(
                    self.dimension, is_complex=is_complex
                )
                self.mtf_data.from_numpy(self._exponents, self._coeffs)

    def _materialize_from_dense(self):
        """Converts internal dense array back to sparse representation."""
        if self._dense_coeffs is None:
            return
        # Check tolerance
        mask = np.abs(self._dense_coeffs) > self._ETOL
        indices = np.nonzero(mask)[0]
        
        self._coeffs = self._dense_coeffs[indices]
        self._exponents = self._IDX_TO_EXP[indices]
        # Also cache the indices since we have them!
        self._indices = indices.astype(np.int32)

    @property
    def exponents(self):
        if (self._exponents is None or self._exponents.size == 0) and self._dense_coeffs is not None:
             self._materialize_from_dense()
        self._ensure_synced()
        return self._exponents

    @exponents.setter
    def exponents(self, value):
        self._exponents = value
        self._indices = None
        self._dense_coeffs = None
        # Invalidate backend data since we are modifying Python side manually
        self.mtf_data = None

    @property
    def coeffs(self):
        if (self._coeffs is None or self._coeffs.size == 0) and self._dense_coeffs is not None:
             self._materialize_from_dense()
        self._ensure_synced()
        return self._coeffs

    def _get_indices(self):
        """Cached accessor for dense indices."""
        if not hasattr(self, "_indices") or self._indices is None:
            if self._EXP_TO_IDX is None:
                return None  # Dense mode not active

            # Compute and cache
            idx_list = [self._EXP_TO_IDX.get(tuple(e), -1) for e in self.exponents]
            self._indices = np.array(idx_list, dtype=np.int32)
            
        return self._indices

    @coeffs.setter
    def coeffs(self, value):
        self._coeffs = value
        # Invalidate backend data since we are modifying Python side manually
        self.mtf_data = None

    @classmethod
    def from_constant(cls, constant_value, dimension=None):
        """
        Creates a MultivariateTaylorFunction representing a constant value.

        Parameters
        ----------
        constant_value : float or int
            The constant value of the function.
        dimension : int, optional
            The dimension of the function's domain. If not provided, the
            globally configured `_MAX_DIMENSION` is used.

        Returns
        -------
        MultivariateTaylorFunction
            A new MTF instance representing the constant function.
        """
        if dimension is None:
            dimension = cls.get_max_dimension()
        if isinstance(constant_value, complex):
            coeffs = {(0,) * dimension: constant_value}
        else:
            coeffs = {(0,) * dimension: float(constant_value)}
        return cls(coefficients=coeffs, dimension=dimension)

    @classmethod
    def var(
        cls, var_index: int, dimension: Optional[int] = None
    ) -> "MultivariateTaylorFunction":
        """
        Creates a MultivariateTaylorFunction representing a single variable.

        This is a convenience factory function for creating a single variable.
        The dimension is inferred from the global settings if not provided.

        Parameters
        ----------
        var_index : int
            The 1-based index of the variable to create (1-based).
        dimension : int, optional
            The total number of variables in the function's domain. If None,
            the global `_MAX_DIMENSION` is used.

        Returns
        -------
        MultivariateTaylorFunction
            An mtf object representing the variable `x_i`.

        Raises
        ------
        ValueError
            If `var_index` is not between 1 and `dimension`.
        """
        if dimension is None:
            dimension = cls.get_max_dimension()

        if not (1 <= var_index <= dimension):
            raise ValueError(
                f"Variable index must be between 1 and {dimension}, inclusive."
            )
        exponent = [0] * dimension
        exponent[var_index - 1] = 1
        coeffs = {tuple(exponent): 1.0}
        return cls(coefficients=coeffs, dimension=dimension, var_name=f"x_{var_index}")

    @classmethod
    def from_cosy_indices(cls, indices: np.ndarray, dimension: int) -> np.ndarray:
        """
        Factory: Wraps a numpy array of integer COSY indices into an array of 
        MultivariateTaylorFunction objects in one go.
        
        This avoids the overhead of calling __init__ and checking types for each
        element when we know we have raw low-level indices.
        
        Parameters
        ----------
        indices : np.ndarray
            Array of int32 indices referencing COSY DA objects.
        dimension : int
            The dimension for all created objects.
            
        Returns
        -------
        np.ndarray
            Object array of MultivariateTaylorFunction instances.
        """
        n = len(indices)
        objs = np.empty(n, dtype=object)
        
        # Determine backend availability
        if not _COSY_BACKEND_AVAILABLE or cls._IMPLEMENTATION != "cosy":
             raise RuntimeError("bulk creation from_cosy_indices only valid when COSY backend is active")
             
        from .backends.cosy import cosy_backend
        
        for i in range(n):
            # Bypass __init__ overhead
            obj = cls.__new__(cls)
            # Initialize minimal state
            obj.mtf_data = cosy_backend.CosyMtfData(dimension=dimension, is_complex=False, idx=indices[i], owned=True)
            obj.dimension = dimension
            obj.var_name = None
            obj._exponents = None
            obj._coeffs = None
            obj._indices = None
            obj._dense_coeffs = None
            objs[i] = obj
            
        return objs


    @staticmethod
    def list2pd(mtfs, column_names=None):
        """
        Merges a list of MTFs into a single pandas DataFrame for comparison.

        Each MTF's coefficients are presented in a separate column, making it
        easy to view multiple functions side-by-side.

        Parameters
        ----------
        mtfs : list of MultivariateTaylorFunction
            A list of MTF objects to be merged.
        column_names : list of str, optional
            A list of names for the coefficient columns. If not provided,
            columns are named generically.

        Returns
        -------
        pd.DataFrame
            A DataFrame where each row is a term and each column represents
            the coefficients of one of the input MTFs.

        Raises
        ------
        TypeError
            If the input is not a list of MTF objects.
        ValueError
            If the MTFs in the list have different dimensions.
        """
        if isinstance(mtfs, np.ndarray):
            mtfs = list(mtfs)

        if not isinstance(mtfs, list):
            raise TypeError("Input 'mtfs' must be a list.")

        if not mtfs:
            return pd.DataFrame(columns=["Order", "Exponents"])

        valid_mtf_types = (MultivariateTaylorFunction,)
        for mtf_instance in mtfs:
            if not isinstance(mtf_instance, valid_mtf_types):
                raise TypeError(
                    f"All elements in 'mtfs' must be instances of "
                    f"{MultivariateTaylorFunction.__name__}, but found "
                    f"{type(mtf_instance).__name__}."
                )

        first_dim = mtfs[0].dimension
        for i, mtf_instance in enumerate(mtfs[1:]):
            if mtf_instance.dimension != first_dim:
                raise ValueError(
                    f"mtf at index {i + 1} has dimension {mtf_instance.dimension}, "
                    f"but the first mtf has dimension {first_dim}. All mtfs must have "
                    "the same dimension."
                )

        dfs = []
        for i, mtf_instance in enumerate(mtfs):
            df = mtf_instance.get_tabular_dataframe()
            if column_names and len(column_names) == len(mtfs):
                if "Coefficient" in df.columns:
                    df = df.rename(columns={"Coefficient": f"Coeff_{column_names[i]}"})
            else:
                mtf_name = getattr(mtf_instance, "name", str(i + 1))
                if "Coefficient" in df.columns:
                    df = df.rename(columns={"Coefficient": f"Coefficient_{mtf_name}"})
            dfs.append(df)

        tmap = reduce(
            lambda left, right: pd.merge(
                left, right, on=["Order", "Exponents"], how="outer"
            ),
            dfs,
        )

        coef_cols_initial = [col for col in tmap.columns if col.startswith("Coeff")]
        cols = coef_cols_initial + ["Order", "Exponents"]
        tmap = tmap[cols]
        tmap[coef_cols_initial] = tmap[coef_cols_initial].fillna(0)

        tmap = tmap.sort_values(
            by=["Order", "Exponents"], ascending=[True, False]
        ).reset_index(drop=True)
        return tmap

    @staticmethod
    def to_mtf(input_val, dimension=None):
        """
        Converts input to MultivariateTaylorFunction or
        ComplexMultivariateTaylorFunction.
        """
        if isinstance(input_val, MultivariateTaylorFunction):
            return input_val

        # Standardize dimension
        if dimension is None:
            dimension = MultivariateTaylorFunction.get_max_dimension()

        # Handle NumPy 0-dim array
        if isinstance(input_val, np.ndarray) and input_val.shape == ():
            input_val = input_val.item()

        # Now check types
        if isinstance(input_val, complex) or np.iscomplexobj(input_val):
            from .complex_taylor_function import ComplexMultivariateTaylorFunction

            return ComplexMultivariateTaylorFunction.from_constant(
                input_val, dimension=dimension
            )
        elif isinstance(input_val, (int, float, np.number)):
            return MultivariateTaylorFunction.from_constant(
                float(input_val), dimension=dimension
            )
        else:
            raise TypeError(
                f"Unsupported input type: {type(input_val)}. Cannot convert to "
                f"MTF/CMTF."
            )

    def to_json(self):
        """
        Serializes the MTF object to a JSON string.

        Returns
        -------
        str
            A JSON string representation of the MTF object.
        """
        exponents = self.exponents.tolist()

        is_complex = np.iscomplexobj(self.coeffs)
        if is_complex:
            # Store as [real, imag] pairs
            coeffs = [[c.real, c.imag] for c in self.coeffs]
        else:
            coeffs = self.coeffs.tolist()

        data = {
            "dimension": self.dimension,
            "exponents": exponents,
            "coeffs": coeffs,
            "is_complex": is_complex,
            "var_name": self.var_name,
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str):
        """
        Creates an MTF object from a JSON string.

        Parameters
        ----------
        json_str : str
            A JSON string representation of an MTF object.

        Returns
        -------
        MultivariateTaylorFunction
            The reconstructed MTF object.
        """
        data = json.loads(json_str)

        dimension = data["dimension"]
        exponents = data["exponents"]
        raw_coeffs = data["coeffs"]
        is_complex = data.get("is_complex", False)
        var_name = data.get("var_name", None)

        coeffs_dict = {}

        if is_complex:
            # Reconstruct complex numbers from [real, imag] pairs
            for exp, c_pair in zip(exponents, raw_coeffs):
                coeffs_dict[tuple(exp)] = complex(c_pair[0], c_pair[1])
        else:
            for exp, c in zip(exponents, raw_coeffs):
                coeffs_dict[tuple(exp)] = c

        # If data is complex but we are calling from base class, switch to CMTF
        if is_complex and cls.__name__ == "MultivariateTaylorFunction":
            from .complex_taylor_function import ComplexMultivariateTaylorFunction

            return ComplexMultivariateTaylorFunction(
                coeffs_dict, dimension=dimension, var_name=var_name
            )

        return cls(coeffs_dict, dimension=dimension, var_name=var_name)

    def __call__(self, evaluation_point):
        """
        Evaluates the Taylor function at a given point. Alias for `eval`.

        Parameters
        ----------
        evaluation_point : array_like
            A 1D array or list representing the point at which to evaluate the
            function. Its length must match the function's dimension.

        Returns
        -------
        float or complex
            The result of the evaluation.
        """
        return self.eval(evaluation_point).item()

    def eval(self, evaluation_point):
        """
        Evaluates the Taylor function at a single point.

        This method provides a convenient way to evaluate the function for a
        single input vector. For evaluating multiple points efficiently, use
        the `neval` method.

        Parameters
        ----------
        evaluation_point : array_like
            A 1D array or list representing the point at which to evaluate the
            function. Its length must match the function's dimension.

        Returns
        -------
        np.ndarray
            A 1-element array containing the result of the evaluation.

        Raises
        ------
        ValueError
            If the `evaluation_point` has an incorrect shape or dimension.
        """
        evaluation_point = np.array(evaluation_point)

        # Optimized backend evaluation
        # Optimized backend evaluation
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            # Ensure point is correct shape/type for backend
            # COSY expects list or array of floats.
            if evaluation_point.ndim == 1:
                if evaluation_point.shape[0] != self.dimension:
                    raise ValueError(
                        f"Evaluation point dimension must match MTF dimension ({self.dimension})."
                    )
            elif evaluation_point.ndim == 2:
                if not (
                    evaluation_point.shape[0] == 1
                    and evaluation_point.shape[1] == self.dimension
                ):
                    raise ValueError(
                        "For 2D input, eval() supports only a single evaluation point with shape (1, dimension)."
                    )
            else:
                raise ValueError("Evaluation point must be a 1D or 2D array.")

            return np.array([self.mtf_data.eval(evaluation_point.flatten())])

        if evaluation_point.ndim == 1:
            if evaluation_point.shape[0] != self.dimension:
                raise ValueError(
                    f"Evaluation point dimension must match MTF dimension "
                    f"({self.dimension})."
                )
            evaluation_points = evaluation_point.reshape(1, -1)
            return self.neval(evaluation_points)
        elif evaluation_point.ndim == 2:
            if (
                evaluation_point.shape[0] == 1
                and evaluation_point.shape[1] == self.dimension
            ):
                return self.neval(evaluation_point)
            else:
                raise ValueError(
                    "For 2D input, eval() supports only a single evaluation point with "
                    "shape (1, dimension)."
                )
        else:
            raise ValueError("Evaluation point must be a 1D or 2D array.")

    def neval(self, evaluation_points):
        """
        Evaluates the Taylor function at multiple points in a vectorized manner.

        This method is optimized for performance when evaluating the function
        at a large number of points simultaneously.

        Parameters
        ----------
        evaluation_points : array_like
            A 2D numpy array of shape `(n_points, dimension)`, where each row
            is a point at which to evaluate the function.

        Returns
        -------
        np.ndarray or torch.Tensor
            An array containing the evaluation result for each input point.
            The return type matches the backend used (NumPy or PyTorch).

        Raises
        ------
        ValueError
            If the `evaluation_points` array has an incorrect shape.

        Examples
        --------
        >>> from sandalwood import mtf
        >>> mtf.initialize_mtf(max_order=2, max_dimension=2)
        >>> x, y = mtf.var(1), mtf.var(2)
        >>> f = 1 + x*y
        >>>
        >>> # --- NumPy Backend ---
        >>> import numpy as np
        >>> points_np = np.array([[1, 2], [3, 4], [5, 6]])
        >>> result_np = f.neval(points_np)
        >>> print(type(result_np), result_np)
        <class 'numpy.ndarray'> [ 3. 13. 31.]
        >>>
        >>> # --- PyTorch Backend ---
        >>> import torch
        >>> # Check if torch is available and a GPU is present
        >>> if torch.cuda.is_available():
        ...     device = torch.device("cuda")
        ...     points_torch = torch.tensor([[1, 2], [3, 4], [5, 6]], device=device)
        ...     result_torch = f.neval(points_torch)
        ...     print(type(result_torch), result_torch.device)
        ...     # <class 'torch.Tensor'> cuda:0
        """
        backend = get_backend(evaluation_points)
        evaluation_points = backend.atleast_2d(evaluation_points)
        if evaluation_points.shape[1] != self.dimension:
            raise ValueError(
                f"Evaluation points array must have shape (n_points, {self.dimension})."
            )

        if self.coeffs.size == 0:
            return backend.zeros(evaluation_points.shape[0])

        # Convert coefficients and exponents to the correct tensor type
        coeffs = backend.from_numpy(self.coeffs)
        exponents = backend.from_numpy(self.exponents)

        if _NUMBA_AVAILABLE and isinstance(evaluation_points, np.ndarray):
             # Numba Parallel Evaluation
             # Prepare output array
             results = backend.zeros(evaluation_points.shape[0], dtype=self.coeffs.dtype)
             
             # Ensure types match for Numba (float64 or complex128)
             # Numba is picky about type matching and contiguity
             pts_c = np.ascontiguousarray(evaluation_points)
             exps_c = np.ascontiguousarray(self.exponents).astype(np.int32)
             coeffs_c = np.ascontiguousarray(self.coeffs)
             
             numba_kernels.evaluate_dense_kernel(pts_c, exps_c, coeffs_c, results)
             return results


        # Fallback to Iterative Reduction (NumPy)
        # BATCHING: Process points in chunks to avoid OOM
        BATCH_SIZE = 10000 
        n_points = evaluation_points.shape[0]
        
        if n_points > BATCH_SIZE:
            results = backend.zeros(n_points, dtype=evaluation_points.dtype)
            for i in range(0, n_points, BATCH_SIZE):
                end = min(i + BATCH_SIZE, n_points)
                results[i:end] = self.neval(evaluation_points[i:end])
            return results

        # Iterative reduction to save memory:
        # Avoids creating (n_points, n_terms, dimension) tensor which is O(N*M*D).
        # Instead uses O(N*M) space accumulator.
        n_points = evaluation_points.shape[0]
        n_terms = coeffs.shape[0]
        dtype = evaluation_points.dtype
        
        term_values = backend.ones((n_points, n_terms), dtype=dtype)
        
        for d in range(self.dimension):
             # Extract d-th component: (N, 1)
             pts_d = evaluation_points[:, d:d+1] 
             # Extract d-th exponents: (1, M)
             exps_d = exponents[np.newaxis, :, d]
             
             col_vals = backend.power(pts_d, exps_d)
             term_values *= col_vals

        # Dot product of term values and coefficients
        results = backend.dot(term_values, coeffs)

        return results

    def _add_cosy(self, other):
        if not isinstance(other, MultivariateTaylorFunction):
            try:
                other = self.to_mtf(other, self.dimension)
            except (TypeError, ValueError):
                return NotImplemented

        if self.dimension != other.dimension:
            raise ValueError("MTF dimensions must match for addition.")

        self._ensure_backend()
        other._ensure_backend()

        if self.mtf_data is not None and other.mtf_data is not None:
            res_data = self.mtf_data.add(other.mtf_data)
            result_mtf = type(self)(mtf_data=res_data, dimension=self.dimension)
            if self._TRUNCATE_AFTER_OPERATION:
                result_mtf._cleanup_after_operation()
            return result_mtf
        # Fallback shouldn't happen if ensured
        return NotImplemented

    def _add_python(self, other):
        if not isinstance(other, MultivariateTaylorFunction):
            try:
                other = self.to_mtf(other, self.dimension)
            except (TypeError, ValueError):
                return NotImplemented

        if self.dimension != other.dimension:
            raise ValueError("MTF dimensions must match for addition.")

        # Python Implementation (Optimized with dictionary)
        is_complex = np.iscomplexobj(self.coeffs) or np.iscomplexobj(other.coeffs)
        summed_coeffs_dict = defaultdict(complex) if is_complex else defaultdict(float)

        for i in range(self.coeffs.shape[0]):
            exp_tuple = tuple(self.exponents[i])
            summed_coeffs_dict[exp_tuple] += self.coeffs[i]

        for i in range(other.coeffs.shape[0]):
            exp_tuple = tuple(other.exponents[i])
            summed_coeffs_dict[exp_tuple] += other.coeffs[i]

        if not summed_coeffs_dict:
            unique_exponents = np.empty((0, self.dimension), dtype=np.int32)
            summed_coeffs = np.empty(
                (0,), dtype=np.complex128 if is_complex else np.float64
            )
        else:
            unique_exponents = np.array(list(summed_coeffs_dict.keys()), dtype=np.int32)
            summed_coeffs = np.array(
                list(summed_coeffs_dict.values()),
                dtype=np.complex128 if is_complex else np.float64,
            )

        result_mtf = type(self)((unique_exponents, summed_coeffs), self.dimension)
        if self._TRUNCATE_AFTER_OPERATION:
            result_mtf._cleanup_after_operation()
        return result_mtf

    def __add__(self, other):
        # Default implementation (will be swapped out)
        if self._IMPLEMENTATION == "cosy":
            return self._add_cosy(other)
        else:
            return self._add_python(other)


    def __radd__(self, other):
        """Defines reverse addition for commutative property."""
        return self.__add__(other)

    def _sub_cosy(self, other):
        if not isinstance(other, MultivariateTaylorFunction):
            try:
                other = self.to_mtf(other, self.dimension)
            except (TypeError, ValueError):
                return NotImplemented

        if self.dimension != other.dimension:
            raise ValueError("MTF dimensions must match for subtraction.")

        self._ensure_backend()
        other._ensure_backend()

        if self.mtf_data is not None and other.mtf_data is not None:
             res_data = self.mtf_data.subtract(other.mtf_data)
             result_mtf = type(self)(mtf_data=res_data, dimension=self.dimension)
             if self._TRUNCATE_AFTER_OPERATION:
                 result_mtf._cleanup_after_operation()
             return result_mtf
        return NotImplemented

    def _sub_python(self, other):
        # Fallback to neg and add or direct implementation
        # Direct is better
        # Reuse add logic with negative sign?
        # Actually __neg__ creates a copy.
        # Efficient Subtraction:
        # Same structure as add but subtract second terms
        
        if not isinstance(other, MultivariateTaylorFunction):
            try:
                other = self.to_mtf(other, self.dimension)
            except (TypeError, ValueError):
                return NotImplemented

        if self.dimension != other.dimension:
            raise ValueError("MTF dimensions must match for subtraction.")

        is_complex = np.iscomplexobj(self.coeffs) or np.iscomplexobj(other.coeffs)
        summed_coeffs_dict = defaultdict(complex) if is_complex else defaultdict(float)

        for i in range(self.coeffs.shape[0]):
            exp_tuple = tuple(self.exponents[i])
            summed_coeffs_dict[exp_tuple] += self.coeffs[i]

        for i in range(other.coeffs.shape[0]):
            exp_tuple = tuple(other.exponents[i])
            summed_coeffs_dict[exp_tuple] -= other.coeffs[i]
            
        if not summed_coeffs_dict:
             unique_exponents = np.empty((0, self.dimension), dtype=np.int32)
             summed_coeffs = np.empty((0,), dtype=np.complex128 if is_complex else np.float64)
        else:
             unique_exponents = np.array(list(summed_coeffs_dict.keys()), dtype=np.int32)
             summed_coeffs = np.array(list(summed_coeffs_dict.values()), dtype=np.complex128 if is_complex else np.float64)
        
        result_mtf = type(self)((unique_exponents, summed_coeffs), self.dimension)
        if self._TRUNCATE_AFTER_OPERATION:
             result_mtf._cleanup_after_operation()
        return result_mtf

    def __sub__(self, other):
        if self._IMPLEMENTATION == "cosy":
            return self._sub_cosy(other)
        else:
            return self._sub_python(other)


    def __rsub__(self, other):
        """Defines reverse subtraction for non-commutative property."""
        if isinstance(other, (int, float, complex, np.number)):
            return self.to_mtf(other, self.dimension) - self
        return -(self - other)

    def _mul_cosy(self, other):
        if isinstance(other, (int, float, complex, np.number)):
            # Scalar mult cosy
             # Note: mtf_data usually handles scalar mult via promote or custom op?
             # For now using to_complex promote or standard conversion
             # But if it is scalar, we can skip create?
             # Let's ensure consistency:
             pass 
        
        # Standardize 'other'
        if isinstance(other, (int, float, complex, np.number)):
             # We can handle scalar directly if mtf_data supports it, or wrap
             # Current implementation wraps via __mul__ path?
             # Let's just wrap it to keep it simple or implement scalar op
             # Existing __mul__ logic:
             pass
        
        # Let's reuse the logic structure
        if isinstance(other, (int, float, complex, np.number)):
             # Optimized scalar
             if self.mtf_data is not None:
                 res_data = self.mtf_data * other
                 result_mtf = type(self)(mtf_data=res_data, dimension=self.dimension)
                 if self._TRUNCATE_AFTER_OPERATION:
                     result_mtf._cleanup_after_operation()
                 return result_mtf

        if not isinstance(other, MultivariateTaylorFunction):
            return NotImplemented

        if self.dimension != other.dimension:
             raise ValueError("MTF dimensions must match for multiplication.")

        self._ensure_backend()
        other._ensure_backend()
        
        if self.mtf_data is not None and other.mtf_data is not None:
            res_data = self.mtf_data.multiply(other.mtf_data)
            result_mtf = type(self)(mtf_data=res_data, dimension=self.dimension)
            if self._TRUNCATE_AFTER_OPERATION:
                result_mtf._cleanup_after_operation()
            return result_mtf
        return NotImplemented

    def _mul_python(self, other):
        if isinstance(other, (int, float, complex, np.number)):
            # Scalar multiplication
            if self.coeffs.size == 0:
                return self.copy()
            return type(self)(
                (self.exponents.copy(), self.coeffs * other), self.dimension
            )

        if not isinstance(other, MultivariateTaylorFunction):
            return NotImplemented

        if self.dimension != other.dimension:
            raise ValueError("MTF dimensions must match for multiplication.")

        if self.coeffs.size == 0 or other.coeffs.size == 0:
            dtype = np.result_type(self.coeffs.dtype, other.coeffs.dtype)
            return type(self)(
                (
                    np.empty((0, self.dimension), dtype=np.int32),
                    np.empty((0,), dtype=dtype),
                ),
                self.dimension,
            )

        # Dense Mode Optimization
        if self._MULT_TABLE is not None:
            # Check if we can use cached dense coefficients directly
            # or if we need to get indices from sparse exponents
            
            n_total_terms = self._MULT_TABLE.shape[0]
            
            # Get indices for A
            if self._dense_coeffs is not None:
                # If we are already dense, finding non-zeros is fast 
                # or we track them. For now, just use nonzero on the array.
                idx_a = np.nonzero(self._dense_coeffs)[0].astype(np.int32)
                coeffs_a = self._dense_coeffs[idx_a]
            else:
                idx_a = self._get_indices()
                coeffs_a = self.coeffs
                
            # Get indices for B
            if other._dense_coeffs is not None:
                idx_b = np.nonzero(other._dense_coeffs)[0].astype(np.int32)
                coeffs_b = other._dense_coeffs[idx_b]
            else:
                idx_b = other._get_indices()
                coeffs_b = other.coeffs

            if idx_a is not None and idx_b is not None:
                if _NUMBA_AVAILABLE:
                    # Ensure common dtype for Numba kernel
                    dtype = np.result_type(coeffs_a, coeffs_b)
                    if coeffs_a.dtype != dtype:
                        coeffs_a = coeffs_a.astype(dtype)
                    if coeffs_b.dtype != dtype:
                        coeffs_b = coeffs_b.astype(dtype)
                        
                    # Use new Parallel Kernel
                    dense_result = numba_kernels.multiply_dense_parallel(
                        idx_a, coeffs_a, 
                        idx_b, coeffs_b, 
                        self._MULT_TABLE, 
                        n_total_terms
                    )
                else:
                    # Fallback to NumPy (broadcast)
                    # BroadCast to get all pairs pairs (N, M)
                    res_indices_mat = self._MULT_TABLE[idx_a[:, np.newaxis], idx_b[np.newaxis, :]]
                    
                    # Calculate products
                    prod_coeffs = (coeffs_a[:, np.newaxis] * coeffs_b[np.newaxis, :])
                    
                    # Flatten
                    res_indices_flat = res_indices_mat.ravel()
                    prod_coeffs_flat = prod_coeffs.ravel()
                    
                    # Filter out -1
                    valid_mask = res_indices_flat != -1
                    res_indices_valid = res_indices_flat[valid_mask]
                    prod_coeffs_valid = prod_coeffs_flat[valid_mask]
                    
                    dtype = np.result_type(coeffs_a, coeffs_b)
                    dense_result = np.zeros(n_total_terms, dtype=dtype)
                    
                    np.add.at(dense_result, res_indices_valid, prod_coeffs_valid)

                # --- LAZY RETURN ---
                # Create a new MTF but DO NOT populate exponents/coeffs yet.
                # We interpret this "empty" MTF as valid if it has _dense_coeffs.
                result_mtf = type(self)(coefficients={}, dimension=self.dimension)
                result_mtf._dense_coeffs = dense_result
                # We leave _exponents and _coeffs empty/dummy for now.
                
                return result_mtf

        # Vectorized Implementation
        # 1. Compute all exponent combinations: (N, 1, D) + (1, M, D) -> (N, M, D)
        new_exps = (
            self.exponents[:, np.newaxis, :] + other.exponents[np.newaxis, :, :]
        ).reshape(-1, self.dimension)

        # 2. Compute all coefficient products: (N, 1) * (1, M) -> (N, M)
        new_coeffs = (self.coeffs[:, np.newaxis] * other.coeffs[np.newaxis, :]).ravel()

        # 3. Aggregate common exponents
        # Optimization: Use void view for faster unique row finding
        # This treats each row as a block of bytes, allowing 1D sorting/unique
        new_exps_c = np.ascontiguousarray(new_exps)
        void_dtype = np.dtype((np.void, new_exps.dtype.itemsize * new_exps.shape[1]))
        view = new_exps_c.view(void_dtype).ravel()
        
        _, unique_indices, inverse_indices = np.unique(
            view, return_index=True, return_inverse=True
        )
        unique_exponents = new_exps_c[unique_indices]

        # 4. Sum coefficients corresponding to same exponents
        dtype = np.result_type(self.coeffs, other.coeffs)
        summed_coeffs = np.zeros(unique_exponents.shape[0], dtype=dtype)
        np.add.at(summed_coeffs, inverse_indices, new_coeffs)

        result_mtf = type(self)((unique_exponents, summed_coeffs), self.dimension)
        if self._TRUNCATE_AFTER_OPERATION:
            result_mtf._cleanup_after_operation()
        return result_mtf

    def __mul__(self, other):
        if self._IMPLEMENTATION == "cosy":
            return self._mul_cosy(other)
        else:
            return self._mul_python(other)


    def __rmul__(self, other):
        """Defines reverse multiplication for commutative property."""
        return self.__mul__(other)

    def __imul__(self, other):
        """Defines in-place multiplication (*=) with a scalar."""
        if isinstance(other, (int, float, np.number)):
            self.coeffs *= other
            return self
        else:
            return NotImplemented

    def __pow__(self, power):
        """
        Raises the MTF to a power.

        Supports integer powers (positive, negative, and zero) and specific
        float powers (0.5 for square root, -0.5 for inverse square root).

        Parameters
        ----------
        power : int or float
            The power to raise the MTF to.

        Returns
        -------
        MultivariateTaylorFunction
            A new MTF representing the result of the exponentiation.

        Raises
        ------
        ValueError
            If the power is not a supported type or value (e.g., a float
            other than 0.5 or -0.5).
        """
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            try:
                res_data = self.mtf_data**power
                return self._create_result(res_data)
            except NotImplementedError:
                pass  # Fallback to default if not supported (e.g. complex)

        if isinstance(power, numbers.Integral):
            if power < 0:
                # Generalize for any negative integer power
                inv_self = self._inv_mtf_internal(self)
                return inv_self ** abs(power)
            if power == 0:
                return type(self).from_constant(1.0, dimension=self.dimension)
            if power == 1:
                return self.copy()

            # Optimized power using binary exponentiation (exponentiation by
            # squaring) for non-negative integers
            result = type(self).from_constant(1.0, dimension=self.dimension)
            base = self.copy()
            while power > 0:
                if power % 2 == 1:
                    result *= base
                base *= base
                power //= 2
            return result

        elif isinstance(power, float):
            if power == 0.5:
                return self.sqrt()
            elif power == -0.5:
                return self.isqrt()
            else:
                raise ValueError("Power must be an integer, 0.5, or -0.5.")
        else:
            raise ValueError("Power must be an integer, 0.5, or -0.5.")


    def __neg__(self):
        """
        Negates the MultivariateTaylorFunction.

        Returns
        -------
        MultivariateTaylorFunction
            A new MTF with all coefficients negated.
        """
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            res_data = self.mtf_data.negate()
            return type(self)(mtf_data=res_data, dimension=self.dimension)
        return type(self)((self.exponents.copy(), -self.coeffs), self.dimension)

    def inverse(self):
        """
        Computes the multiplicative inverse (1/f).
        """
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            try:
                res_data = self.mtf_data.inverse()
                return self._create_result(res_data)
            except NotImplementedError:
                pass  # Fallback

        return self._inv_mtf_internal(self)

    def _truediv_cosy(self, other):
        if isinstance(other, (int, float, complex, np.number)):
             # Let's use to_mtf or special handling
             # Reuse existing structure
             try:
                 other = self.to_mtf(other, self.dimension)
             except (TypeError, ValueError):
                 return NotImplemented
        
        if not isinstance(other, MultivariateTaylorFunction):
            return NotImplemented

        if self.dimension != other.dimension:
            raise ValueError("MTF dimensions must match for division.")

        angle_data = self.mtf_data # Ensure ensure_backend called before access if needed?
        # Check ensure
        self._ensure_backend()
        other._ensure_backend()
        
        if self.mtf_data is not None and other.mtf_data is not None:
             res_data = self.mtf_data.divide(other.mtf_data)
             result_mtf = type(self)(mtf_data=res_data, dimension=self.dimension)
             if self._TRUNCATE_AFTER_OPERATION:
                 result_mtf._cleanup_after_operation()
             return result_mtf
        return NotImplemented

    def _truediv_python(self, other):
        # Default Python implementation
        if isinstance(other, (int, float, complex, np.number)):
            return self * (1.0 / other)
            
        if not isinstance(other, MultivariateTaylorFunction):
            try:
                 other = self.to_mtf(other, self.dimension)
            except:
                 return NotImplemented
                 
        inverse_other_mtf = self._inv_mtf_internal(other)
        return self * inverse_other_mtf

    def __truediv__(self, other):
        if self._IMPLEMENTATION == "cosy":
            return self._truediv_cosy(other)
        else:
            return self._truediv_python(other)


    def __rtruediv__(self, other):
        if not isinstance(other, MultivariateTaylorFunction):
            try:
                # Convert scalar to MTF to use backend division
                other_mtf = self.to_mtf(other, self.dimension)
            except (TypeError, ValueError):
                return NotImplemented
        else:
            other_mtf = other

        # Backend routing
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            res_data = other_mtf.mtf_data.divide(self.mtf_data)
            result_mtf = type(self)(mtf_data=res_data, dimension=self.dimension)
            if self._TRUNCATE_AFTER_OPERATION:
                result_mtf._cleanup_after_operation()
            return result_mtf

        # Default Python implementation
        inverse_self_mtf = self._inv_mtf_internal(self)
        return inverse_self_mtf * other

    def deriv(self, var_idx):
        """
        Computes the partial derivative with respect to variable `var_idx`.

        Parameters
        ----------
        var_idx : int
            The 1-based index of the variable (e.g., 1 for x1).

        Returns
        -------
        MultivariateTaylorFunction
            The partial derivative.
        """
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            res_data = self.mtf_data.partial_derivative(var_idx)
            return type(self)(mtf_data=res_data, dimension=self.dimension)

        from .elementary_functions import _derivative

        return _derivative(self, var_idx)

    def derivative(self, var_idx):
        """Alias for deriv."""
        return self.deriv(var_idx)

    def integrate(self, var_idx, lower_limit=None, upper_limit=None):
        """
        Computes the indefinite or definite integral with respect to variable `var_idx`.

        Parameters
        ----------
        var_idx : int
            The 1-based index of the variable.
        lower_limit : float, optional
            Lower limit for definite integration.
        upper_limit : float, optional
            Upper limit for definite integration.

        Returns
        -------
        MultivariateTaylorFunction
            The integrated function.
        """
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            if lower_limit is None and upper_limit is None:
                res_data = self.mtf_data.integrate(var_idx)
                return type(self)(mtf_data=res_data, dimension=self.dimension)

        from .elementary_functions import _integrate

        return _integrate(self, var_idx, lower_limit, upper_limit)

    def poisson_bracket(self, other):
        """
        Computes the Poisson Bracket [self, other].

        Parameters
        ----------
        other : MultivariateTaylorFunction
            The other function in the bracket.

        Returns
        -------
        MultivariateTaylorFunction
            The result of the Poisson Bracket.
        """
        if not isinstance(other, MultivariateTaylorFunction):
            try:
                other = self.to_mtf(other, self.dimension)
            except (TypeError, ValueError):
                return NotImplemented

        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            res_data = self.mtf_data.poisson_bracket(other.mtf_data)
            return type(self)(mtf_data=res_data, dimension=self.dimension)

        raise NotImplementedError(
            "poisson_bracket() not yet implemented for Python backend"
        )

    def _inv_mtf_internal(self, mtf_instance, order=None):
        """Internal method to calculate Taylor expansion of 1/mtf_instance."""
        if self._IMPLEMENTATION == "cosy" and mtf_instance.mtf_data is not None:
            res_data = mtf_instance.mtf_data.inverse()
            result_mtf = type(self)(mtf_data=res_data, dimension=self.dimension)
            if self._TRUNCATE_AFTER_OPERATION:
                result_mtf._cleanup_after_operation()
            return result_mtf

        if order is None:
            order = self.get_max_order()
        constant_term_coeff = mtf_instance.extract_coefficient(
            tuple([0] * mtf_instance.dimension)
        )
        c0 = constant_term_coeff.item()
        if abs(c0) < self.get_etol():
            raise ValueError(
                "Cannot invert MTF with zero constant term (or very close to zero)."
            )
        rescaled_mtf = mtf_instance / c0
        # Dynamic coefficient generation for inverse: (-1)^n
        # 1/(1+x) = 1 - x + x^2 - x^3 ...
        coeffs_to_use = [(-1) ** i for i in range(order + 1)]
        coeff_items = []
        for i, coeff_val in enumerate(coeffs_to_use):
            exponent_tuple = (i,)
            coeff_items.append((exponent_tuple, coeff_val))
        inverse_series_1d_mtf = type(self)(coefficients=dict(coeff_items), dimension=1)
        composed_mtf = inverse_series_1d_mtf.compose({
            1: rescaled_mtf
            - type(self).from_constant(1.0, dimension=rescaled_mtf.dimension)
        })
        final_mtf = composed_mtf / c0
        truncated_mtf = final_mtf.truncate(order)
        return truncated_mtf

    def substitute_variable(self, var_index, value):
        """
        Substitutes a variable with a numerical value.

        This operation effectively evaluates the function along a specific
        axis, resulting in a new Taylor function with the same dimension,
        but where one variable's influence is now fixed.

        Parameters
        ----------
        var_index : int
            The 1-based index of the variable to substitute.
        value : numeric
            The numerical value to substitute for the variable.

        Returns
        -------
        MultivariateTaylorFunction
            A new MTF with the variable substituted.

        Raises
        ------
        TypeError
            If `var_index` is not an integer or `value` is not a number.
        ValueError
            If `var_index` is out of the valid range [1, dimension].
        """
        if not isinstance(var_index, int):
            raise TypeError("var_index must be an integer dimension index (1-based).")
        if not (1 <= var_index <= self.dimension):
            raise ValueError(
                f"var_index must be between 1 and {self.dimension}, inclusive."
            )
        if not isinstance(value, (int, float, complex, np.number)):
            raise TypeError("value must be a number.")

        new_coeffs = self.coeffs.copy()
        new_exponents = self.exponents.copy()

        dim_exponents = new_exponents[:, var_index - 1]

        # Calculate the multipliers for each term
        multipliers = np.power(value, dim_exponents)

        # Apply the multipliers
        new_coeffs *= multipliers

        # Zero out the exponents for the substituted variable
        new_exponents[:, var_index - 1] = 0

        # Use a dictionary to group and sum coefficients, as bincount does not
        # support complex numbers
        is_result_complex = np.iscomplexobj(new_coeffs) or np.iscomplexobj(value)
        summed_coeffs_dict = (
            defaultdict(complex) if is_result_complex else defaultdict(float)
        )
        for i, exp in enumerate(new_exponents):
            summed_coeffs_dict[tuple(exp)] += new_coeffs[i]

        unique_exponents = np.array(list(summed_coeffs_dict.keys()), dtype=np.int32)
        summed_coeffs = np.array(list(summed_coeffs_dict.values()))

        return type(self)((unique_exponents, summed_coeffs), self.dimension)

    def truncate_inplace(self, order=None):
        """Truncates the MTF **in place** to a specified order."""
        if order is None:
            order = self.get_max_order()
        elif not isinstance(order, int) or order < 0:
            raise ValueError("Order must be a non-negative integer.")

        if self.exponents.shape[0] == 0:
            return self

        term_orders = np.sum(self.exponents, axis=1)
        keep_mask = term_orders <= order

        self.exponents = self.exponents[keep_mask]
        self.coeffs = self.coeffs[keep_mask]

        return self

    def truncate(self, order=None):
        """
        Truncates the Taylor series to a specified maximum order.

        Removes all terms whose total order (sum of exponents) is greater
        than the specified `order`. Also removes terms with coefficients
        smaller than the global error tolerance `_ETOL`.

        Parameters
        ----------
        order : int, optional
            The maximum order to keep. If None, the global `_MAX_ORDER` is
            used.

        Returns
        -------
        MultivariateTaylorFunction
            A new, truncated MTF.
        """
        etol = self.get_etol()
        if order is None:
            order = self.get_max_order()
        elif not isinstance(order, int) or order < 0:
            raise ValueError("Order must be a non-negative integer.")

        if self.coeffs.size == 0:
            return self.copy()

        term_orders = np.sum(self.exponents, axis=1)
        order_mask = term_orders <= order

        etol_mask = np.abs(self.coeffs) > etol

        keep_mask = np.logical_and(order_mask, etol_mask)

        new_exponents = self.exponents[keep_mask]
        new_coeffs = self.coeffs[keep_mask]

        return type(self)((new_exponents, new_coeffs), self.dimension)

    def substitute_variable_inplace(self, var_dimension, value):
        """Substitutes a variable in the MTF with a numerical value IN-PLACE."""
        if not isinstance(var_dimension, int) or not (
            1 <= var_dimension <= self.dimension
        ):
            raise ValueError("Invalid var_dimension.")
        if not isinstance(value, (int, float, complex, np.number)):
            raise TypeError("Value must be a number.")

        new_self = self.substitute_variable(var_dimension, value)
        self.exponents = new_self.exponents
        self.coeffs = new_self.coeffs

    def is_zero_mtf(self, mtf, zero_tolerance=None):
        """Checks if an MTF is effectively zero."""
        if zero_tolerance is None:
            zero_tolerance = self.get_etol()
        if mtf.coeffs.size == 0:
            return True
        return np.all(np.abs(mtf.coeffs) < zero_tolerance)

    def compose(
        self, other_function_dict: dict[int, "MultivariateTaylorFunction"]
    ) -> "MultivariateTaylorFunction":
        """
        Composes this function with other Taylor functions.

        This method performs function composition, substituting specified
        variables of this function (`self`) with other
        `MultivariateTaylorFunction` objects. The composition is of the form
        `f(g_1, g_2, ..., g_n)`, where `f` is `self` and each `g_i` is an
        MTF from the `other_function_dict`.

        Parameters
        ----------
        other_function_dict : dict[int, MultivariateTaylorFunction]
            A dictionary mapping variable indices (1-based) of `self` to the
            `MultivariateTaylorFunction` objects that should be substituted
            in their place.

        Returns
        -------
        MultivariateTaylorFunction
            A new MTF representing the composed function.

        Raises
        ------
        TypeError
            If `other_function_dict` is not a dictionary or if its values are
            not MTF objects.
        ValueError
            If the dimensions of the inner functions (`g_i`) are inconsistent,
            or if a variable index is out of bounds.

        Examples
        --------
        >>> mtf.initialize_mtf(max_order=2, max_dimension=2)
        >>> x1 = mtf.from_variable(1, 2)
        >>> x2 = mtf.from_variable(2, 2)
        >>> f = x1 * x1 + x2
        >>> g1 = x1 + 1
        >>> g2 = x2 * 2
        >>> # Compose f with g1 and g2, i.e., f(g1, g2)
        >>> h = f.compose({1: g1, 2: g2})
        >>> print(h.get_tabular_dataframe())
           Coefficient  Order Exponents
        0          1.0      0    (0, 0)
        1          2.0      1    (0, 1)
        2          2.0      1    (1, 0)
        3          1.0      2    (2, 0)
        """
        if not isinstance(other_function_dict, dict):
            raise TypeError("other_function_dict must be a dictionary.")

        # If the dictionary is empty, just return a copy of self.
        if not other_function_dict:
            return self.copy()

        # Validate inputs and determine the dimension of the resulting
        # function.
        result_dim = None
        for var_index, g in other_function_dict.items():
            if not isinstance(var_index, int):
                raise TypeError(
                    "Keys of other_function_dict must be integers (variable indices)."
                )
            if not isinstance(g, MultivariateTaylorFunction):
                raise TypeError(
                    f"Value for key {var_index} must be a "
                    "MultivariateTaylorFunction object."
                )
            if not (1 <= var_index <= self.dimension):
                raise ValueError(
                    f"Variable index {var_index} is out of bounds for the outer "
                    f"function's dimension {self.dimension}."
                )

            if result_dim is None:
                result_dim = g.dimension
            elif result_dim != g.dimension:
                raise ValueError("All inner functions must have the same dimension.")

        # Create the full substitution mapping.
        # This mapping is used for both Python backend and determining arguments for COSY.
        substitutions = {}
        for i in range(1, self.dimension + 1):
            if i in other_function_dict:
                substitutions[i] = other_function_dict[i]
            else:
                # If a variable is not being substituted, it becomes a variable
                # in the new space.
                if i > result_dim:
                    raise ValueError(
                        f"Outer function variable {i} is not being substituted, but "
                        f"the result dimension is only {result_dim}."
                    )
                substitutions[i] = type(self).var(i, dimension=result_dim)

        # COSY Backend Optimization: Use POLVAL for fast composition
        if False and self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            # Construct ordered list of CosyDA objects corresponding to variables 1..self.dimension
            args_da_list = []
            for i in range(1, self.dimension + 1):
                # We can rely on substitutions dict which is fully populated above
                mtf_arg = substitutions[i]
                if mtf_arg.mtf_data is None:
                    # This should not happen if backend is consistent, but safeguard
                    raise RuntimeError(f"Argument for var {i} has no COSY data")
                args_da_list.append(mtf_arg.mtf_data.da)

            res_da = self.mtf_data.da.compose_polval(args_da_list)

            # The result should be wrapped in CosyMtfData
            # We need to access CosyMtfData class. It is available via self.mtf_data.__class__
            res_data = self.mtf_data.__class__(result_dim, idx=res_da.idx, owned=True)
            # Be careful: compose_polval returns CosyDA(owned=True).
            # CosyMtfData(..., idx=..., owned=True) will take ownership.
            # res_da is a temporary wrapper; extracting idx handles transfer if we don't close res_da?
            # CosyDA.__del__ calls cosy_free_ if owned.
            # So: res_da owns it. We pass idx to res_data. We must detach ownership from res_da or set res_data.da = res_da?
            # Current CosyMtfData implementation:
            # def __init__(self, dimension, idx=None, ... owned=False):
            #     if idx is not None: self.da = CosyDA(idx=idx, owned=owned)
            # So passing idx and owned=True works, BUT res_da ALSO thinks it owns it.
            # When res_da goes out of scope, it frees idx. Then res_data has dangling ptr.
            # Fix: res_da.owned = False after transfer.
            res_da.owned = False

            return type(self)(mtf_data=res_data, dimension=result_dim)

        # The final MTF will be initialized as a zero constant of the correct
        # dimension.
        final_mtf = type(self).from_constant(0.0, dimension=result_dim)

        # If self is a zero function, the composition is also a zero function.
        if self.coeffs.size == 0:
            return final_mtf

        # Iterate through terms of self (the outer function) and build the
        # final mtf.
        for i in range(self.coeffs.size):
            coeff = self.coeffs[i]
            exponents = self.exponents[i]

            # Start with the coefficient of the term
            term_result = type(self).from_constant(coeff, dimension=result_dim)

            # Multiply by the substituted functions raised to their powers
            for j in range(self.dimension):
                power = exponents[j]
                if power > 0:
                    var_index = j + 1
                    g_j = substitutions[var_index]
                    term_result *= g_j**power

            final_mtf += term_result

        return final_mtf

    def to_dict(self):
        """
        Returns the exponents and coefficients as a dictionary.

        Returns
        -------
        dict
            A dictionary with keys 'exponents' and 'coeffs'.
        """
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            return self.mtf_data.to_dict()
        self._ensure_synced()
        return {"exponents": self._exponents, "coeffs": self._coeffs}

    def _create_result(self, res_data):
        """Helper to create a result MTF, promoting to complex if needed."""
        is_complex = False
        if hasattr(res_data, "is_complex"):
            is_complex = res_data.is_complex

        if is_complex:
            from .complex_taylor_function import ComplexMultivariateTaylorFunction

            cls = ComplexMultivariateTaylorFunction
        else:
            cls = type(self)

        result_mtf = cls(mtf_data=res_data, dimension=self.dimension)
        if self._TRUNCATE_AFTER_OPERATION:
            result_mtf._cleanup_after_operation()
        return result_mtf

    def get_tabular_dataframe(self):
        """
        Returns a pandas DataFrame representation of the Taylor function.

        The DataFrame provides a clear, tabular view of the non-zero terms
        of the Taylor series, including their coefficients, orders, and
        exponents.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns 'Coefficient', 'Order', and 'Exponents',
            sorted by order and then by exponents. If the function is zero,
            it returns a DataFrame with a single row representing the
            zero term.
        """
        if self.coeffs.size == 0:
            return pd.DataFrame([
                {
                    "Coefficient": 0.0,
                    "Order": 0,
                    "Exponents": (0,) * self.dimension,
                }
            ])

        data = []
        for i in range(self.coeffs.size):
            exponents = tuple(self.exponents[i])
            coeff = self.coeffs[i]
            order = sum(exponents)
            data.append({"Coefficient": coeff, "Order": order, "Exponents": exponents})

        df = pd.DataFrame(data)
        df = df.sort_values(
            by=["Order", "Exponents"], ascending=[True, False]
        ).reset_index(drop=True)
        return df

    def get_all_terms(self):
        """
        Returns a list of all non-zero terms as (exponents, coefficient) pairs.
        """
        return list(zip(map(tuple, self.exponents), self.coeffs))

    def extract_coefficient(self, exponents):
        """
        Extracts the coefficient for a given multi-index exponent.

        Parameters
        ----------
        exponents : tuple
            A tuple of integers representing the multi-index exponent of the
            term whose coefficient is to be extracted. The length of the
            tuple must match the function's dimension.

        Returns
        -------
        np.ndarray
            A 1-element array containing the coefficient value. Returns an
            array with 0.0 if the term does not exist.

        Raises
        ------
        TypeError
            If `exponents` is not a tuple.
        ValueError
            If the length of `exponents` does not match the MTF dimension.
        """
        if self.exponents.size == 0:
            return np.array([0.0], dtype=self.coeffs.dtype)
        if not isinstance(exponents, tuple):
            raise TypeError("Exponents must be a tuple.")
        if len(exponents) != self.dimension:
            raise ValueError(
                f"Exponents tuple length must match MTF dimension ({self.dimension})."
            )

        exponent_row = np.array(exponents, dtype=np.int32)
        match = np.all(self.exponents == exponent_row, axis=1)
        match_indices = np.where(match)[0]

        if match_indices.size > 0:
            return np.array([self.coeffs[match_indices[0]]]).reshape(1)
        else:
            return np.array([0.0], dtype=self.coeffs.dtype)

    def set_coefficient(self, exponents, value):
        """
        Sets the coefficient for a given multi-index exponent.

        If a term with the specified exponent already exists, its
        coefficient is updated. Otherwise, a new term is created.

        Parameters
        ----------
        exponents : tuple
            A tuple of integers for the multi-index exponent. Its length
            must match the function's dimension.
        value : numeric
            The new coefficient value.

        Raises
        ------
        TypeError
            If `exponents` is not a tuple or `value` is not a number.
        ValueError
            If the length of `exponents` does not match the MTF dimension.
        """
        if not isinstance(exponents, tuple):
            raise TypeError("Exponents must be a tuple.")
        if len(exponents) != self.dimension:
            raise ValueError(
                f"Exponents tuple length must match MTF dimension ({self.dimension})."
            )
        if not isinstance(value, (int, float, np.number, complex)):
            raise TypeError("Coefficient value must be a number.")

        exponent_row = np.array(exponents, dtype=np.int32)
        match = np.all(self.exponents == exponent_row, axis=1)
        match_indices = np.where(match)[0]

        if match_indices.size > 0:
            self.coeffs[match_indices[0]] = value
        else:
            self.exponents = np.vstack([self.exponents, exponent_row])
            self.coeffs = np.append(self.coeffs, value)

            sorted_indices = np.lexsort(self.exponents.T)
            self.exponents = self.exponents[sorted_indices]
            self.coeffs = self.coeffs[sorted_indices]

    def get_max_coefficient(self):
        """Finds the maximum absolute value among all coefficients."""
        if self.coeffs.size == 0:
            return 0.0
        return np.max(np.abs(self.coeffs))

    def get_min_coefficient(self, tolerance=np.nan):
        """Finds the minimum absolute value among non-negligible coefficients."""
        if tolerance is np.nan:
            tolerance = self.get_etol()

        if self.coeffs.size == 0:
            return 0.0

        non_negligible_coeffs = self.coeffs[np.abs(self.coeffs) > tolerance]
        if non_negligible_coeffs.size == 0:
            return 0.0

        return np.min(np.abs(non_negligible_coeffs))

    def __str__(self):
        """Returns a string representation of the MTF (tabular format)."""
        df = self.get_tabular_dataframe()
        return f"{df}\n"

    def __repr__(self):
        """Returns a detailed string representation of the MTF (for debugging)."""
        df = self.get_tabular_dataframe()
        return f"{df}\n"

    def symprint(self, symbols=None, precision=6, coeff_formatter=None):
        """
        Converts the MTF to a SymPy expression for pretty printing.

        Parameters
        ----------
        symbols : list of str, optional
            A list of symbolic names for the variables. If not provided,
            defaults like 'x', 'y', 'z' are used.
        precision : int, optional
            The number of decimal digits for coefficients when using the
            default formatter. Defaults to 6.
        coeff_formatter : callable, optional
            A custom function to format coefficients into SymPy-compatible
            numbers. It should accept a coefficient and precision.

        Returns
        -------
        sympy.Expr
            A SymPy expression representing the Taylor series.

        Raises
        ------
        ImportError
            If SymPy is not installed.
        ValueError
            If the number of provided symbols is less than the function's
            dimension.
        """
        try:
            import sympy as sp
        except ImportError:
            raise ImportError(
                "SymPy is required for the symprint method. Please install it using "
                "'pip install sympy'."
            )

        if symbols is None:
            symbols = ["x", "y", "z", "u", "v", "w", "p", "q", "s", "t"]

        if self.dimension > len(symbols):
            raise ValueError(
                f"Not enough symbols provided for the {self.dimension}-dimensional "
                "function."
            )

        sympy_vars = sp.symbols(symbols[: self.dimension])

        if coeff_formatter is None:

            def default_formatter(c, p):
                if np.iscomplexobj(c):
                    return sp.Float(c.real, p) + sp.I * sp.Float(c.imag, p)
                else:
                    return sp.Float(c, p)

            coeff_formatter = default_formatter

        sympy_expression = sum(
            coeff_formatter(coeff, precision)
            * sp.prod(
                sympy_vars[j] ** power for j, power in enumerate(exp_tuple) if power > 0
            )
            for coeff, exp_tuple in zip(self.coeffs, self.exponents)
        )

        return sympy_expression

    def copy(self):
        """Returns a copy of the MTF."""
        return type(self)(
            (self.exponents.copy(), self.coeffs.copy()),
            self.dimension,
            var_name=self.var_name,
        )

    def _cleanup_after_operation(self):
        """
        Removes coefficients smaller than the global error tolerance in-place.
        """
        # If using backend and Python data is not yet synchronized, assume backend
        # handles cleanup internally or ignore to preserve mtf_data.
        if (
            self._IMPLEMENTATION == "cosy"
            and self.mtf_data is not None
            and self._exponents is None
        ):
            return

        etol = self.get_etol()

        if self.coeffs.size == 0:
            return

        keep_mask = (
            (np.abs(self.coeffs) > etol) | np.isnan(self.coeffs) | np.isinf(self.coeffs)
        )

        self.exponents = self.exponents[keep_mask]
        self.coeffs = self.coeffs[keep_mask]

    def __eq__(self, other):
        """Defines equality (==) for MultivariateTaylorFunction objects."""
        if not isinstance(other, MultivariateTaylorFunction):
            return False
        if self.dimension != other.dimension:
            return False

        self_cleaned = self.copy()
        self_cleaned._cleanup_after_operation()
        other_cleaned = other.copy()
        other_cleaned._cleanup_after_operation()

        if self_cleaned.coeffs.shape[0] == 0 and other_cleaned.coeffs.shape[0] == 0:
            return True

        if self_cleaned.coeffs.shape[0] != other_cleaned.coeffs.shape[0]:
            return False

        if not np.array_equal(self_cleaned.exponents, other_cleaned.exponents):
            self_map = {
                tuple(exp): coeff
                for exp, coeff in zip(self_cleaned.exponents, self_cleaned.coeffs)
            }
            other_map = {
                tuple(exp): coeff
                for exp, coeff in zip(other_cleaned.exponents, other_cleaned.coeffs)
            }
            return self_map == other_map

        return np.allclose(self_cleaned.coeffs, other_cleaned.coeffs)

    def __ne__(self, other):
        """Defines inequality (!=) for MultivariateTaylorFunction objects."""
        return not self.__eq__(other)

    def sin(self) -> "MultivariateTaylorFunction":
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            return self._create_result(self.mtf_data.sin())
        from .elementary_functions import _sin_taylor

        return _sin_taylor(self)

    def cos(self) -> "MultivariateTaylorFunction":
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            return self._create_result(self.mtf_data.cos())
        from .elementary_functions import _cos_taylor

        return _cos_taylor(self)

    def tan(self) -> "MultivariateTaylorFunction":
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            return self._create_result(self.mtf_data.tan())
        from .elementary_functions import _tan_taylor

        return _tan_taylor(self)

    def exp(self) -> "MultivariateTaylorFunction":
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            return self._create_result(self.mtf_data.exp())
        from .elementary_functions import _exp_taylor

        return _exp_taylor(self)

    def log(self) -> "MultivariateTaylorFunction":
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            return self._create_result(self.mtf_data.log())
        from .elementary_functions import _log_taylor

        return _log_taylor(self)

    def sinh(self) -> "MultivariateTaylorFunction":
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            return self._create_result(self.mtf_data.sinh())
        from .elementary_functions import _sinh_taylor

        return _sinh_taylor(self)

    def cosh(self) -> "MultivariateTaylorFunction":
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            return self._create_result(self.mtf_data.cosh())
        from .elementary_functions import _cosh_taylor

        return _cosh_taylor(self)

    def tanh(self) -> "MultivariateTaylorFunction":
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            return self._create_result(self.mtf_data.tanh())
        from .elementary_functions import _tanh_taylor

        return _tanh_taylor(self)

    def arcsin(self) -> "MultivariateTaylorFunction":
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            return self._create_result(self.mtf_data.asin())
        from .elementary_functions import _arcsin_taylor

        return _arcsin_taylor(self)

    def arccos(self) -> "MultivariateTaylorFunction":
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            return self._create_result(self.mtf_data.acos())
        from .elementary_functions import _arccos_taylor

        return _arccos_taylor(self)

    def arctan(self) -> "MultivariateTaylorFunction":
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            return self._create_result(self.mtf_data.atan())
        from .elementary_functions import _arctan_taylor

        return _arctan_taylor(self)

    def sqrt(self) -> "MultivariateTaylorFunction":
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            return self._create_result(self.mtf_data.sqrt())
        return _sqrt_taylor(self)

    def isqrt(self) -> "MultivariateTaylorFunction":
        return self.inv_sqrt()

    def inv_sqrt(self) -> "MultivariateTaylorFunction":
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            return self._create_result(self.mtf_data.inv_sqrt())
        from .elementary_functions import _isqrt_taylor

        return _isqrt_taylor(self)

    def inv_pow_3_2(self) -> "MultivariateTaylorFunction":
        """Computes 1 / x^(3/2) using COSY DAISR3."""
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            res_data = self.mtf_data.inv_cbrt()  # This calls DAISR3 which is 1/x^1.5
            return type(self)(mtf_data=res_data, dimension=self.dimension)
        raise NotImplementedError("inv_pow_3_2 is only available for the COSY backend")

    def inv_cbrt(self) -> "MultivariateTaylorFunction":
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            # DAISR3 is actually 1/x^1.5, so we can't use it for inv_cbrt.
            # We could use self ** (-1/3) if __pow__ supported it.
            raise NotImplementedError(
                "inv_cbrt not yet implemented for COSY backend (DAISR3 is 1/x^1.5)"
            )
        raise NotImplementedError(
            "inv_cbrt will be implemented in future for Python backend"
        )

    def erf(self) -> "MultivariateTaylorFunction":
        # COSY's DAERF/REERF are under development and trigger errors for orders > 6.
        # Per user request, we ignore COSY for erf and use the Python Taylor fallback.
        from .elementary_functions import _erf_taylor

        return _erf_taylor(self)

    def coth(self) -> "MultivariateTaylorFunction":
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            return self._create_result(self.mtf_data.coth())
        from .elementary_functions import _coth_taylor

        return _coth_taylor(self)

    def cot(self) -> "MultivariateTaylorFunction":
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            return self._create_result(self.mtf_data.cot())
        from .elementary_functions import _cot_taylor

        return _cot_taylor(self)

    def norm(self) -> float:
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            return self.mtf_data.norm()
        return float(self.get_max_coefficient())

    def weighted_norm(self, weight) -> float:
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            return self.mtf_data.weighted_norm(weight)
        raise NotImplementedError(
            "weighted_norm is only available for the COSY backend"
        )

    def estimate_stability(self, var_id: int = 0, order: int = None) -> float:
        """
        Estimates the stability/order decay of the Taylor function.

        Only available for the 'cosy' backend.
        """
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            return self.mtf_data.estimate_stability(var_id, order)
        raise NotImplementedError(
            "estimate_stability is only available for the COSY backend"
        )

    def asin(self):
        return self.arcsin()

    def acos(self):
        return self.arccos()

    def atan(self):
        return self.arctan()

    def arcsinh(self):
        # We use Python fallback for inverse hyperbolics as COSY bindings were buggy
        from .elementary_functions import _arcsinh_taylor

        return _arcsinh_taylor(self)

    def arccosh(self):
        from .elementary_functions import _arccosh_taylor

        return _arccosh_taylor(self)

    def arctanh(self):
        from .elementary_functions import _arctanh_taylor

        return _arctanh_taylor(self)

    def asinh(self):
        return self.arcsinh()

    def acosh(self):
        return self.arccosh()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Implements NumPy ufunc protocol, directly calling taylor functions from
        elementary_functions.py. Handles scalar inputs by converting them to
        MultivariateTaylorFunction constants.
        """
        UNARY_UFUNC_MAP = {
            np.sin: "sin",
            np.cos: "cos",
            np.tan: "tan",
            np.exp: "exp",
            np.sqrt: "sqrt",
            np.log: "log",
            np.arctan: "arctan",
            np.sinh: "sinh",
            np.cosh: "cosh",
            np.tanh: "tanh",
            np.arcsin: "arcsin",
            np.arccos: "arccos",
            # np.arctanh: "arctanh",
            np.reciprocal: "_inv_mtf_internal",
            np.negative: "__neg__",
        }

        BINARY_UFUNC_MAP = {
            np.add: "__add__",
            np.subtract: "__sub__",
            np.multiply: "__mul__",
            np.divide: "__truediv__",
            np.true_divide: "__truediv__",
        }

        if method == "__call__":
            # -----------------------------------------------------------------
            # Batch / Vectorized Path
            # -----------------------------------------------------------------
            has_array = any(isinstance(x, np.ndarray) for x in inputs)
            if has_array and self._IMPLEMENTATION == "cosy":
                # Only support basic binary arithmetic for now in batch mode
                BATCH_UFUNC_MAP = {
                    np.add: cosy_backend.batch_add,
                    np.subtract: cosy_backend.batch_sub,
                    np.multiply: cosy_backend.batch_mul,
                    np.divide: cosy_backend.batch_div,
                    np.true_divide: cosy_backend.batch_div,
                }

                if ufunc in BATCH_UFUNC_MAP and len(inputs) == 2:
                    # Helper to get indices
                    def get_indices(x, size):
                        if isinstance(x, np.ndarray):
                            # Assume array of MTFs
                            # Fast extraction: we assume they are initialized and synced
                            # Validating every element is slow, we trust the array for speed
                            try:
                                return [m.mtf_data.da.idx for m in x.flat]
                            except AttributeError:
                                # Fallback if elements aren't MTFs (e.g. array of floats)
                                # Convert to constants?
                                # This is slow path: array(float) -> array(MTF)
                                return [
                                    self.to_mtf(v, self.dimension).mtf_data.da.idx
                                    for v in x.flat
                                ]
                        elif isinstance(x, MultivariateTaylorFunction):
                            # Broadcast scalar MTF
                            return [x.mtf_data.da.idx] * size
                        else:
                            # Scalar number
                            c_mtf = self.to_mtf(x, self.dimension)
                            return [c_mtf.mtf_data.da.idx] * size

                    # Determine result size (simple broadcasting check)
                    a_is_arr = isinstance(inputs[0], np.ndarray)
                    b_is_arr = isinstance(inputs[1], np.ndarray)

                    if a_is_arr and b_is_arr:
                        if inputs[0].size != inputs[1].size:
                            return NotImplemented  # Let numpy handle complex broadcasting if shapes mismatch
                        size = inputs[0].size
                        shape = inputs[0].shape
                    elif a_is_arr:
                        size = inputs[0].size
                        shape = inputs[0].shape
                    else:
                        size = inputs[1].size
                        shape = inputs[1].shape

                    try:
                        idx_a = get_indices(inputs[0], size)
                        idx_b = get_indices(inputs[1], size)

                        # Call Fortran Batch Routine
                        res_indices = BATCH_UFUNC_MAP[ufunc](idx_a, idx_b)

                        # Wrap results back to MTF array
                        # This creation loop is Python-side overhead but necessary
                        res_flat = np.empty(size, dtype=object)
                        for i, idx in enumerate(res_indices):
                            # Create MTF wrapping the result index
                            # Note: owned=True by default in CosyMtfData/CosyDA logic?
                            # We need to construct CosyMtfData manually to inject the index
                            cda = cosy_backend.CosyDA(idx=idx, owned=True)
                            c_data = cosy_backend.CosyMtfData(
                                self.dimension, is_complex=False
                            )  # Complex? logic needed
                            c_data.da = cda
                            res_flat[i] = type(self)(
                                mtf_data=c_data, dimension=self.dimension
                            )

                        return res_flat.reshape(shape)

                    except Exception:
                        # If anything fails (dimensions, types), fallback to slow loop
                        # print(f"Batch optimization failed: {e}")
                        pass

            # -----------------------------------------------------------------
            # Fallback for Object Arrays (NumPy fallback)
            # -----------------------------------------------------------------
            # If we reached here with arrays in inputs, it means batch path was skipped or failed.
            # We must explicitly handle element-wise operations using python operators to support
            # object arrays of MTFs, otherwise returning NotImplemented causes TypeError.
            if has_array:
                # Map ufunc to operator
                op = None
                if ufunc == np.add:
                    op = lambda x, y: x + y
                elif ufunc == np.subtract:
                    op = lambda x, y: x - y
                elif ufunc == np.multiply:
                    op = lambda x, y: x * y
                elif ufunc == np.divide or ufunc == np.true_divide:
                    op = lambda x, y: x / y
                
                if op is not None:
                     return np.vectorize(op, otypes=[object])(*inputs)

            # -----------------------------------------------------------------
            # Scalar / Fallback Path (Wrapped Conversion)
            # -----------------------------------------------------------------
            mtf_inputs = []
            for i in inputs:
                if isinstance(i, MultivariateTaylorFunction):
                    mtf_inputs.append(i)
                else:
                    try:
                        # Use self's dimension as default.
                        mtf_inputs.append(self.to_mtf(i, dimension=self.dimension))
                    except (TypeError, ValueError):
                        return NotImplemented

            # Now `mtf_inputs` has MTF objects for all inputs.

            if ufunc in UNARY_UFUNC_MAP:
                if len(mtf_inputs) == 1:
                    method_name = UNARY_UFUNC_MAP[ufunc]
                    if method_name == "_inv_mtf_internal":
                        # Special case for reciprocal which takes an argument in internal impl
                        # But _inv_mtf_internal(self, mtf_instance) ... wait
                        # self._inv_mtf_internal(mtf_inputs[0])?
                        # It seems _inv_mtf_internal is an instance method that uses `self` as a factory?
                        # Let's check definition.
                        return mtf_inputs[0]._inv_mtf_internal(mtf_inputs[0])
                    op = getattr(mtf_inputs[0], method_name)
                    return op()
                else:
                    return NotImplemented

            if ufunc == np.positive:
                return mtf_inputs[0]
            if ufunc == np.square:
                return mtf_inputs[0] * mtf_inputs[0]

            if ufunc in BINARY_UFUNC_MAP:
                if len(mtf_inputs) == 2:
                    # We need to call the method on the first object.
                    # e.g., mtf_inputs[0].__add__(mtf_inputs[1])
                    method_name = BINARY_UFUNC_MAP[ufunc]
                    op = getattr(mtf_inputs[0], method_name)
                    return op(mtf_inputs[1])
                else:
                    return NotImplemented

            if ufunc in (np.power, np.float_power):
                if len(inputs) == 2 and isinstance(inputs[1], (int, float, np.number)):
                    return mtf_inputs[0] ** inputs[1]

            return NotImplemented

        return NotImplemented

    def __reduce__(self):
        return (
            self.__class__,
            ((self.exponents, self.coeffs), self.dimension, self.var_name),
        )

    def get_constant(self) -> float:
        """
        Retrieves the constant (zeroth-order) term of the Taylor series.

        This is equivalent to evaluating the function at the origin.

        Returns
        -------
        float
            The value of the constant term.

        Examples
        --------
        >>> from sandalwood import mtf
        >>> mtf.initialize_mtf(max_order=2, max_dimension=2) # doctest: +ELLIPSIS
        ...
        >>> x, y = mtf.var(1), mtf.var(2)
        >>> f = 5.0 + x + y**2
        >>> f.get_constant()
        5.0
        """
        if self._IMPLEMENTATION == "cosy" and self.mtf_data is not None:
            if hasattr(self.mtf_data, "get_constant"):
                val = self.mtf_data.get_constant()
                if isinstance(val, complex):
                    return val
                return float(val)

        constant_exp = np.zeros(self.dimension, dtype=np.int32)
        match = np.all(self.exponents == constant_exp, axis=1)
        const_idx = np.where(match)[0]

        if const_idx.size > 0:
            val = self.coeffs[const_idx[0]]
            if isinstance(val, (complex, np.complex128)):
                return val
            return float(val)
        else:
            return 0.0

    def get_polynomial_part(self):
        """
        Returns a new MTF representing the polynomial part of the function.

        The new function contains all terms of order > 0.

        Returns
        -------
        MultivariateTaylorFunction
            A new MTF object with the same dimensions but with the constant
            term removed.

        Examples
        --------
        >>> from sandalwood import mtf
        >>> mtf.initialize_mtf(max_order=2, max_dimension=2) # doctest: +ELLIPSIS
        ...
        >>> x, y = mtf.var(1), mtf.var(2)
        >>> f = 5.0 + x + y**2
        >>> p = f.get_polynomial_part()
        >>> print(p) # doctest: +NORMALIZE_WHITESPACE
                   Coefficient  Order Exponents
        0          1.0      1    (1, 0)
        1          1.0      2    (0, 2)
        """
        constant_exp = np.zeros(self.dimension, dtype=np.int32)
        match = np.all(self.exponents == constant_exp, axis=1)
        poly_mask = ~match

        if not np.any(poly_mask):
            return type(self)(
                (
                    np.empty((0, self.dimension), dtype=np.int32),
                    np.empty((0,), dtype=self.coeffs.dtype),
                ),
                self.dimension,
            )

        poly_exponents = self.exponents[poly_mask]
        poly_coeffs = self.coeffs[poly_mask]
        return type(self)((poly_exponents, poly_coeffs), self.dimension)

    @classmethod
    def from_numpy_array(
        cls, np_array: np.ndarray, dimension: Optional[int] = None
    ) -> np.ndarray:
        """
        Converts a NumPy array of numbers into a NumPy array of mtf
        objects. Each element of the output array is a constant mtf.

        Parameters
        ----------
        np_array : np.ndarray
            The NumPy array of numerical values to convert.
        dimension : int, optional
            The number of variables for each MultivariateTaylorFunction object.
            If None, the globally configured _MAX_DIMENSION is used.

        Returns
        -------
        np.ndarray
            A NumPy array of MultivariateTaylorFunction objects with the same
            shape as the input array.

        Raises
        ------
        TypeError
            If the input is not a NumPy array.
        """
        if not isinstance(np_array, np.ndarray):
            raise TypeError("Input must be a NumPy array.")

        if dimension is None:
            dimension = cls.get_max_dimension()

        # Vectorize the from_constant method to apply it to each element
        # of the input NumPy array while preserving the shape.
        vectorized_from_constant = np.vectorize(cls.from_constant, otypes=[object])
        return vectorized_from_constant(np_array, dimension=dimension)

    @classmethod
    def to_numpy_array(cls, mtf_array: np.ndarray) -> np.ndarray:
        """
        Converts a NumPy array of MultivariateTaylorFunction objects into a
        NumPy array of their constant values.

        Parameters
        ----------
        mtf_array : np.ndarray
            The NumPy array of MultivariateTaylorFunction objects to convert.

        Returns
        -------
        np.ndarray
            A NumPy array of the constant values (zeroth-order terms)
            with the same shape as the input array.

        Raises
        ------
        TypeError
            If the input is not a NumPy array or if its elements are not
            MultivariateTaylorFunction objects.
        """
        if not isinstance(mtf_array, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        if mtf_array.size > 0 and not isinstance(mtf_array.flat[0], cls):
            raise TypeError("All elements of the input array must be mtf objects.")

        # Use the get_constant method, which is a more efficient way to
        # extract the zeroth-order term than evaluating the full series.
        flat_list = [mtf_obj.get_constant() for mtf_obj in mtf_array.flat]

        # Then, reshape the flat list to the original array's shape
        return np.array(flat_list, dtype=np.float64).reshape(mtf_array.shape)


mtf = MultivariateTaylorFunction  # Alias to MultivariateTaylorFunction


def _generate_exponent_combinations(dimension, order):
    """Generates all combinations of exponents for a given dimension and order."""
    if dimension <= 0 or order < 0:
        return []
    if dimension == 1:
        return [(o,) for o in range(order + 1)]
    if order == 0:
        return [(0,) * dimension]
    exponent_combinations = []
    for o in range(order + 1):
        for comb in _generate_exponent_combinations(dimension - 1, o):
            remaining_order = order - o
            for i in range(remaining_order + 1):
                exponent_combinations.append(comb + (i,))
    return exponent_combinations


def _split_constant_polynomial_part(
    input_mtf: MultivariateTaylorFunction,
) -> tuple[float, MultivariateTaylorFunction]:
    """
    Splits an MTF into its constant and polynomial parts.

    This is a common operation in algorithms that use constant factoring,
    such as the elementary function expansions. It separates the function
    `f` into `C + p(x)`, where `C` is the constant term (order 0) and
    `p(x)` contains all terms of order > 0.

    Parameters
    ----------
    input_mtf : MultivariateTaylorFunction
        The function to split.

    Returns
    -------
    tuple[float, MultivariateTaylorFunction]
        A tuple containing:
        - The constant term value (`C`).
        - A new MTF representing the polynomial part (`p(x)`).
    """
    constant_term_C_value = input_mtf.get_constant()
    polynomial_part_mtf = input_mtf.get_polynomial_part()

    return constant_term_C_value, polynomial_part_mtf


def _sqrt_taylor(variable, order: Optional[int] = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of the square root of an MTF.

    This function implements `sqrt(C + p(x))` by factoring out the constant
    term `C` to compute `sqrt(C) * sqrt(1 + p(x)/C)`, where the Taylor
    series for `sqrt(1+u)` is known and can be composed with `p(x)/C`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function to take the square root of. Must have a positive
        constant term.
    order : int, optional
        The truncation order for the resulting Taylor series. If None, the
        global `_MAX_ORDER` is used.

    Returns
    -------
    MultivariateTaylorFunction
        A new MTF representing the square root of the input.

    Raises
    ------
    ValueError
        If the constant term of the input function is not positive.
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    constant_term_C_value, polynomial_part_B_mtf = _split_constant_polynomial_part(
        input_mtf
    )

    # Avoid circular import by importing locally
    from .complex_taylor_function import ComplexMultivariateTaylorFunction

    is_complex_input = isinstance(
        input_mtf, ComplexMultivariateTaylorFunction
    ) or isinstance(constant_term_C_value, complex)

    if abs(constant_term_C_value) < 1e-12:
        raise ValueError(
            "Constant part of input to sqrt_taylor is too close to zero. "
            "Square root is not defined at zero for this expansion."
        )

    if not is_complex_input and constant_term_C_value < 0:
        raise ValueError(
            "Constant part of input to sqrt_taylor is negative. This method "
            "requires a non-negative constant for Real inputs. For complex "
            "roots, use ComplexMultivariateTaylorFunction."
        )

    constant_factor_sqrt_C = cmath.sqrt(constant_term_C_value)
    polynomial_part_x_mtf = polynomial_part_B_mtf / constant_term_C_value
    sqrt_1_plus_x_mtf = sqrt_taylor_1D_expansion(polynomial_part_x_mtf, order=order)
    result_mtf = sqrt_1_plus_x_mtf * constant_factor_sqrt_C
    return result_mtf.truncate(order)


def sqrt_taylor_1D_expansion(
    variable, order: Optional[int] = None
) -> MultivariateTaylorFunction:
    """
    Helper: 1D Taylor expansion of sqrt(1+u) around zero, precomputed coefficients.
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    sqrt_taylor_1d_coefficients = {}
    taylor_dimension_1d = 1
    variable_index_1d = 0

    # Dynamic coefficient generation for sqrt(1+x)
    # n=0: 1
    # n=1: 0.5
    # n>1: a_n = a_(n-1) * (0.5 - (n-1)) / n

    current_coeff = 1.0
    for n_order in range(order + 1):
        if n_order == 0:
            current_coeff = 1.0
        elif n_order == 1:
            current_coeff = 0.5
        else:
            current_coeff = (
                current_coeff
                * (0.5 - (n_order - 1))
                / (n_order - 1)
                * (0.5 - (n_order - 1))
                / n_order
            )  # Wait, previous loop logic was using stored coeff.
            # Let's start clean.
            pass

    coeffs = [0.0] * (order + 1)
    coeffs[0] = 1.0
    if order >= 1:
        coeffs[1] = 0.5
        for n in range(2, order + 1):
            coeffs[n] = coeffs[n - 1] * (0.5 - (n - 1)) / n

    for n_order in range(order + 1):
        if abs(coeffs[n_order]) > 1e-16:
            sqrt_taylor_1d_coefficients[
                _generate_exponent(n_order, variable_index_1d, taylor_dimension_1d)
            ] = np.array([coeffs[n_order]]).reshape(1)

    sqrt_taylor_1d_mtf = type(variable)(
        coefficients=sqrt_taylor_1d_coefficients, dimension=taylor_dimension_1d
    )
    composed_mtf = sqrt_taylor_1d_mtf.compose({1: input_mtf})
    return composed_mtf.truncate(order)
