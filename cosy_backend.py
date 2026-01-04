import os
import sys
from ctypes import CDLL, POINTER, RTLD_GLOBAL, byref, c_double, c_int

import numpy as np

# Path to the shared library
LIB_PATH = os.path.join(os.path.dirname(__file__), "backends/cosy/libcosy.so")

# Load Library
try:
    # Use RTLD_GLOBAL to ensure symbols are available for resolution
    libcosy = CDLL(LIB_PATH, mode=RTLD_GLOBAL)
except OSError as e:
    # Build a dummy check if library is missing during development
    sys.stderr.write(f"Warning: Could not load libcosy.so: {e}\n")

    class DummyLib:
        def __getattr__(self, name):
            def func(*args):
                raise RuntimeError(f"COSY library not found at {LIB_PATH}")

            return func

    libcosy = DummyLib()

# --- Wrapper Signatures ---

# void setup_cosy_(int *order, int *n_vars, int *max_nmmax)
# Note: wrapper.f definition: SUBROUTINE SETUP_COSY(MAX_ORDER, MAX_VARS, OUT_NMMAX)
libcosy.setup_cosy_.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int)]
libcosy.setup_cosy_.restype = None

# void get_da_coeff_(int *idx, int *exponents, double *val)
# SUBROUTINE GET_DA_COEFF(IDX, EXPS, VAL)
libcosy.get_da_coeff_.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_double)]
libcosy.get_da_coeff_.restype = None

# void create_da_var_(int *idx, double *real_val, int *var_id)
libcosy.create_da_var_.argtypes = [POINTER(c_int), POINTER(c_double), POINTER(c_int)]
libcosy.create_da_var_.restype = None

# void create_da_const_(int *idx, double *val)
libcosy.create_da_const_.argtypes = [POINTER(c_int), POINTER(c_double)]
libcosy.create_da_const_.restype = None

# void compute_da_add_(int *idx_a, int *idx_b, int *idx_res)
libcosy.compute_da_add_.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int)]
libcosy.compute_da_add_.restype = None

# void compute_da_sub_(int *idx_a, int *idx_b, int *idx_res)
libcosy.compute_da_sub_.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int)]
libcosy.compute_da_sub_.restype = None

# void compute_da_mul_(int *idx_a, int *idx_b, int *idx_res)
libcosy.compute_da_mul_.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int)]
libcosy.compute_da_mul_.restype = None

# void compute_da_div_(int *idx_a, int *idx_b, int *idx_res)
libcosy.compute_da_div_.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int)]
libcosy.compute_da_div_.restype = None

# Differential Operators
libcosy.compute_da_der_.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int)]
libcosy.compute_da_der_.restype = None

libcosy.compute_da_int_.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int)]
libcosy.compute_da_int_.restype = None

# void da_free_(int *idx) -> FOXDAL
# Use generic foxdal wrapper if exposed or assume internal memory management strategy.
# wrapper.f does NOT yet expose a direct single-var free for python side
# unless we add it. For now, we rely on wrappers managing TEMPS.
# Python-side CosyDA objects own persistent handles allocated by wrappers (Result vars).
# We need a `DELETE_DA_VAR` in wrapper.f to call FOXDAL on the held index.
# For now, memory leaks potentially if we don't free.

# Elementary Functions
libcosy.compute_da_sin_.argtypes = [POINTER(c_int), POINTER(c_int)]
libcosy.compute_da_cos_.argtypes = [POINTER(c_int), POINTER(c_int)]
libcosy.compute_da_exp_.argtypes = [POINTER(c_int), POINTER(c_int)]
libcosy.compute_da_log_.argtypes = [POINTER(c_int), POINTER(c_int)]

# Hyperbolic
libcosy.compute_da_sinh_.argtypes = [POINTER(c_int), POINTER(c_int)]
libcosy.compute_da_cosh_.argtypes = [POINTER(c_int), POINTER(c_int)]
libcosy.compute_da_tanh_.argtypes = [POINTER(c_int), POINTER(c_int)]

libcosy.compute_da_tan_.argtypes = [POINTER(c_int), POINTER(c_int)]

libcosy.compute_da_sqr_.argtypes = [POINTER(c_int), POINTER(c_int)]
libcosy.compute_da_sqrt_.argtypes = [POINTER(c_int), POINTER(c_int)]
libcosy.compute_da_isrt_.argtypes = [POINTER(c_int), POINTER(c_int)]

libcosy.compute_da_erf_.argtypes = [POINTER(c_int), POINTER(c_int)]

libcosy.compute_da_cot_.argtypes = [POINTER(c_int), POINTER(c_int)]

libcosy.compute_da_asin_.argtypes = [POINTER(c_int), POINTER(c_int)]
libcosy.compute_da_acos_.argtypes = [POINTER(c_int), POINTER(c_int)]
libcosy.compute_da_atan_.argtypes = [POINTER(c_int), POINTER(c_int)]

libcosy.compute_da_poi_.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int)]
libcosy.compute_da_est_.argtypes = [
    POINTER(c_int),
    POINTER(c_int),
    POINTER(c_int),
    POINTER(c_double),
]

# Complex wrappers
libcosy.create_cda_var_.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_double)]
libcosy.create_cda_const_.argtypes = [
    POINTER(c_int),
    POINTER(c_double),
    POINTER(c_double),
]
libcosy.compute_cd_add_.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int)]
libcosy.compute_cd_sub_.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int)]
libcosy.compute_cd_mul_.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int)]
libcosy.compute_cd_div_.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int)]
# Transcendental
libcosy.compute_cd_exp_.argtypes = [POINTER(c_int), POINTER(c_int)]
libcosy.compute_cd_sin_.argtypes = [POINTER(c_int), POINTER(c_int)]
libcosy.compute_cd_cos_.argtypes = [POINTER(c_int), POINTER(c_int)]

libcosy.cdre_.argtypes = [POINTER(c_int), POINTER(c_int)]
libcosy.cdim_.argtypes = [POINTER(c_int), POINTER(c_int)]
libcosy.set_cd_parts_.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int)]
libcosy.compute_da_to_cd_.argtypes = [POINTER(c_int), POINTER(c_int)]
# Already defined above


class CosyBackend:
    _initialized = False
    _order = 1
    _dim = 1

    @staticmethod
    def is_initialized():
        return CosyBackend._initialized

    @staticmethod
    def initialize(order, dim):
        CosyBackend._order = order
        CosyBackend._dim = dim

        c_order = c_int(order)
        c_dim = c_int(dim)
        c_nmmax = c_int(0)

        if not hasattr(libcosy, "setup_cosy_"):
            raise RuntimeError(f"COSY library not found at {LIB_PATH}")

        # print("Initializing COSY Backend...")
        libcosy.setup_cosy_(byref(c_order), byref(c_dim), byref(c_nmmax))
        # print(f"COSY Setup Complete. NMMAX={c_nmmax.value}")
        CosyBackend._initialized = True

    @staticmethod
    def reset():
        # COSY internal memory is persistent.
        # We could implement a RESET_COSY subroutine if needed.
        pass

    @staticmethod
    def var(var_index):
        """Creates a COSY DA variable for the given index."""
        return CosyDA(var_id=var_index)


class CosyDA:
    """
    Python wrapper for a COSY DA variable (HANDLE).
    Provides operator overloading to allow natural syntax.
    """

    def __init__(self, create_new=False, idx=None, var_id=None, create_mode=None):
        """
        create_new: if True, calls a wrapper to allocate a new 0.0 constant DA.
                    Wait, wrapper.f currently mostly allocates Result vars inside SUBROUTINES.
                    To support 'temp = CosyDA()', we need a way to just 'alloc'.
        """
        # Compatibility with existing code
        if create_mode == "new":
            create_new = True

        if idx is not None:
            self.idx = idx
            # If we wrap an existing index, we typically don't own it unless specified.
            self.owned = False
        elif create_new:
            # Create a 0.0 constant to get a valid handle
            res_idx = c_int(0)
            c_val = c_double(0.0)
            libcosy.create_da_const_(byref(res_idx), byref(c_val))
            self.idx = res_idx.value
            self.owned = True
        elif var_id is not None:
            # Create variable
            res_idx = c_int(0)
            c_varid = c_int(
                var_id
            )  # COSY is 1-based, and input var_id is assumed 1-based
            c_val = c_double(1.0)  # Default unit
            # Fix argument order to matching wrapper.f: (idx, val, varid)
            print(f"DEBUG: creating DA var with var_id={var_id}")
            libcosy.create_da_var_(byref(res_idx), byref(c_val), byref(c_varid))
            self.idx = res_idx.value
            self.owned = True
        else:
            raise ValueError("Must specify idx, create_new=True, or var_id")

    def __del__(self):
        # TODO: call foxdal_ to free if self.owned
        pass

    def get_all_terms(self):
        # We need to extract all terms.
        max_order = CosyBackend._order
        dim = CosyBackend._dim

        def get_multi_indices(d, n):
            if d == 0:
                return [()]
            if d == 1:
                return [(i,) for i in range(n + 1)]
            res = []
            for i in range(n + 1):
                for sub in get_multi_indices(d - 1, n - i):
                    res.append(sub + (i,))
            return res

        coeffs = []  # List of (tuple_exponents, value)

        c_val = c_double()

        # This is iterative and slow.
        # Ideally COSY would return a list of non-zero terms.
        # For now, iterate all possible combos? max_order=2, dim=2 -> small.
        # max_order=10, dim=20 -> Impossible.
        # But we are testing small cases.
        c_exps = (c_int * dim)()  # Buffer for exponents
        c_val = c_double()

        if not hasattr(libcosy, "get_da_coeff_by_index_"):
            libcosy.get_da_coeff_by_index_ = libcosy.get_da_coeff_by_index_
            libcosy.get_da_coeff_by_index_.argtypes = [
                POINTER(c_int),
                POINTER(c_int),
                POINTER(c_int),
                POINTER(c_double),
            ]
            libcosy.get_da_coeff_by_index_.restype = None

        from math import comb

        n_coeffs = comb(max_order + dim, dim)

        for i in range(1, n_coeffs + 1):
            # Passing linear index i by REF
            # wrapper: GET_DA_COEFF_BY_INDEX(IDX, I, EXPS, COEFF)
            libcosy.get_da_coeff_by_index_(
                byref(c_int(self.idx)), byref(c_int(i)), c_exps, byref(c_val)
            )

            val = c_val.value
            # Convert c_exps (C array) to Python tuple
            exps = tuple(c_exps[k] for k in range(dim))

            if val != 0.0:
                coeffs.append((exps, val))
        return coeffs

    def get_constant(self):
        c_exponents = (c_int * 1000)()
        c_val = c_double()
        libcosy.get_da_coeff_(byref(c_int(self.idx)), c_exponents, byref(c_val))
        return c_val.value

    # --- Arithmetic Overloads ---
    def __add__(self, other):
        res = CosyDA(create_new=True)
        if isinstance(other, CosyDA):
            libcosy.compute_da_add_(
                byref(c_int(self.idx)), byref(c_int(other.idx)), byref(c_int(res.idx))
            )
        else:  # Scalar
            # Create scalar constant DA?
            con = CosyDA(create_new=True)
            c_val = c_double(float(other))
            libcosy.create_da_const_(byref(c_int(con.idx)), byref(c_val))
            libcosy.compute_da_add_(
                byref(c_int(self.idx)), byref(c_int(con.idx)), byref(c_int(res.idx))
            )
        return res

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        res = CosyDA(create_new=True)
        if isinstance(other, CosyDA):
            libcosy.compute_da_sub_(
                byref(c_int(self.idx)), byref(c_int(other.idx)), byref(c_int(res.idx))
            )
        else:
            con = CosyDA(create_new=True)
            c_val = c_double(float(other))
            libcosy.create_da_const_(byref(c_int(con.idx)), byref(c_val))
            libcosy.compute_da_sub_(
                byref(c_int(self.idx)), byref(c_int(con.idx)), byref(c_int(res.idx))
            )
        return res

    def __rsub__(self, other):
        if isinstance(other, (int, float, complex, np.number)):
            # Transition to complex if needed
            if isinstance(other, complex):
                return complex(other) - self.to_complex()

            con = CosyDA(create_new=True)
            c_val = c_double(float(other))
            libcosy.create_da_const_(byref(c_int(con.idx)), byref(c_val))
            res = CosyDA(create_new=True)
            libcosy.compute_da_sub_(
                byref(c_int(con.idx)), byref(c_int(self.idx)), byref(c_int(res.idx))
            )
            return res
        return NotImplemented

    def to_complex(self):
        """Converts Real DA to Complex DA (imaginary part 0)."""
        res = CosyCDA(create_mode="new")
        libcosy.cdcpl_da_to_cd_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def __mul__(self, other):
        if isinstance(other, CosyDA):
            res = CosyDA(create_new=True)
            libcosy.compute_da_mul_(
                byref(c_int(self.idx)), byref(c_int(other.idx)), byref(c_int(res.idx))
            )
            return res
        elif isinstance(other, (int, float, np.number)):
            res = CosyDA(create_new=True)
            con = CosyDA(create_new=True)
            c_val = c_double(float(other))
            libcosy.create_da_const_(byref(c_int(con.idx)), byref(c_val))
            libcosy.compute_da_mul_(
                byref(c_int(self.idx)), byref(c_int(con.idx)), byref(c_int(res.idx))
            )
            return res
        elif isinstance(other, complex):
            # (RE + 0j) * other = (RE*other.real) + j*(RE*other.imag)
            res_re = self * other.real
            res_im = self * other.imag
            res_cd = CosyCDA(create_mode="new")
            libcosy.set_cd_parts_(
                byref(c_int(res_cd.idx)),
                byref(c_int(res_re.idx)),
                byref(c_int(res_im.idx)),
            )
            return res_cd
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        res = CosyDA(create_new=True)
        if isinstance(other, CosyDA):
            libcosy.compute_da_div_(
                byref(c_int(self.idx)), byref(c_int(other.idx)), byref(c_int(res.idx))
            )
        else:
            con = CosyDA(create_new=True)
            c_val = c_double(float(other))
            libcosy.create_da_const_(byref(c_int(con.idx)), byref(c_val))
            # self / other
            # But here other is scalar. division by scalar = mul by 1/scalar.
            # Or use DIV:
            libcosy.compute_da_div_(
                byref(c_int(self.idx)), byref(c_int(con.idx)), byref(c_int(res.idx))
            )
        return res

    def __rtruediv__(self, other):
        con = CosyDA(create_new=True)
        c_val = c_double(float(other))
        libcosy.create_da_const_(byref(c_int(con.idx)), byref(c_val))

        res = CosyDA(create_new=True)
        libcosy.compute_da_div_(
            byref(c_int(con.idx)), byref(c_int(self.idx)), byref(c_int(res.idx))
        )
        return res

    # --- Differentiator ---
    def deriv(self, var_id):
        res = CosyDA(create_new=True)
        # Cosy 1-based index
        libcosy.compute_da_der_(
            byref(c_int(self.idx)), byref(c_int(var_id + 1)), byref(c_int(res.idx))
        )
        return res

    def integral(self, var_id):
        res = CosyDA(create_new=True)
        libcosy.compute_da_int_(
            byref(c_int(self.idx)), byref(c_int(var_id + 1)), byref(c_int(res.idx))
        )
        return res

    def estimate(self):
        val = c_double()
        libcosy.compute_da_daest_(byref(c_int(self.idx)), byref(val))
        return val.value

    # --- Funcs ---
    def sin(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_sin_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def cos(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_cos_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def exp(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_exp_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def log(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_log_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def sinh(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_sinh_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def cosh(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_cosh_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def tanh(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_tanh_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def tan(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_tan_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def sinc(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_sinc_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def sqr(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_sqr_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def sqrt(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_sqrt_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def minv(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_minv_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def arcsin(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_arcsin_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def arccos(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_arccos_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def arctan(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_arctan_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def erf(self):
        res = CosyDA(create_mode="new")
        libcosy.compute_da_erf_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def cot(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_cot_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def coth(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_coth_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def isrt3(self):  # Fallback or needs implementation?
        return NotImplemented

    def isqrt(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_isrt_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    # Inverse Hyperbolic (New wrappers)
    def asinh(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_asinh_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def acosh(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_acosh_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def atanh(self):
        res = CosyDA(create_new=True)
        libcosy.compute_da_atanh_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def poisson(self, other):
        if not isinstance(other, CosyDA):
            return NotImplemented
        res = CosyDA(create_new=True)
        libcosy.compute_da_poi_(
            byref(c_int(self.idx)), byref(c_int(other.idx)), byref(c_int(res.idx))
        )
        return res


class CosyCDA(CosyDA):
    # Inherits for generic destructors.

    def __init__(
        self,
        create_new=False,
        idx=None,
        from_var=None,
        from_const=None,
        create_mode=None,
    ):
        """
        create_new: if True, allocates a new CD variable.
        create_mode: 'new' is equivalent to create_new=True. Added for compatibility with math wrappers.
        from_var: tuple (var_id, real_val) to create a variable.
        from_const: tuple (re_val, im_val) to create a constant.
        idx: wrap existing index.
        """
        if create_mode == "new":
            create_new = True

        if idx is not None:
            self.idx = idx
            self.owned = False
            # If explicit ownership transfer needed, handle separately.
        elif create_new and idx is None:
            self.owned = True
            res_idx = c_int(0)
            # Allocate 0+0i
            libcosy.create_cda_const_(
                byref(res_idx), byref(c_double(0)), byref(c_double(0))
            )
            self.idx = res_idx.value
        elif from_var:
            var_id, real_val = from_var
            res_idx = c_int(0)
            libcosy.create_cda_var_(
                byref(res_idx), byref(c_int(var_id)), byref(c_double(real_val))
            )
            self.idx = res_idx.value
            self.owned = True
        elif from_const:
            re_val, im_val = from_const
            res_idx = c_int(0)
            libcosy.create_cda_const_(
                byref(res_idx), byref(c_double(re_val)), byref(c_double(im_val))
            )
            self.idx = res_idx.value
            self.owned = True
        else:
            # Default fallback
            raise ValueError(
                "Must provide idx, create_new=True, from_var, or from_const"
            )

    def get_constant(self):
        """Returns the complex constant term (coeff of 0th order)."""

        # Temp vars for real/im separate extraction
        re_da = CosyDA(create_mode="new")
        im_da = CosyDA(create_mode="new")

        libcosy.cdre_(byref(c_int(self.idx)), byref(c_int(re_da.idx)))
        libcosy.cdim_(byref(c_int(self.idx)), byref(c_int(im_da.idx)))

        c_exponents = (c_int * 1000)()  # Zero init = 0 exponents
        c_val_re = c_double()
        c_val_im = c_double()

        libcosy.get_da_coeff_(byref(c_int(re_da.idx)), c_exponents, byref(c_val_re))
        libcosy.get_da_coeff_(byref(c_int(im_da.idx)), c_exponents, byref(c_val_im))

        return complex(c_val_re.value, c_val_im.value)

    def __add__(self, other):
        res = CosyCDA(create_mode="new")
        if isinstance(other, CosyCDA):
            libcosy.compute_cd_add_(
                byref(c_int(self.idx)), byref(c_int(other.idx)), byref(c_int(res.idx))
            )
        elif isinstance(other, CosyDA):
            other_cd = other.to_complex()
            libcosy.compute_cd_add_(
                byref(c_int(self.idx)),
                byref(c_int(other_cd.idx)),
                byref(c_int(res.idx)),
            )
        elif isinstance(other, (int, float, complex, np.number)):
            c = CosyCDA(create_mode="new")
            libcosy.create_cda_const_(
                byref(c_int(c.idx)),
                byref(c_double(other.real)),
                byref(c_double(other.imag)),
            )
            libcosy.compute_cd_add_(
                byref(c_int(self.idx)), byref(c_int(c.idx)), byref(c_int(res.idx))
            )
        return res

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        res = CosyCDA(create_mode="new")
        if isinstance(other, CosyCDA):
            libcosy.compute_cd_sub_(
                byref(c_int(self.idx)), byref(c_int(other.idx)), byref(c_int(res.idx))
            )
        elif isinstance(other, (int, float, complex, np.number)):
            c = CosyCDA(create_mode="new")
            libcosy.create_cda_const_(
                byref(c_int(c.idx)),
                byref(c_double(other.real)),
                byref(c_double(other.imag)),
            )
            libcosy.compute_cd_sub_(
                byref(c_int(self.idx)), byref(c_int(c.idx)), byref(c_int(res.idx))
            )
        return res

    def __mul__(self, other):
        res = CosyCDA(create_mode="new")
        if isinstance(other, CosyCDA):
            libcosy.compute_cd_mul_(
                byref(c_int(self.idx)), byref(c_int(other.idx)), byref(c_int(res.idx))
            )
        elif isinstance(other, CosyDA):
            other_cd = other.to_complex()
            libcosy.compute_cd_mul_(
                byref(c_int(self.idx)),
                byref(c_int(other_cd.idx)),
                byref(c_int(res.idx)),
            )
        elif isinstance(other, (int, float, complex, np.number)):
            c = CosyCDA(create_mode="new")
            libcosy.create_cda_const_(
                byref(c_int(c.idx)),
                byref(c_double(other.real)),
                byref(c_double(other.imag)),
            )
            libcosy.compute_cd_mul_(
                byref(c_int(self.idx)), byref(c_int(c.idx)), byref(c_int(res.idx))
            )
        return res

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        res = CosyCDA(create_mode="new")
        if isinstance(other, CosyCDA):
            libcosy.compute_cd_div_(
                byref(c_int(self.idx)), byref(c_int(other.idx)), byref(c_int(res.idx))
            )
        elif isinstance(other, (int, float, complex, np.number)):
            c = CosyCDA(create_mode="new")
            libcosy.create_cda_const_(
                byref(c_int(c.idx)),
                byref(c_double(other.real)),
                byref(c_double(other.imag)),
            )
            libcosy.compute_cd_div_(
                byref(c_int(self.idx)), byref(c_int(c.idx)), byref(c_int(res.idx))
            )
        return res

    # --- Complex Math Methods ---
    def exp(self):
        res = CosyCDA(create_mode="new")
        libcosy.compute_cd_exp_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def sin(self):
        res = CosyCDA(create_mode="new")
        libcosy.compute_cd_sin_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def cos(self):
        res = CosyCDA(create_mode="new")
        libcosy.compute_cd_cos_(byref(c_int(self.idx)), byref(c_int(res.idx)))
        return res

    def get_all_terms(self):
        # Use fast linear scan via wrapper
        max_order = CosyBackend._order
        dim = CosyBackend._dim

        if not hasattr(libcosy, "get_cda_coeff_by_index_"):
            libcosy.get_cda_coeff_by_index_ = libcosy.get_cda_coeff_by_index_
            # IDX, I, EXPS, RE, IM
            libcosy.get_cda_coeff_by_index_.argtypes = [
                POINTER(c_int),
                POINTER(c_int),
                POINTER(c_int),
                POINTER(c_double),
                POINTER(c_double),
            ]
            libcosy.get_cda_coeff_by_index_.restype = None

        from math import comb

        n_coeffs = comb(max_order + dim, dim)

        coeffs = []
        c_exps = (c_int * dim)()
        c_val_re = c_double()
        c_val_im = c_double()

        for i in range(1, n_coeffs + 1):
            # Passing linear index i by REF
            libcosy.get_cda_coeff_by_index_(
                byref(c_int(self.idx)),
                byref(c_int(i)),
                c_exps,
                byref(c_val_re),
                byref(c_val_im),
            )

            re = c_val_re.value
            im = c_val_im.value

            if abs(re) > 1e-15 or abs(im) > 1e-15:
                exps = tuple(c_exps[k] for k in range(dim))
                val = complex(re, im)
                coeffs.append((exps, val))

        return coeffs
