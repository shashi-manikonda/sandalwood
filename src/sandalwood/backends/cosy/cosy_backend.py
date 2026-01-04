import os
import sys
from ctypes import CDLL, POINTER, RTLD_GLOBAL, byref, c_double, c_int

import numpy as np

# Path to the shared library
LIB_PATH = os.path.join(os.path.dirname(__file__), "libcosy.so")

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

def bind_cosy_func(name, argtypes):
    try:
        f = getattr(libcosy, name)
        f.argtypes = argtypes
        f.restype = None
        return f
    except AttributeError:
        # Try without trailing underscore if needed
        try:
             f = getattr(libcosy, name + "_")
             f.argtypes = argtypes
             f.restype = None
             return f
        except AttributeError:
             print(f"Warning: COSY function {name} not found.")
             return None

# Core
bind_cosy_func("setup_cosy_", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("get_da_coeff_", [POINTER(c_int), POINTER(c_int), POINTER(c_double)])
bind_cosy_func("create_da_var_", [POINTER(c_int), POINTER(c_double), POINTER(c_int)])
bind_cosy_func("create_da_const_", [POINTER(c_int), POINTER(c_double)])
bind_cosy_func("get_da_coeff_by_index_", [POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_double)])
bind_cosy_func("cosy_free_", [POINTER(c_int)])
bind_cosy_func("cosy_set_coeffs_", [POINTER(c_int), POINTER(c_double), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("eval_da_", [POINTER(c_int), POINTER(c_double), POINTER(c_double)])

# Arithmetic
bind_cosy_func("compute_da_add_", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_sub_", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_mul_", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_div_", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])

# Differential Operators
bind_cosy_func("compute_da_der_", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_int_", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_poi_", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])

# Elementary Functions
bind_cosy_func("compute_da_sin_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_cos_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_exp_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_log_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_sinh_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_cosh_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_tanh_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_tan_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_sqr_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_sqrt_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_isrt_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_erf_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_cot_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_asin_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_acos_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_atan_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_daest_", [POINTER(c_int), POINTER(c_double)])
bind_cosy_func("compute_da_coth_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_asinh_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_acosh_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_atanh_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_minv_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_isrt3_", [POINTER(c_int), POINTER(c_int)])

# Complex wrappers
bind_cosy_func("create_cda_var_", [POINTER(c_int), POINTER(c_int), POINTER(c_double)])
bind_cosy_func("create_cda_const_", [POINTER(c_int), POINTER(c_double), POINTER(c_double)])
bind_cosy_func("compute_cd_add_", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_sub_", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_mul_", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_div_", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_exp_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_sin_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_cos_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("get_cda_re_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("get_cda_im_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("set_cd_parts_", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_to_cd_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("get_cda_coeff_by_index_", [POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_double), POINTER(c_double)])


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
        libcosy.setup_cosy_(byref(c_order), byref(c_dim), byref(c_nmmax))
        CosyBackend._initialized = True

    @staticmethod
    def reset():
        pass

    @staticmethod
    def var(var_index):
        return CosyDA(var_id=var_index)


class CosyDA:
    def __init__(self, create_new=False, idx=None, var_id=None, create_mode=None, owned=False):
        self.owned = owned
        if create_mode == "new":
            create_new = True
        if idx is not None:
            self.idx = idx
        elif create_new:
            self.owned = True
            res_idx = c_int(0)
            libcosy.create_da_const_(byref(res_idx), byref(c_double(0.0)))
            self.idx = res_idx.value
        elif var_id is not None:
            self.owned = True
            res_idx = c_int(0)
            libcosy.create_da_var_(byref(res_idx), byref(c_double(1.0)), byref(c_int(var_id)))
            self.idx = res_idx.value
        else:
            raise ValueError("Must provide idx, create_new=True, or var_id")

    def __del__(self):
        if hasattr(self, "idx") and self.idx > 0:
            try:
                if self.owned and libcosy and hasattr(libcosy, "cosy_free_") and libcosy.cosy_free_:
                    libcosy.cosy_free_(byref(c_int(self.idx)))
            except Exception as e:
                sys.stderr.write(f"Warning: Error freeing COSY DA handle {self.idx}: {e}\n")
        self.idx = 0

    def get_all_terms(self):
        max_order = CosyBackend._order
        dim = CosyBackend._dim
        from math import comb
        n_coeffs = comb(max_order + dim, dim)
        coeffs = []
        c_exps = (c_int * dim)()
        c_val = c_double()
        for i in range(1, n_coeffs + 1):
            libcosy.get_da_coeff_by_index_(byref(c_int(self.idx)), byref(c_int(i)), c_exps, byref(c_val))
            val = c_val.value
            if val != 0.0:
                exps = tuple(c_exps[k] for k in range(dim))
                coeffs.append((exps, val))
        return coeffs

    def get_constant(self):
        c_exponents = (c_int * 1000)()
        c_val = c_double()
        libcosy.get_da_coeff_(byref(c_int(self.idx)), c_exponents, byref(c_val))
        return c_val.value

    def __add__(self, other):
        res_idx = c_int(0)
        if isinstance(other, CosyDA):
            libcosy.compute_da_add_(byref(c_int(self.idx)), byref(c_int(other.idx)), byref(res_idx))
        else:
            con = CosyDA(create_new=True)
            libcosy.create_da_const_(byref(c_int(con.idx)), byref(c_double(float(other))))
            libcosy.compute_da_add_(byref(c_int(self.idx)), byref(c_int(con.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        res_idx = c_int(0)
        if isinstance(other, CosyDA):
            libcosy.compute_da_sub_(byref(c_int(self.idx)), byref(c_int(other.idx)), byref(res_idx))
        else:
            con = CosyDA(create_new=True)
            libcosy.create_da_const_(byref(c_int(con.idx)), byref(c_double(float(other))))
            libcosy.compute_da_sub_(byref(c_int(self.idx)), byref(c_int(con.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def __rsub__(self, other):
        res_idx = c_int(0)
        con = CosyDA(create_new=True)
        libcosy.create_da_const_(byref(c_int(con.idx)), byref(c_double(float(other))))
        libcosy.compute_da_sub_(byref(c_int(con.idx)), byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def __neg__(self):
        res_idx = c_int(0)
        con = CosyDA(create_new=True)
        libcosy.create_da_const_(byref(c_int(con.idx)), byref(c_double(0.0)))
        libcosy.compute_da_sub_(byref(c_int(con.idx)), byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def to_complex(self):
        res_idx = c_int(0)
        libcosy.compute_da_to_cd_(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def __mul__(self, other):
        res_idx = c_int(0)
        if isinstance(other, CosyDA):
            libcosy.compute_da_mul_(byref(c_int(self.idx)), byref(c_int(other.idx)), byref(res_idx))
        elif isinstance(other, (int, float, np.number)):
            con = CosyDA(create_new=True)
            libcosy.create_da_const_(byref(c_int(con.idx)), byref(c_double(float(other))))
            libcosy.compute_da_mul_(byref(c_int(self.idx)), byref(c_int(con.idx)), byref(res_idx) )
        elif isinstance(other, complex):
            return self.to_complex() * other
        else:
            return NotImplemented
        return CosyDA(idx=res_idx.value, owned=True)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        res_idx = c_int(0)
        if isinstance(other, CosyDA):
            libcosy.compute_da_div_(byref(c_int(self.idx)), byref(c_int(other.idx)), byref(res_idx))
        else:
            con = CosyDA(create_new=True)
            libcosy.create_da_const_(byref(c_int(con.idx)), byref(c_double(float(other))))
            libcosy.compute_da_div_(byref(c_int(self.idx)), byref(c_int(con.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def __rtruediv__(self, other):
        res_idx = c_int(0)
        con = CosyDA(create_new=True)
        libcosy.create_da_const_(byref(c_int(con.idx)), byref(c_double(float(other))))
        libcosy.compute_da_div_(byref(c_int(con.idx)), byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def deriv(self, var_id):
        res_idx = c_int(0)
        libcosy.compute_da_der_(byref(c_int(self.idx)), byref(c_int(var_id + 1)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def integral(self, var_id):
        res_idx = c_int(0)
        libcosy.compute_da_int_(byref(c_int(self.idx)), byref(c_int(var_id + 1)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def sin(self):
        res_idx = c_int(0)
        libcosy.compute_da_sin_(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def cos(self):
        res_idx = c_int(0)
        libcosy.compute_da_cos_(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def exp(self):
        res_idx = c_int(0)
        libcosy.compute_da_exp_(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def log(self):
        res_idx = c_int(0)
        libcosy.compute_da_log_(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def sinh(self):
        res_idx = c_int(0)
        libcosy.compute_da_sinh_(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def cosh(self):
        res_idx = c_int(0)
        libcosy.compute_da_cosh_(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def tanh(self):
        res_idx = c_int(0)
        libcosy.compute_da_tanh_(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def tan(self):
        res_idx = c_int(0)
        libcosy.compute_da_tan_(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def sqrt(self):
        res_idx = c_int(0)
        libcosy.compute_da_sqrt_(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def arcsin(self):
        res_idx = c_int(0)
        libcosy.compute_da_asin_(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def arccos(self):
        res_idx = c_int(0)
        libcosy.compute_da_acos_(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def arctan(self):
        res_idx = c_int(0)
        libcosy.compute_da_atan_(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def erf(self):
        res_idx = c_int(0)
        libcosy.compute_da_erf_(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def cot(self):
        res_idx = c_int(0)
        libcosy.compute_da_cot_(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)


class CosyCDA(CosyDA):
    def __init__(self, create_new=False, idx=None, from_var=None, from_const=None, create_mode=None, owned=False):
        self.owned = owned
        if create_mode == "new":
            create_new = True
        if idx is not None:
            self.idx = idx
        elif create_new:
            self.owned = True
            res_idx = c_int(0)
            libcosy.create_cda_const_(byref(res_idx), byref(c_double(0.0)), byref(c_double(0.0)))
            self.idx = res_idx.value
        elif from_var:
            var_id, real_val = from_var
            self.owned = True
            res_idx = c_int(0)
            libcosy.create_cda_var_(byref(res_idx), byref(c_int(var_id)), byref(c_double(real_val)))
            self.idx = res_idx.value
        elif from_const:
            re_val, im_val = from_const
            self.owned = True
            res_idx = c_int(0)
            libcosy.create_cda_const_(byref(res_idx), byref(c_double(re_val)), byref(c_double(im_val)))
            self.idx = res_idx.value
        else:
            raise ValueError("Must provide idx, create_new=True, from_var, or from_const")

    def get_constant(self):
        re_da = CosyDA(create_new=True)
        im_da = CosyDA(create_new=True)
        libcosy.get_cda_re_(byref(c_int(self.idx)), byref(c_int(re_da.idx)))
        libcosy.get_cda_im_(byref(c_int(self.idx)), byref(c_int(im_da.idx)))
        return complex(re_da.get_constant(), im_da.get_constant())

    def get_all_terms(self):
        max_order = CosyBackend._order
        dim = CosyBackend._dim
        from math import comb
        n_coeffs = comb(max_order + dim, dim)
        coeffs = []
        c_exps = (c_int * dim)()
        c_val_re = c_double()
        c_val_im = c_double()
        for i in range(1, n_coeffs + 1):
            libcosy.get_cda_coeff_by_index_(byref(c_int(self.idx)), byref(c_int(i)), c_exps, byref(c_val_re), byref(c_val_im))
            re, im = c_val_re.value, c_val_im.value
            if abs(re) > 1e-15 or abs(im) > 1e-15:
                exps = tuple(c_exps[k] for k in range(dim))
                coeffs.append((exps, complex(re, im)))
        return coeffs

    def __add__(self, other):
        res_idx = c_int(0)
        if isinstance(other, CosyCDA):
            libcosy.compute_cd_add_(byref(c_int(self.idx)), byref(c_int(other.idx)), byref(res_idx))
        elif isinstance(other, CosyDA):
            other_cd = other.to_complex()
            libcosy.compute_cd_add_(byref(c_int(self.idx)), byref(c_int(other_cd.idx)), byref(res_idx))
        else:
            re_v = float(other.real) if hasattr(other, 'real') else float(other)
            im_v = float(other.imag) if hasattr(other, 'imag') else 0.0
            con = CosyCDA(from_const=(re_v, im_v))
            libcosy.compute_cd_add_(byref(c_int(self.idx)), byref(c_int(con.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def __mul__(self, other):
        res_idx = c_int(0)
        if isinstance(other, CosyCDA):
            libcosy.compute_cd_mul_(byref(c_int(self.idx)), byref(c_int(other.idx)), byref(res_idx))
        elif isinstance(other, CosyDA):
            other_cd = other.to_complex()
            libcosy.compute_cd_mul_(byref(c_int(self.idx)), byref(c_int(other_cd.idx)), byref(res_idx))
        else:
            re_v = float(other.real) if hasattr(other, 'real') else float(other)
            im_v = float(other.imag) if hasattr(other, 'imag') else 0.0
            con = CosyCDA(from_const=(re_v, im_v))
            libcosy.compute_cd_mul_(byref(c_int(self.idx)), byref(c_int(con.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)


class CosyBackendManager:
    @staticmethod
    def initialize(order: int, dimension: int):
        CosyBackend.initialize(order, dimension)
    @staticmethod
    def check_init():
        if not CosyBackend.is_initialized():
            raise RuntimeError("COSY backend not initialized.")
        return True


class CosyMtfData:
    def __init__(self, dimension: int = None):
        CosyBackendManager.check_init()
        self.da = CosyDA(create_new=True)
        self.dimension = CosyBackend._dim

    def copy(self):
        new_obj = CosyMtfData(self.dimension)
        new_obj.da = self.da + 0.0
        return new_obj

    def from_numpy(self, exponents: np.ndarray, coeffs: np.ndarray):
        if len(coeffs) == 0: return
        flat_coeffs = coeffs.astype(np.float64)
        c_coeffs = (c_double * len(flat_coeffs))(*flat_coeffs)
        flat_exps = exponents.flatten().astype(np.int32)
        c_exps = (c_int * len(flat_exps))(*flat_exps)
        libcosy.cosy_set_coeffs_(byref(c_int(self.da.idx)), c_coeffs, c_exps, byref(c_int(len(coeffs))))

    def to_dict(self):
        terms = self.da.get_all_terms()
        data = {}
        if not terms:
            data["exponents"] = np.empty((0, self.dimension), dtype=int)
            data["coeffs"] = np.array([])
            return data
        data["exponents"] = np.array([t[0] for t in terms])
        data["coeffs"] = np.array([t[1] for t in terms])
        return data

    def eval(self, point):
        c_point = (c_double * len(point))(*point)
        res = c_double()
        libcosy.eval_da_(byref(c_int(self.da.idx)), c_point, byref(res))
        return res.value

    def add(self, other):
        res = CosyMtfData(self.dimension)
        res.da = self.da + other.da
        return res

    def subtract(self, other):
        res = CosyMtfData(self.dimension)
        res.da = self.da - other.da
        return res

    def multiply(self, other):
        res = CosyMtfData(self.dimension)
        res.da = self.da * other.da
        return res

    def multiply_inplace(self, other):
        self.da = self.da * other.da

    def divide(self, other):
        res = CosyMtfData(self.dimension)
        res.da = self.da / other.da
        return res

    def negate(self):
        res = CosyMtfData(self.dimension)
        res.da = -self.da
        return res

    def partial_derivative(self, deriv_dim):
        res = CosyMtfData(self.dimension)
        res.da = self.da.deriv(deriv_dim - 1)
        return res

    def integrate(self, var_idx):
        res = CosyMtfData(self.dimension)
        res.da = self.da.integral(var_idx - 1)
        return res

    def sin(self):
        res = CosyMtfData(self.dimension)
        res.da = self.da.sin()
        return res

    def cos(self):
        res = CosyMtfData(self.dimension)
        res.da = self.da.cos()
        return res

    def tan(self):
        res = CosyMtfData(self.dimension)
        res.da = self.da.tan()
        return res

    def exp(self):
        res = CosyMtfData(self.dimension)
        res.da = self.da.exp()
        return res

    def log(self):
        res = CosyMtfData(self.dimension)
        res.da = self.da.log()
        return res

    def sqrt(self):
        res = CosyMtfData(self.dimension)
        res.da = self.da.sqrt()
        return res

    def asin(self):
        res = CosyMtfData(self.dimension)
        res.da = self.da.arcsin()
        return res

    def acos(self):
        res = CosyMtfData(self.dimension)
        res.da = self.da.arccos()
        return res

    def atan(self):
        res = CosyMtfData(self.dimension)
        res.da = self.da.arctan()
        return res

    def sinh(self):
        res = CosyMtfData(self.dimension)
        res.da = self.da.sinh()
        return res

    def cosh(self):
        res = CosyMtfData(self.dimension)
        res.da = self.da.cosh()
        return res

    def tanh(self):
        res = CosyMtfData(self.dimension)
        res.da = self.da.tanh()
        return res
