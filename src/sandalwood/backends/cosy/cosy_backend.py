import os
import sys
from ctypes import CDLL, POINTER, RTLD_GLOBAL, byref, c_double, c_int

import numpy as np

# Path to the shared library (platform-aware)
if sys.platform == "win32":
    LIB_NAME = "libcosy.dll"
elif sys.platform == "darwin":
    LIB_NAME = "libcosy.dylib"
else:
    LIB_NAME = "libcosy.so"

LIB_PATH = os.path.join(os.path.dirname(__file__), LIB_NAME)

# Load Library
try:
    # Use RTLD_GLOBAL to ensure symbols are available for resolution if supported
    mode = getattr(os, "RTLD_GLOBAL", 0)
    libcosy = CDLL(LIB_PATH, mode=mode)
except OSError as e:
    # Build a dummy check if library is missing during development
    sys.stderr.write(f"Warning: Could not load {LIB_NAME}: {e}\n")

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

# ... (Previous bindings)
bind_cosy_func("get_mem_state_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("set_mem_state_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("get_all_coeffs_flat_", [POINTER(c_int), POINTER(c_double), POINTER(c_int), POINTER(c_int), POINTER(c_int)])

# Control flags
_USE_OMP = bool(os.environ.get("COSY_USE_OMP", "0"))

class CosyScope:
    def __init__(self):
        self.ivar = c_int(0)
        self.imem = c_int(0)
        
    def __enter__(self):
        if not CosyBackend.is_initialized():
            raise RuntimeError("COSY backend not initialized")
        libcosy.get_mem_state_(byref(self.ivar), byref(self.imem))
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        libcosy.set_mem_state_(byref(self.ivar), byref(self.imem))
        return False

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
            # Use 0.0 as default value for variables (Monomial x_i)
            # Convert 0-based Python index to 1-based COSY index
            libcosy.create_da_var_(byref(res_idx), byref(c_double(0.0)), byref(c_int(var_id + 1)))
            self.idx = res_idx.value
        else:
            raise ValueError("Must provide idx, create_new=True, or var_id")

    def __del__(self):
        # With CosyScope, manual freeing is dangerous if the scope already rewound.
        # But for global variables (outside scope), we still need it.
        # Ideally, we should track if we are inside a scope.
        # For now, we rely on the user to be careful or the OS to reclaim eventually.
        # If we are using Scope, we should probably NOT free individually.
        pass  # Disabled manual free to rely on Scope or OS. 
              # CAUTION: This means without Scope, we leak until reset.
              # But with COSY stack allocator, individual free only works for TOP element anyway.
              # So individual free was already broken for non-top elements.

    def get_all_terms(self):
        max_order = CosyBackend._order
        dim = CosyBackend._dim
        from math import comb
        n_coeffs_est = comb(max_order + dim, dim)
        
        # Use batch getter
        max_len = c_int(n_coeffs_est + 100) # Buffer
        c_vals = (c_double * max_len.value)()
        c_exps = (c_int * (max_len.value * CosyBackend._dim))()
        actual_len = c_int(0)
        
        libcosy.get_all_coeffs_flat_(byref(c_int(self.idx)), c_vals, c_exps, byref(max_len), byref(actual_len))
        
        n_terms = actual_len.value
        coeffs = []
        for i in range(n_terms):
            val = c_vals[i]
            # Exponents are flattened: [e1_1, e1_2... e1_d, e2_1...]
            base_idx = i * CosyBackend._dim
            exps = tuple(c_exps[base_idx + k] for k in range(CosyBackend._dim))
            coeffs.append((exps, val))
            
        return coeffs

    def get_constant(self):
        c_exponents = (c_int * 1000)()
        c_val = c_double()
        libcosy.get_da_coeff_(byref(c_int(self.idx)), c_exponents, byref(c_val))
        return c_val.value

    @classmethod
    def from_const(cls, val):
        res_idx = c_int(0)
        libcosy.create_da_const_(byref(res_idx), byref(c_double(float(val))))
        return cls(idx=res_idx.value, owned=True)

    def __add__(self, other):
        res_idx = c_int(0)
        if isinstance(other, CosyDA):
            libcosy.compute_da_add_(byref(c_int(self.idx)), byref(c_int(other.idx)), byref(res_idx))
        else:
            con = CosyDA.from_const(other)
            libcosy.compute_da_add_(byref(c_int(self.idx)), byref(c_int(con.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        res_idx = c_int(0)
        if isinstance(other, CosyDA):
            libcosy.compute_da_sub_(byref(c_int(self.idx)), byref(c_int(other.idx)), byref(res_idx))
        else:
            con = CosyDA.from_const(other)
            libcosy.compute_da_sub_(byref(c_int(self.idx)), byref(c_int(con.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def __rsub__(self, other):
        res_idx = c_int(0)
        con = CosyDA.from_const(other)
        libcosy.compute_da_sub_(byref(c_int(con.idx)), byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def __neg__(self):
        res_idx = c_int(0)
        con = CosyDA.from_const(0.0)
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
            con = CosyDA.from_const(other)
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
            con = CosyDA.from_const(other)
            libcosy.compute_da_div_(byref(c_int(self.idx)), byref(c_int(con.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def __rtruediv__(self, other):
        res_idx = c_int(0)
        con = CosyDA.from_const(other)
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
    
    # __del__ is inherited, so it does nothing (which is good for Scope)

    def get_constant(self):
        re_da = CosyDA(create_new=True)
        im_da = CosyDA(create_new=True)
        libcosy.get_cda_re_(byref(c_int(self.idx)), byref(c_int(re_da.idx)))
        libcosy.get_cda_im_(byref(c_int(self.idx)), byref(c_int(im_da.idx)))
        return complex(re_da.get_constant(), im_da.get_constant())

    def get_all_terms(self):
        # Complex batch getting not yet fully implemented in wrapper.
        # Fallback to iterative method.
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
    def __init__(self, dimension: int = None, is_complex: bool = False):
        CosyBackendManager.check_init()
        if is_complex:
            self.da = CosyCDA(create_new=True)
        else:
            self.da = CosyDA(create_new=True)
        
        # Physical dimension is fixed by backend
        self.physical_dimension = CosyBackend._dim
        # Logical dimension can be smaller
        if dimension is not None:
             if dimension > self.physical_dimension:
                 raise ValueError(f"Requested dimension {dimension} > COSY backend dimension {self.physical_dimension}")
             self.dimension = dimension
        else:
             self.dimension = self.physical_dimension

    def copy(self):
        # Determine if complex by checking type of self.da
        is_complex = isinstance(self.da, CosyCDA)
        new_obj = CosyMtfData(self.dimension, is_complex=is_complex)
        new_obj.da = self.da + 0.0
        return new_obj

    def from_numpy(self, exponents: np.ndarray, coeffs: np.ndarray):
        if len(coeffs) == 0: return
        
        current_dim = exponents.shape[1]
        
        # Pad exponents if necessary to match global COSY dimension
        global_dim = self.physical_dimension
        
        if current_dim < global_dim:
            padded_exponents = np.zeros((exponents.shape[0], global_dim), dtype=np.int32)
            padded_exponents[:, :current_dim] = exponents
            flat_exps = padded_exponents.flatten()
        elif current_dim > global_dim:
            raise ValueError(f"Exponents dimension {current_dim} exceeds COSY global dimension {global_dim}")
        else:
            flat_exps = exponents.flatten().astype(np.int32)
            
        c_exps = (c_int * len(flat_exps))(*flat_exps)

        if isinstance(self.da, CosyCDA) or np.iscomplexobj(coeffs):
            # Ensure we have complex DA
            if not isinstance(self.da, CosyCDA):
                 old_idx = self.da.idx
                 self.da = CosyCDA(create_mode="new") 
            
            flat_coeffs = coeffs.astype(np.complex128).flatten()
            flat_re = flat_coeffs.real.astype(np.float64)
            flat_im = flat_coeffs.imag.astype(np.float64)
            
            c_re = (c_double * len(flat_re))(*flat_re)
            c_im = (c_double * len(flat_im))(*flat_im)
            
            if not hasattr(libcosy, 'cosy_set_cd_coeffs_'):
                 bind_cosy_func("cosy_set_cd_coeffs_", [POINTER(c_int), POINTER(c_double), POINTER(c_double), POINTER(c_int), POINTER(c_int)])
            
            libcosy.cosy_set_cd_coeffs_(byref(c_int(self.da.idx)), c_re, c_im, c_exps, byref(c_int(len(coeffs))))
        else:
            flat_coeffs = coeffs.astype(np.float64)
            c_coeffs = (c_double * len(flat_coeffs))(*flat_coeffs)
            libcosy.cosy_set_coeffs_(byref(c_int(self.da.idx)), c_coeffs, c_exps, byref(c_int(len(coeffs))))

    def get_constant(self):
        return self.da.get_constant()

    def to_dict(self):
        terms = self.da.get_all_terms()
        data = {}
        if not terms:
            data["exponents"] = np.empty((0, self.dimension), dtype=int)
            data["coeffs"] = np.array([])
            return data
        
        # terms contain exponents of length physical_dimension
        all_exponents = np.array([t[0] for t in terms])
        coeffs = np.array([t[1] for t in terms])
        
        # Truncate exponents to logical dimension
        if self.dimension < self.physical_dimension:
             data["exponents"] = all_exponents[:, :self.dimension]
        else:
             data["exponents"] = all_exponents
             
        data["coeffs"] = coeffs
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
