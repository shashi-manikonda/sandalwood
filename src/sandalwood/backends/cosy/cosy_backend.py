import os
import sys
from ctypes import CDLL, POINTER, byref, c_double, c_int

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

# Mixed-Mode Arithmetic
bind_cosy_func("compute_da_add_const_", [POINTER(c_int), POINTER(c_double), POINTER(c_int)])
bind_cosy_func("compute_da_sub_const_", [POINTER(c_int), POINTER(c_double), POINTER(c_int)])
bind_cosy_func("compute_da_sub_r_const_", [POINTER(c_int), POINTER(c_double), POINTER(c_int)])
bind_cosy_func("compute_da_mul_const_", [POINTER(c_int), POINTER(c_double), POINTER(c_int)])
bind_cosy_func("compute_da_div_const_", [POINTER(c_int), POINTER(c_double), POINTER(c_int)])
bind_cosy_func("compute_da_div_r_const_", [POINTER(c_int), POINTER(c_double), POINTER(c_int)])

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
bind_cosy_func("compute_da_cot_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_asin_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_acos_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_atan_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_daest_", [POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_double)])
bind_cosy_func("compute_da_daest_", [POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_double)])
bind_cosy_func("compute_da_coth_", [POINTER(c_int), POINTER(c_int)])

# Batch Operations
bind_cosy_func("compute_da_add_batch_", [POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_sub_batch_", [POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_mul_batch_", [POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_div_batch_", [POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_minv_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_isrt3_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_norm_", [POINTER(c_int), POINTER(c_double)])
bind_cosy_func("compute_da_wnorm_", [POINTER(c_int), POINTER(c_int), POINTER(c_double)])

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
bind_cosy_func("compute_cd_tan_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_sinh_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_cosh_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_tanh_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_sqrt_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_der_", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_pei_", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_pkp_", [POINTER(c_int), POINTER(c_double), POINTER(c_int)])
bind_cosy_func("compute_cd_mui_", [POINTER(c_int), POINTER(c_int)])

# ... (Previous bindings)
bind_cosy_func("get_mem_state_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("set_mem_state_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("create_nda_var_", [POINTER(c_int), POINTER(c_double), POINTER(c_int)])
bind_cosy_func("get_all_coeffs_flat_", [POINTER(c_int), POINTER(c_double), POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("eval_da_batch_", [POINTER(c_int), POINTER(c_double), POINTER(c_int), POINTER(c_double), POINTER(c_int), POINTER(c_double), POINTER(c_int)])

# --- Math Framework Bindings ---
bind_cosy_func("da_deriv_safe_", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("da_integ_", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("da_poisson_", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("da_lin_comb_", [POINTER(c_int), POINTER(c_double), POINTER(c_int), POINTER(c_double), POINTER(c_int)])
bind_cosy_func("da_mat_inv_", [POINTER(c_double), POINTER(c_double), POINTER(c_int)])
bind_cosy_func("da_mat_inv_", [POINTER(c_double), POINTER(c_double), POINTER(c_int)])
bind_cosy_func("compute_da_polval_", [POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_mui_", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_pep_", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_pkp_", [POINTER(c_int), POINTER(c_double), POINTER(c_int)])

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
    is_complex = False
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
            # Use safe NDA creation to avoid NST issues
            libcosy.create_nda_var_(byref(res_idx), byref(c_double(0.0)), byref(c_int(var_id + 1)))
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

    def _should_promote(self, other):
        return self.is_complex or isinstance(other, complex) or (isinstance(other, CosyDA) and other.is_complex)

    def __add__(self, other):
        if self._should_promote(other):
             return self.to_complex() + other
        res_idx = c_int(0)
        if isinstance(other, CosyDA):
            libcosy.compute_da_add_(byref(c_int(self.idx)), byref(c_int(other.idx)), byref(res_idx))
        elif isinstance(other, (int, float, np.number)):
             libcosy.compute_da_add_const_(byref(c_int(self.idx)), byref(c_double(float(other))), byref(res_idx))
        else:
            con = CosyDA.from_const(other)
            libcosy.compute_da_add_(byref(c_int(self.idx)), byref(c_int(con.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if self._should_promote(other):
             return self.to_complex() - other
        res_idx = c_int(0)
        if isinstance(other, CosyDA):
            libcosy.compute_da_sub_(byref(c_int(self.idx)), byref(c_int(other.idx)), byref(res_idx))
        elif isinstance(other, (int, float, np.number)):
             libcosy.compute_da_sub_const_(byref(c_int(self.idx)), byref(c_double(float(other))), byref(res_idx))
        else:
            con = CosyDA.from_const(other)
            libcosy.compute_da_sub_(byref(c_int(self.idx)), byref(c_int(con.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def __rsub__(self, other):
        if self._should_promote(other):
             return CosyCDA.from_const(other) - self
        res_idx = c_int(0)
        if isinstance(other, (int, float, np.number)):
             libcosy.compute_da_sub_r_const_(byref(c_int(self.idx)), byref(c_double(float(other))), byref(res_idx))
        else:
            con = CosyDA.from_const(other)
            libcosy.compute_da_sub_(byref(c_int(con.idx)), byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def __neg__(self):
        if self.is_complex:
             return self.to_complex() * -1.0
        res_idx = c_int(0)
        con = CosyDA.from_const(0.0)
        libcosy.compute_da_sub_(byref(c_int(con.idx)), byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def to_complex(self):
        res_idx = c_int(0)
        libcosy.compute_da_to_cd_(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def __mul__(self, other):
        if self._should_promote(other):
             return self.to_complex() * other
        res_idx = c_int(0)
        if isinstance(other, CosyDA):
            libcosy.compute_da_mul_(byref(c_int(self.idx)), byref(c_int(other.idx)), byref(res_idx))
        elif isinstance(other, (int, float, np.number)):
            # Optimized
            libcosy.compute_da_mul_const_(byref(c_int(self.idx)), byref(c_double(float(other))), byref(res_idx))
        else:
            return NotImplemented
        return CosyDA(idx=res_idx.value, owned=True)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if self._should_promote(other):
             return self.to_complex() / other
        res_idx = c_int(0)
        if isinstance(other, CosyDA):
            libcosy.compute_da_div_(byref(c_int(self.idx)), byref(c_int(other.idx)), byref(res_idx))
        elif isinstance(other, (int, float, np.number)):
            libcosy.compute_da_div_const_(byref(c_int(self.idx)), byref(c_double(float(other))), byref(res_idx))
        else:
            con = CosyDA.from_const(other)
            libcosy.compute_da_div_(byref(c_int(self.idx)), byref(c_int(con.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def __rtruediv__(self, other):
        if self._should_promote(other):
             return CosyCDA.from_const(other) / self
        res_idx = c_int(0)
        if isinstance(other, (int, float, np.number)):
             libcosy.compute_da_div_r_const_(byref(c_int(self.idx)), byref(c_double(float(other))), byref(res_idx))
        else:
            con = CosyDA.from_const(other)
            libcosy.compute_da_div_(byref(c_int(con.idx)), byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def deriv(self, var_id):
        res_idx = c_int(0)
        c_var = c_int(var_id + 1)
        c_idx = c_int(self.idx)
        print(f"DEBUG PYTHON deriv (pre): var_id={var_id}, idx={self.idx}, res_idx={res_idx.value}")
        libcosy.da_deriv_safe_(byref(c_var), byref(c_idx), byref(res_idx))
        print(f"DEBUG PYTHON deriv (post): res_idx={res_idx.value}")
        return CosyDA(idx=res_idx.value, owned=True)

    def integral(self, var_id):
        res_idx = c_int(0)
        c_var = c_int(var_id + 1)
        c_idx = c_int(self.idx)
        print(f"DEBUG PYTHON integral (pre): var_id={var_id}, idx={self.idx}, res_idx={res_idx.value}")
        libcosy.da_integ_(byref(c_var), byref(c_idx), byref(res_idx))
        print(f"DEBUG PYTHON integral (post): res_idx={res_idx.value}")
        return CosyDA(idx=res_idx.value, owned=True)

    def poisson_bracket(self, other):
        res_idx = c_int(0)
        # Note: DA_POISSON wrapper might NOT allocate INC. It calls DAPOI(INA, INB, INC, SCRATCH).
        # We need to verify if DA_POISSON fixed.
        # Assuming we need to allocate for POISSON if wrapper doesn't.
        # But for now, let's keep original alloc for Poisson or fix wrapper?
        # Let's fix wrapper later if test fails. Reverting to create_da_const for Poisson just in case?
        # Or better: check wrapper.
        c_zero_val = c_double(0.0)
        c_zero_int = c_int(0)
        libcosy.create_da_var_(byref(res_idx), byref(c_zero_val), byref(c_zero_int))
        
        if isinstance(other, CosyDA):
            c_idx_self = c_int(self.idx)
            c_idx_other = c_int(other.idx)
            libcosy.da_poisson_(byref(c_idx_self), byref(c_idx_other), byref(res_idx))
        else:
            libcosy.create_da_const_(byref(res_idx), byref(c_double(0.0)))
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

    def inv_sqrt(self):
        res_idx = c_int(0)
        libcosy.compute_da_isrt_(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def inv_cbrt(self):
        res_idx = c_int(0)
        libcosy.compute_da_isrt3_(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def coth(self):
        res_idx = c_int(0)
        libcosy.compute_da_coth_(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def cot(self):
        res_idx = c_int(0)
        libcosy.compute_da_cot_(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def norm(self):
        val = c_double(0.0)
        libcosy.compute_da_norm_(byref(c_int(self.idx)), byref(val))
        return val.value

    def weighted_norm(self, weight_da):
        val = c_double(0.0)
        libcosy.compute_da_wnorm_(byref(c_int(self.idx)), byref(c_int(weight_da.idx)), byref(val))
        return val.value

    def estimate_stability(self, var_id=0, order=None):
        """
        Estimate stability/order decay.
        var_id: variable index (1-based for COSY, 0 means all).
        order: order to estimate at (defaults to global max_order).
        """
        if order is None:
            order = CosyBackend._order
        val = c_double(0.0)
        libcosy.compute_da_daest_(byref(c_int(self.idx)), byref(c_int(var_id)), byref(c_int(order)), byref(val))
        return val.value

    def compose_polval(self, args_da_list):
        """
        Compose this DA with a list of argument DAs using POLVAL.
        args_da_list: list of CosyDA objects (length must match N_ARGS expected by POLVAL, usually dimension).
        """
        n_args = len(args_da_list)
        # Verify dimension? self.dimension unavailable on CosyDA directly usually, but caller context has it.
        # Create array of indices
        args_indices = (c_int * n_args)()
        for i, arg in enumerate(args_da_list):
            args_indices[i] = arg.idx
        
        res_idx = c_int(0)
        libcosy.compute_da_polval_(byref(res_idx), byref(c_int(self.idx)), args_indices, byref(c_int(n_args)))
        return CosyDA(idx=res_idx.value, owned=True)

    def inverse(self):
        res_idx = c_int(0)
        libcosy.compute_da_mui_(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def __pow__(self, exponent):
        res_idx = c_int(0)
        if isinstance(exponent, int):
            libcosy.compute_da_pep_(byref(c_int(self.idx)), byref(c_int(exponent)), byref(res_idx))
        elif isinstance(exponent, float):
            libcosy.compute_da_pkp_(byref(c_int(self.idx)), byref(c_double(exponent)), byref(res_idx))
        else:
             return NotImplemented
        return CosyDA(idx=res_idx.value, owned=True)


class CosyCDA(CosyDA):
    is_complex = True
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

    def _ensure_cd(self, other):
        if isinstance(other, CosyCDA):
            return other
        if isinstance(other, (CosyDA, int, float, complex, np.number)):
            return CosyCDA.from_const(other)
        return NotImplemented


    def __truediv__(self, other):
        b = self._ensure_cd(other)
        if b is NotImplemented: return NotImplemented
        res_idx = c_int(0)
        libcosy.compute_cd_div_(byref(c_int(self.idx)), byref(c_int(b.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def __rtruediv__(self, other):
        a = self._ensure_cd(other)
        if a is NotImplemented: return NotImplemented
        return a / self

    def inverse(self):
        res_idx = c_int(0)
        libcosy.compute_cd_mui_(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def __pow__(self, exponent):
        res_idx = c_int(0)
        if isinstance(exponent, int):
             libcosy.compute_cd_pei_(byref(c_int(self.idx)), byref(c_int(exponent)), byref(res_idx))
        elif isinstance(exponent, float):
             libcosy.compute_cd_pkp_(byref(c_int(self.idx)), byref(c_double(exponent)), byref(res_idx))
        else:
             return NotImplemented
        return CosyCDA(idx=res_idx.value, owned=True)

    def exp(self):
        res_idx = c_int(0)
        libcosy.compute_cd_exp_(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def log(self):
        res_idx = c_int(0)
        libcosy.compute_cd_log_(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def sin(self):
        res_idx = c_int(0)
        libcosy.compute_cd_sin_(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def cos(self):
        res_idx = c_int(0)
        libcosy.compute_cd_cos_(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def tan(self):
        res_idx = c_int(0)
        libcosy.compute_cd_tan_(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def sinh(self):
        res_idx = c_int(0)
        libcosy.compute_cd_sinh_(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def cosh(self):
        res_idx = c_int(0)
        libcosy.compute_cd_cosh_(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def tanh(self):
        res_idx = c_int(0)
        libcosy.compute_cd_tanh_(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def sqrt(self):
        res_idx = c_int(0)
        libcosy.compute_cd_sqrt_(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)


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

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        res_idx = c_int(0)
        if isinstance(other, CosyCDA):
            libcosy.compute_cd_sub_(byref(c_int(self.idx)), byref(c_int(other.idx)), byref(res_idx))
        elif isinstance(other, CosyDA):
            other_cd = other.to_complex()
            libcosy.compute_cd_sub_(byref(c_int(self.idx)), byref(c_int(other_cd.idx)), byref(res_idx))
        else:
            re_v = float(other.real) if hasattr(other, 'real') else float(other)
            im_v = float(other.imag) if hasattr(other, 'imag') else 0.0
            con = CosyCDA(from_const=(re_v, im_v))
            libcosy.compute_cd_sub_(byref(c_int(self.idx)), byref(c_int(con.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def __rsub__(self, other):
        res_idx = c_int(0)
        re_v = float(other.real) if hasattr(other, 'real') else float(other)
        im_v = float(other.imag) if hasattr(other, 'imag') else 0.0
        con = CosyCDA(from_const=(re_v, im_v))
        libcosy.compute_cd_sub_(byref(c_int(con.idx)), byref(c_int(self.idx)), byref(res_idx))
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

    def __rmul__(self, other):
        return self.__mul__(other)

    def deriv(self, var_id):
        res_idx = c_int(0)
        c_var = c_int(var_id + 1)
        libcosy.compute_cd_der_(byref(c_var), byref(c_int(self.idx)), byref(res_idx))
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
    def __init__(self, dimension: int = None, is_complex: bool = False, idx=None, owned=False):
        CosyBackendManager.check_init()
        if idx is not None:
            if is_complex:
                self.da = CosyCDA(idx=idx, owned=owned)
            else:
                self.da = CosyDA(idx=idx, owned=owned)
        elif is_complex:
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

    def inverse(self):
        c0 = self.get_constant()
        if abs(c0) == 0:
             raise ValueError("Inversion of zero constant (Division by zero).")
        res_da = self.da.inverse()
        is_complex = isinstance(res_da, CosyCDA)
        return CosyMtfData(self.dimension, is_complex=is_complex, idx=res_da.idx, owned=True)

    def __pow__(self, other):
        if isinstance(other, (int, float)):
             res_da = self.da ** other
             is_complex = isinstance(res_da, CosyCDA)
             return CosyMtfData(self.dimension, is_complex=is_complex, idx=res_da.idx, owned=True)
        return NotImplemented

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

    def eval(self, points):
        points = np.asarray(points, dtype=np.float64)
    def eval(self, point):
        """
        Evaluate the DA vector at the given point(s).
        
        Args:
            point: 1D array (single point) or 2D array (batch of points).
                   Dimensions must match the number of variables (CosyBackend._dim).
        
        Returns:
            float (if single point) or np.ndarray (if batch).
        """
        if not CosyBackend.is_initialized():
            raise RuntimeError("COSY backend not initialized")

        point = np.asarray(point, dtype=np.float64)
        
        # Handle Batch Evaluation (2D input)
        if point.ndim == 2:
            n_points, dim = point.shape
            # If input dim < NVMAX, pad with zeros
            if dim < CosyBackend._dim:
                padded = np.zeros((n_points, CosyBackend._dim), dtype=np.float64)
                padded[:, :dim] = point
                point_flat = padded.flatten()
            else:
                point_flat = point.flatten() # Assumes dim == NVMAX

            vals = np.zeros(n_points, dtype=np.float64)
            max_terms = 100000 
            c_max_terms = c_int(max_terms)
            
            # Allocating workspace arrays
            # TEMP_EXPS: (MAX_TERMS, 40) integers
            # TEMP_COEFFS: (MAX_TERMS) doubles
            nvmax = 40 # Standard COSY limit
            temp_exps = np.zeros(max_terms * nvmax, dtype=np.int32)
            temp_coeffs = np.zeros(max_terms, dtype=np.float64)
            
            c_temp_exps = (c_int * len(temp_exps)).from_buffer(temp_exps)
            c_temp_coeffs = (c_double * len(temp_coeffs)).from_buffer(temp_coeffs)
            
            c_points = (c_double * len(point_flat)).from_buffer(point_flat)
            c_vals = (c_double * len(vals)).from_buffer(vals)
            c_n_points = c_int(n_points)
            
            libcosy.eval_da_batch_(byref(c_int(self.da.idx)), 
                                   c_points, 
                                   byref(c_n_points), 
                                   c_vals,
                                   c_temp_exps,
                                   c_temp_coeffs,
                                   byref(c_max_terms))
                                   
            return vals
        # Handle Single Point Evaluation (1D input)
        elif point.ndim == 1:
            points_reshaped = point.reshape(1, -1)
            # Recursively call with 2D array and extract result
            return self.eval(points_reshaped)[0]
        else:
            raise ValueError(f"Invalid input shape {point.shape}")

    def add(self, other):
        return self._create_res(self.da + other.da)

    def subtract(self, other):
        return self._create_res(self.da - other.da)

    def multiply(self, other):
        return self._create_res(self.da * other.da)

    def multiply_inplace(self, other):
        self.da = self.da * other.da

    def divide(self, other):
        c0 = other.get_constant()
        if abs(c0) == 0:
             raise ValueError("Division by zero (constant part is zero).")
        return self._create_res(self.da / other.da)

    def negate(self):
        return self._create_res(-self.da)

    def partial_derivative(self, deriv_dim):
        res = CosyMtfData(self.dimension)
        res.da = self.da.deriv(deriv_dim - 1)
        return res

    def integrate(self, var_idx):
        res = CosyMtfData(self.dimension)
        res.da = self.da.integral(var_idx - 1)
        return res

    def poisson_bracket(self, other):
        res = CosyMtfData(self.dimension)
        res.da = self.da.poisson_bracket(other.da)
        return res

    def _create_res(self, res_da):
        is_complex = isinstance(res_da, (CosyCDA, complex)) # complex for scalar cases if any
        if not is_complex and hasattr(res_da, 'is_complex'):
             is_complex = res_da.is_complex
        # In case it's already a CosyCDA
        if isinstance(res_da, CosyCDA):
             is_complex = True
        
        # Determine idx
        idx = res_da.idx if hasattr(res_da, 'idx') else None
        
        return CosyMtfData(self.dimension, is_complex=is_complex, idx=idx, owned=True)

    def sin(self):
        return self._create_res(self.da.sin())

    def cos(self):
        return self._create_res(self.da.cos())

    def tan(self):
        return self._create_res(self.da.tan())

    def cot(self):
        return self._create_res(self.da.cot())

    def exp(self):
        return self._create_res(self.da.exp())

    def log(self):
        c0 = self.get_constant()
        if isinstance(c0, complex) or isinstance(self.da, CosyCDA):
            if abs(c0) == 0:
                raise ValueError("Logarithm undefined for zero constant part.")
        else:
            if c0 <= 0:
                raise ValueError("Logarithm undefined for non-positive constant part.")
        return self._create_res(self.da.log())

    def sqrt(self):
        c0 = self.get_constant()
        if not (isinstance(c0, complex) or isinstance(self.da, CosyCDA)):
            if c0 < 0:
                raise ValueError("Square root undefined for negative constant part (Real domain).")
        return self._create_res(self.da.sqrt())

    def asin(self):
        return self._create_res(self.da.asin() if hasattr(self.da, 'asin') else self.da.arcsin())

    def acos(self):
        return self._create_res(self.da.acos() if hasattr(self.da, 'acos') else self.da.arccos())

    def atan(self):
        return self._create_res(self.da.atan() if hasattr(self.da, 'atan') else self.da.arctan())

    def sinh(self):
        return self._create_res(self.da.sinh())

    def cosh(self):
        return self._create_res(self.da.cosh())

    def tanh(self):
        return self._create_res(self.da.tanh())

    def coth(self):
        return self._create_res(self.da.coth())

    def inv_sqrt(self):
        return self._create_res(self.da.inv_sqrt())

    def inv_cbrt(self):
        return self._create_res(self.da.inv_cbrt())

    def norm(self):
        return self.da.norm()

    def weighted_norm(self, weight):
        if hasattr(weight, "mtf_data"):
             weight = weight.mtf_data.da
        elif hasattr(weight, "da"):
             weight = weight.da
        return self.da.weighted_norm(weight)

    def estimate_stability(self, var_id=0, order=None):
        return self.da.estimate_stability(var_id, order)

def da_mat_inv(matrix_flat, n):
    """
    Invert a scalar matrix using COSY's MATINV.
    Args:
        matrix_flat: Flat list/array of n*n doubles (row-major).
        n: Dimension (max 50).
    Returns:
        Flat list of n*n doubles (inverse).
    """
    if n > 50:
        raise ValueError("COSY MATINV supports max dimension 50")
    
    mat_arr = (c_double * (n * n))(*matrix_flat)
    inv_arr = (c_double * (n * n))()
    
    libcosy.da_mat_inv_(mat_arr, inv_arr, byref(c_int(n)))
    
    return [inv_arr[i] for i in range(n * n)]

def da_lin_comb(da1, c1, da2, c2):
    """
    Compute Linear Combination: res = c1*da1 + c2*da2
    """
    res_idx = c_int(0)
    libcosy.create_da_var_(byref(res_idx), byref(c_double(0.0)), byref(c_int(0)))
    
    libcosy.da_lin_comb_(
        byref(c_int(da1.idx)), byref(c_double(c1)),
        byref(c_int(da2.idx)), byref(c_double(c2)),
        byref(res_idx)
    )
    return CosyDA(idx=res_idx.value, owned=True)

# -----------------------------------------------------------------------------
# Batch Operations Helpers
# -----------------------------------------------------------------------------

def _prepare_batch_args(idx_arr_a, idx_arr_b):
    """Helper to ensure contiguous int32 arrays for batch ops."""
    n = len(idx_arr_a)
    if len(idx_arr_b) != n:
        raise ValueError("Batch operation dimension mismatch")
        
    a_ptr = np.ascontiguousarray(idx_arr_a, dtype=np.int32)
    b_ptr = np.ascontiguousarray(idx_arr_b, dtype=np.int32)
    res_ptr = np.zeros(n, dtype=np.int32)
    
    return n, a_ptr, b_ptr, res_ptr

def batch_add(idx_arr_a, idx_arr_b):
    n, a_ptr, b_ptr, res_ptr = _prepare_batch_args(idx_arr_a, idx_arr_b)
    libcosy.compute_da_add_batch_(
        byref(c_int(n)),
        a_ptr.ctypes.data_as(POINTER(c_int)),
        b_ptr.ctypes.data_as(POINTER(c_int)),
        res_ptr.ctypes.data_as(POINTER(c_int))
    )
    return res_ptr

def batch_sub(idx_arr_a, idx_arr_b):
    n, a_ptr, b_ptr, res_ptr = _prepare_batch_args(idx_arr_a, idx_arr_b)
    libcosy.compute_da_sub_batch_(
        byref(c_int(n)),
        a_ptr.ctypes.data_as(POINTER(c_int)),
        b_ptr.ctypes.data_as(POINTER(c_int)),
        res_ptr.ctypes.data_as(POINTER(c_int))
    )
    return res_ptr

def batch_mul(idx_arr_a, idx_arr_b):
    n, a_ptr, b_ptr, res_ptr = _prepare_batch_args(idx_arr_a, idx_arr_b)
    libcosy.compute_da_mul_batch_(
        byref(c_int(n)),
        a_ptr.ctypes.data_as(POINTER(c_int)),
        b_ptr.ctypes.data_as(POINTER(c_int)),
        res_ptr.ctypes.data_as(POINTER(c_int))
    )
    return res_ptr

def batch_div(idx_arr_a, idx_arr_b):
    n, a_ptr, b_ptr, res_ptr = _prepare_batch_args(idx_arr_a, idx_arr_b)
    libcosy.compute_da_div_batch_(
        byref(c_int(n)),
        a_ptr.ctypes.data_as(POINTER(c_int)),
        b_ptr.ctypes.data_as(POINTER(c_int)),
        res_ptr.ctypes.data_as(POINTER(c_int))
    )
    return res_ptr
