import ctypes
import os
import sys
import numpy as np
from typing import Tuple

# Load the library
_lib_path = os.path.join(os.path.dirname(__file__), "libcosy.so")
try:
    _lib = ctypes.CDLL(_lib_path)
except OSError as e:
    raise ImportError(f"Could not load COSY library at {_lib_path}: {e}")

# Define types
c_int_p = ctypes.POINTER(ctypes.c_int)
c_double_p = ctypes.POINTER(ctypes.c_double)

# --- Fortran Name Mangling ---
# gfortran typically appends an underscore to function names.
# We define a helper to find the function.
def get_fortran_func(name):
    # Try name + '_' (standard gfortran)
    func = getattr(_lib, name + "_", None)
    if func:
        return func
    # Try name (e.g. if bind(c) was used, though not with legacy)
    func = getattr(_lib, name, None)
    if func:
        return func
    # Try name + '__'
    func = getattr(_lib, name + "__", None)
    if func:
        return func
    raise AttributeError(f"Could not find Fortran function {name}")

# --- Bindings ---

# SUBROUTINE SETUP_COSY(KORDER, KVAR, KMAXMEM)
_setup_cosy = get_fortran_func("setup_cosy")
_setup_cosy.argtypes = [c_int_p, c_int_p, c_int_p]
_setup_cosy.restype = None

# SUBROUTINE GET_VAR_IDX(IDX)
_get_var_idx = get_fortran_func("get_var_idx")
_get_var_idx.argtypes = [c_int_p]
_get_var_idx.restype = None

# SUBROUTINE CREATE_DA_VAR(KIVAR, RES_IDX)
_create_da_var = get_fortran_func("create_da_var")
_create_da_var.argtypes = [c_int_p, c_int_p]
_create_da_var.restype = None

# SUBROUTINE CREATE_DA_CONST(VAL, RES_IDX)
_create_da_const = get_fortran_func("create_da_const")
_create_da_const.argtypes = [c_double_p, c_int_p]
_create_da_const.restype = None

# EVAL_DA(DA_IDX, POINT, RES)
_eval_da = get_fortran_func("eval_da")
_eval_da.argtypes = [c_int_p, c_double_p, c_double_p]
_eval_da.restype = None

# COMPUTE wrappers
def bind_compute(name):
    func = get_fortran_func(name)
    func.argtypes = [c_int_p, c_int_p]
    func.restype = None
    return func

_compute_da_exp = bind_compute("compute_da_exp")
_compute_da_sin = bind_compute("compute_da_sin")
_compute_da_cos = bind_compute("compute_da_cos")
_compute_da_tan = bind_compute("compute_da_tan")
_compute_da_sinh = bind_compute("compute_da_sinh")
_compute_da_cosh = bind_compute("compute_da_cosh")
_compute_da_tanh = bind_compute("compute_da_tanh")
_compute_da_asin = bind_compute("compute_da_asin")
_compute_da_acos = bind_compute("compute_da_acos")
_compute_da_atan = bind_compute("compute_da_atan")
_compute_da_sqrt = bind_compute("compute_da_sqrt")
_compute_da_log = bind_compute("compute_da_log")

# Arithmetic (Div is improved? Add/Mul still direct DAADA/DAMDA?)
# Provided wrapper has COMPUTE_DA_DIV
_compute_da_div = get_fortran_func("compute_da_div")
_compute_da_div.argtypes = [c_int_p, c_int_p, c_int_p]

# COSY_GET_COEFFS (still present in optimized wrapper)
_cosy_get_coeffs = get_fortran_func("cosy_get_coeffs")
_cosy_get_coeffs.argtypes = [c_int_p, c_double_p, c_int_p, c_int_p, c_int_p]
_cosy_get_coeffs.restype = None

# COSY_SET_COEFFS (still present)
_cosy_set_coeffs = get_fortran_func("cosy_set_coeffs")
_cosy_set_coeffs.argtypes = [c_int_p, c_double_p, c_int_p, c_int_p]
_cosy_set_coeffs.restype = None

# Standard Arithmetic (still needed for Add/Mul as they weren't wrapped with COMPUTE_ prefix except DIV)
_daada = get_fortran_func("daada")
_daada.argtypes = [c_int_p, c_int_p, c_int_p]
_dasda = get_fortran_func("dasda")
_dasda.argtypes = [c_int_p, c_int_p, c_int_p]
_damda = get_fortran_func("damda")
_damda.argtypes = [c_int_p, c_int_p, c_int_p]
_dacop = get_fortran_func("dacop")
_dacop.argtypes = [c_int_p, c_int_p]


# --- Python Interface ---

def get_constants():
    lvar = ctypes.c_int()
    lmem = ctypes.c_int()
    ldim = ctypes.c_int()
    lno = ctypes.c_int()
    lnv = ctypes.c_int()
    lea = ctypes.c_int()
    _cosy_get_constants(lvar, lmem, ldim, lno, lnv, lea)
    return {
        "LVAR": lvar.value,
        "LMEM": lmem.value,
        "LDIM": ldim.value,
        "LNO": lno.value,
        "LNV": lnv.value,
        "LEA": lea.value,
    }

def init(order: int, nvars: int):
    korder = ctypes.c_int(order)
    knvars = ctypes.c_int(nvars)
    kmaxmem = ctypes.c_int()
    _setup_cosy(korder, knvars, kmaxmem)
    return kmaxmem.value

def alloc(n: int = 1) -> ctypes.Array:
    # Supporting only 1 for now as per optimized wrapper or loop
    indices = (ctypes.c_int * n)()
    for i in range(n):
        idx = ctypes.c_int()
        _get_var_idx(idx)
        indices[i] = idx.value
    return indices

def free(indices: ctypes.Array):
    # Optimized wrapper relies on COSY internal management?
    # Or we leak if we don't assume a free function.
    # The new wrapper DOES NOT expose a free function for generic variables directly?
    # Actually, it has no `free` equivalent exposed in my read. 
    # It seems to rely on COSY's stack or expected us not to free manually or I missed it.
    # Re-checking wrapper.f: It calls FOXALL but no dedicated free except internally.
    # So `free` is a no-op or dangerous. Let's make it no-op for now.
    pass

def da_add(ina, inb, inc):
    _daada(ctypes.c_int(ina), ctypes.c_int(inb), ctypes.c_int(inc))

def da_sub(ina, inb, inc):
    _dasda(ctypes.c_int(ina), ctypes.c_int(inb), ctypes.c_int(inc))

def da_mul(ina, inb, inc):
    _damda(ctypes.c_int(ina), ctypes.c_int(inb), ctypes.c_int(inc))

def da_div(ina, inb, inc):
    _compute_da_div(ctypes.c_int(ina), ctypes.c_int(inb), ctypes.c_int(inc))

def da_copy(ina, inc):
    _dacop(ctypes.c_int(ina), ctypes.c_int(inc))

def da_const(inc, val):
    # Use new dedicated routine
    _create_da_const(ctypes.c_double(val), ctypes.c_int(inc))

def da_var(inc, val, idx):
    # This was used to create var. New wrapper has CREATE_DA_VAR(IVAR, RES_IDX)
    # But that creates a fresh variable.
    # Here we are usually initializing an existing variable index 'inc'.
    # If standard DAVAR is not exposed, we might use CREATE_DA_VAR if we are allocating?
    # But `CosyMtfData` pattern is: alloc -> init.
    # If `da_var` assumes `inc` is already alloc'd, `CREATE_DA_VAR` allocates NEW.
    # Wait, `CREATE_DA_VAR` in new wrapper:
    # CALL FOXALL(IC, 1, NMMAX) -> RES_IDX
    # So it allocates. 
    # Current usages of `da_var` in `cosy_backend.py`?
    # It is NOT used in `cosy_backend.py` currently! 
    # `cosy_backend.py` uses `raw_set_coeffs` to init.
    # So `da_var` might be unused. I'll leave it or comment out.
    pass

def da_exp(ina, inc):
    _compute_da_exp(ctypes.c_int(ina), ctypes.c_int(inc))

def da_sin(ina, inc):
    _compute_da_sin(ctypes.c_int(ina), ctypes.c_int(inc))

def da_cos(ina, inc):
    _compute_da_cos(ctypes.c_int(ina), ctypes.c_int(inc))

# New eval function
def da_eval(ivar, point):
    # point must be array of doubles
    c_point = (ctypes.c_double * len(point))(*point)
    res = ctypes.c_double()
    _eval_da(ctypes.c_int(ivar), c_point, res)
    return res.value

def da_exp(ina, inc):
    _daexp(ctypes.c_int(ina), ctypes.c_int(inc))

def da_sin(ina, inc):
    _dasin(ctypes.c_int(ina), ctypes.c_int(inc))

def da_cos(ina, inc):
    _dacos(ctypes.c_int(ina), ctypes.c_int(inc))

def get_coeffs(ivar):
    # Retrieve coeffs and exponents
    # We guess a max size or retrieve it from constants
    # Using LEA from constants is safe?
    # For now, let's use a large buffer, e.g. 10000 terms
    max_terms = 10000 
    
    # We need to know NVMAX (current nvars) to decode exponents properly? 
    # The wrapper uses NVMAX from common block, so flattened exponents array size depends on it.
    # We should cache NVMAX somewhere or get it from get_constants if we trust it doesn't change 
    # (actually NVMAX changes with DAINI).
    # Since we don't expose NVMAX getter yet, let's assume the user knows the dimension they initialized.
    # Wait, the wrapper needs global NVMAX for decoding (decoding happens inside wrapper.f). 
    # The OUT exponents array is 1D: (NUM_TERMS * NVMAX).
    # We need to pass the buffer size.
    
    # Let's dynamically resize if needed? No, ctypes straightforwardness.
    # We'll stick to a reasonable max for now or fetch property.
    
    # Let's improve this: Add a COSY_GET_NVMAX to wrapper?
    # Or just rely on what we set in `init`.
    pass 
    # Implemented in CosyMtfData instead using these primitives.

def raw_get_coeffs(ivar, max_terms, nvmax):
    coeffs = (ctypes.c_double * max_terms)()
    exponents = (ctypes.c_int * (max_terms * nvmax))()
    num_terms = ctypes.c_int()
    
    _cosy_get_coeffs(ctypes.c_int(ivar), coeffs, exponents, ctypes.c_int(max_terms), num_terms)
    
    real_num = num_terms.value
    # Slicing ctypes array returns python list, convenient
    return coeffs[:real_num], exponents[:real_num * nvmax]

def raw_set_coeffs(ivar, coeffs, exponents, nvmax):
    num_terms = len(coeffs)
    c_coeffs = (ctypes.c_double * num_terms)(*coeffs)
    c_exponents = (ctypes.c_int * (num_terms * nvmax))(*exponents)
    
    _cosy_set_coeffs(ctypes.c_int(ivar), c_coeffs, c_exponents, ctypes.c_int(num_terms))

