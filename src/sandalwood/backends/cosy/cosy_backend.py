import logging
import os
import sys
import tempfile
import threading
from ctypes import CDLL, POINTER, byref, c_double, c_int
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# Path to the shared library (platform-aware)
if sys.platform == "win32":
    LIB_NAME = "libcosy.dll"
    # Python 3.8+ on Windows ignores PATH for DLL loading.
    # We must explicitly add Intel/oneAPI compiler paths and GCC/gfortran runtime paths if they exist in PATH or standard locations.
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        # 1. Search PATH
        for p in os.environ.get("PATH", "").split(os.pathsep):
            if p and os.path.exists(p):
                p_lower = p.lower()
                if any(
                    k in p_lower
                    for k in ("oneapi", "intel", "mingw", "msys", "gfortran", "gcc")
                ):
                    try:
                        os.add_dll_directory(p)
                    except OSError:
                        pass

        # 2. Search Standard Locations (if typical path didn't work)
        std_paths = [
            r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\redist\intel64_win\compiler",
            r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin",
            r"C:\msys64\mingw64\bin",
            r"C:\msys64\ucrt64\bin",
        ]
        for p in std_paths:
            if os.path.exists(p):
                try:
                    os.add_dll_directory(p)
                except OSError:
                    pass
elif sys.platform == "darwin":
    LIB_NAME = "libcosy.dylib"
else:
    LIB_NAME = "libcosy.so"
    # Linux: Try to preload Intel libraries if they are missing from LD_LIBRARY_PATH
    # This fixes issues where setvars.sh wasn't sourced or RPATH is missing.
    try:
        # Check standard Intel OneAPI locations
        intel_search_paths = [
            "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin",
            "/opt/intel/oneapi/compiler/latest/lib",
            # Also try specific versions if 'latest' isn't there (heuristic)
            "/opt/intel/oneapi/compiler/2025.3/lib",
        ]

        # Dependencies to preload in correct dependency order
        # (libintlc -> libimf -> libsvml -> libifcoremt -> libifport)
        libs_to_load = [
            "libintlc.so.5",
            "libiomp5.so",
            "libimf.so",
            "libsvml.so",
            "libifcoremt.so.5",
            "libifport.so.5",
        ]

        found_path = None
        for p in intel_search_paths:
            if os.path.exists(p):
                # Check if libs exist here
                if all(
                    os.path.exists(os.path.join(p, lib)) for lib in libs_to_load[:1]
                ):
                    found_path = p
                    break

        if found_path:
            import ctypes

            for lib in libs_to_load:
                full_path = os.path.join(found_path, lib)
                if os.path.exists(full_path):
                    try:
                        ctypes.CDLL(full_path, mode=os.RTLD_GLOBAL)
                    except OSError:
                        pass  # Ignore if already loaded or incompatible
    except Exception:
        pass  # Fallback to standard loading


LIB_PATH = os.path.join(os.path.dirname(__file__), LIB_NAME)

_cosy_lib_lock = threading.RLock()

# Load Library
try:
    # Use RTLD_GLOBAL to ensure symbols are available for resolution if supported
    mode = getattr(os, "RTLD_GLOBAL", 0)
    raw_libcosy = CDLL(LIB_PATH, mode=mode)
except OSError as e:
    # Hide the raw OS error in debug mode instead of printing to stderr
    logger.debug(f"COSY library not loaded ({LIB_NAME}): {e}")

    class DummyLib:
        def __getattr__(self, name):
            def func(*args):
                raise RuntimeError(f"COSY library not found at {LIB_PATH}")

            return func

    libcosy = DummyLib()
    COSY_AVAILABLE = False
else:
    COSY_AVAILABLE = True

    class LockedFunc:
        def __init__(self, func, lock):
            self._func = func
            self._lock = lock

        def __call__(self, *args, **kwargs):
            with self._lock:
                return self._func(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._func, name)

        def __setattr__(self, name, value):
            if name in ("_func", "_lock"):
                super().__setattr__(name, value)
            else:
                setattr(self._func, name, value)

    class LockedLib:
        def __init__(self, lib, lock):
            self._lib = lib
            self._lock = lock
            self._funcs = {}

        def __getattr__(self, name):
            attr = getattr(self._lib, name)
            if callable(attr):
                if name not in self._funcs:
                    self._funcs[name] = LockedFunc(attr, self._lock)
                return self._funcs[name]
            return attr

    libcosy = LockedLib(raw_libcosy, _cosy_lib_lock)


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
            # Try uppercase (Windows/Intel convention)
            try:
                f = getattr(libcosy, name.upper())
                f.argtypes = argtypes
                f.restype = None
                return f
            except AttributeError:
                logger.warning(f"Warning: COSY function {name} not found.")
                return None


# Core
bind_cosy_func("setup_cosy", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("get_da_coeff", [POINTER(c_int), POINTER(c_int), POINTER(c_double)])
bind_cosy_func("create_da_var", [POINTER(c_int), POINTER(c_double), POINTER(c_int)])
bind_cosy_func("create_da_const", [POINTER(c_int), POINTER(c_double)])
bind_cosy_func(
    "get_da_coeff_by_index",
    [POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_double)],
)
bind_cosy_func("cosy_free", [POINTER(c_int)])
bind_cosy_func(
    "cosy_set_coeffs",
    [POINTER(c_int), POINTER(c_double), POINTER(c_int), POINTER(c_int)],
)
bind_cosy_func("eval_da", [POINTER(c_int), POINTER(c_double), POINTER(c_double)])
bind_cosy_func("da_reset", [POINTER(c_int)])
bind_cosy_func("da_reset_cd", [POINTER(c_int)])
bind_cosy_func(
    "compute_da_div_batch",
    [POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)],
)
bind_cosy_func("compute_cd_int", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_poi", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])

# Arithmetic
bind_cosy_func("compute_da_add", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_sub", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_mul", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_div", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])

# Differential Operators
bind_cosy_func("compute_da_der", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_int", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_poi", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])

# Mixed-Mode Arithmetic
bind_cosy_func(
    "compute_da_add_const", [POINTER(c_int), POINTER(c_double), POINTER(c_int)]
)
bind_cosy_func(
    "compute_da_sub_const", [POINTER(c_int), POINTER(c_double), POINTER(c_int)]
)
bind_cosy_func(
    "compute_da_sub_r_const", [POINTER(c_int), POINTER(c_double), POINTER(c_int)]
)
bind_cosy_func(
    "compute_da_mul_const", [POINTER(c_int), POINTER(c_double), POINTER(c_int)]
)
bind_cosy_func(
    "compute_da_div_const", [POINTER(c_int), POINTER(c_double), POINTER(c_int)]
)
bind_cosy_func(
    "compute_da_div_r_const", [POINTER(c_int), POINTER(c_double), POINTER(c_int)]
)

# Elementary Functions
bind_cosy_func("compute_da_sin", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_cos", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_exp", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_log", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_sinh", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_cosh", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_tanh", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_tan", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_sqr", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_sqrt", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_isrt", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_cot", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_asin", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_acos", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_atan", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func(
    "compute_da_daest",
    [POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_double)],
)
bind_cosy_func(
    "compute_da_daest",
    [POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_double)],
)
bind_cosy_func("compute_da_coth", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_erf", [POINTER(c_int), POINTER(c_int)])

# Batch Operations
bind_cosy_func(
    "compute_da_add_batch",
    [POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)],
)
bind_cosy_func(
    "compute_da_sub_batch",
    [POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)],
)
bind_cosy_func(
    "compute_da_mul_batch",
    [POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)],
)
bind_cosy_func(
    "compute_da_div_batch",
    [POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)],
)
bind_cosy_func("compute_da_minv", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_isrt3", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_norm", [POINTER(c_int), POINTER(c_double)])
bind_cosy_func("compute_da_wnorm", [POINTER(c_int), POINTER(c_int), POINTER(c_double)])

# Complex wrappers
bind_cosy_func("create_cda_var", [POINTER(c_int), POINTER(c_int), POINTER(c_double)])
bind_cosy_func(
    "create_cda_const", [POINTER(c_int), POINTER(c_double), POINTER(c_double)]
)
bind_cosy_func("compute_cd_add", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_sub", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_mul", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_div", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
# Complex Math
bind_cosy_func("compute_cd_exp", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_log", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_sin", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_cos", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_tan", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_sinh", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_cosh", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_tanh", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_sqrt", [POINTER(c_int), POINTER(c_int)])

# Other
bind_cosy_func("get_cda_re", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("get_cda_im", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("set_cd_parts", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_to_cd", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func(
    "get_cda_coeff_by_index",
    [
        POINTER(c_int),
        POINTER(c_int),
        POINTER(c_int),
        POINTER(c_double),
        POINTER(c_double),
    ],
)
bind_cosy_func("compute_cd_tan", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_sinh", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_cosh", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_tanh", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_sqrt", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_der", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_pei", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_pkp", [POINTER(c_int), POINTER(c_double), POINTER(c_int)])
bind_cosy_func("compute_cd_mui", [POINTER(c_int), POINTER(c_int)])

# ... (Previous bindings)
bind_cosy_func("get_mem_state", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("set_mem_state", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("create_nda_var", [POINTER(c_int), POINTER(c_double), POINTER(c_int)])
bind_cosy_func(
    "get_all_coeffs_flat",
    [POINTER(c_int), POINTER(c_double), POINTER(c_int), POINTER(c_int), POINTER(c_int)],
)
bind_cosy_func(
    "eval_da_batch",
    [
        POINTER(c_int),
        POINTER(c_double),
        POINTER(c_int),
        POINTER(c_double),
        POINTER(c_int),
        POINTER(c_double),
        POINTER(c_int),
    ],
)

bind_cosy_func(
    "compute_biot_savart_batch",
    [
        POINTER(c_int),  # NP
        POINTER(c_int),  # NE
        POINTER(c_int),  # POS_X
        POINTER(c_int),  # POS_Y
        POINTER(c_int),  # POS_Z
        POINTER(c_int),  # SRC_X
        POINTER(c_int),  # SRC_Y
        POINTER(c_int),  # SRC_Z
        POINTER(c_int),  # DL_X
        POINTER(c_int),  # DL_Y
        POINTER(c_int),  # DL_Z
        POINTER(c_int),  # B_X (Result)
        POINTER(c_int),  # B_Y (Result)
        POINTER(c_int),  # B_Z (Result)
    ],
)

bind_cosy_func(
    "compute_biot_savart_batch_fast",
    [
        c_int,  # NP (Value)
        c_int,  # NE (Value)
        POINTER(c_double),  # POS_X
        POINTER(c_double),  # POS_Y
        POINTER(c_double),  # POS_Z
        POINTER(c_double),  # SRC_X
        POINTER(c_double),  # SRC_Y
        POINTER(c_double),  # SRC_Z
        POINTER(c_double),  # DL_X
        POINTER(c_double),  # DL_Y
        POINTER(c_double),  # DL_Z
        POINTER(c_double),  # B_X (Result)
        POINTER(c_double),  # B_Y (Result)
        POINTER(c_double),  # B_Z (Result)
    ],
)

# --- Math Framework Bindings ---
bind_cosy_func("da_deriv_safe", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("da_integ", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("da_poisson", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_int", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_cd_poi", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])

bind_cosy_func(
    "da_lin_comb",
    [
        POINTER(c_int),
        POINTER(c_double),
        POINTER(c_int),
        POINTER(c_double),
        POINTER(c_int),
    ],
)
bind_cosy_func("da_mat_inv", [POINTER(c_double), POINTER(c_double), POINTER(c_int)])
bind_cosy_func("da_mat_inv", [POINTER(c_double), POINTER(c_double), POINTER(c_int)])
bind_cosy_func(
    "compute_da_polval",
    [POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)],
)
bind_cosy_func("compute_da_mui", [POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_pep", [POINTER(c_int), POINTER(c_int), POINTER(c_int)])
bind_cosy_func("compute_da_pkp", [POINTER(c_int), POINTER(c_double), POINTER(c_int)])

# Control flags
_USE_OMP = bool(os.environ.get("COSY_USE_OMP", "0"))


class CosyScope:
    _depth = 0

    def __init__(self):
        self.ivar = c_int(0)
        self.imem = c_int(0)

    def __enter__(self):
        CosyScope._depth += 1
        if not CosyBackend.is_initialized():
            raise RuntimeError("COSY backend not initialized")
        libcosy.get_mem_state(byref(self.ivar), byref(self.imem))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        libcosy.set_mem_state(byref(self.ivar), byref(self.imem))
        CosyScope._depth -= 1
        return False


class CosyIndexPool:
    _free_indices_da: List[int] = []
    _free_indices_cda: List[int] = []
    _epoch = 0
    _chunk_size = int(os.environ.get("SANDALWOOD_COSY_POOL_SIZE", "1024"))
    # Reentrant lock so that acquire/release are safe from multiple threads.
    # The COSY Fortran allocator is single-threaded; this lock prevents
    # concurrent pool mutations that could produce duplicate or orphaned indices.
    _lock = _cosy_lib_lock
    _thread_local = threading.local()

    @classmethod
    def _get_thread_local_pool(cls, is_complex=False):
        if not hasattr(cls._thread_local, "epoch"):
            cls._thread_local.epoch = cls._epoch
            cls._thread_local.free_da = []
            cls._thread_local.free_cda = []
        elif cls._thread_local.epoch != cls._epoch:
            cls._thread_local.epoch = cls._epoch
            cls._thread_local.free_da.clear()
            cls._thread_local.free_cda.clear()
        return cls._thread_local.free_cda if is_complex else cls._thread_local.free_da

    @classmethod
    def acquire(cls, is_complex=False):
        # Bypass pool if inside a CosyScope stack context
        if CosyScope._depth > 0:
            res_idx = c_int(0)
            if is_complex:
                libcosy.create_cda_const(
                    byref(res_idx), byref(c_double(0.0)), byref(c_double(0.0))
                )
            else:
                libcosy.create_da_const(byref(res_idx), byref(c_double(0.0)))
            return res_idx.value

        local_pool = cls._get_thread_local_pool(is_complex)

        if not local_pool:
            with cls._lock:
                global_pool = cls._free_indices_cda if is_complex else cls._free_indices_da
                if len(global_pool) < cls._chunk_size:
                    # Allocate a new chunk
                    for _ in range(cls._chunk_size):
                        res_idx = c_int(0)
                        if is_complex:
                            libcosy.create_cda_const(
                                byref(res_idx), byref(c_double(0.0)), byref(c_double(0.0))
                            )
                        else:
                            libcosy.create_da_const(byref(res_idx), byref(c_double(0.0)))
                        global_pool.append(res_idx.value)

                # Fetch a chunk from global_pool to local_pool
                num_to_fetch = min(cls._chunk_size, len(global_pool))
                for _ in range(num_to_fetch):
                    local_pool.append(global_pool.pop())

        idx = local_pool.pop()
        # Mathematically clean reset
        if is_complex:
            libcosy.da_reset_cd(byref(c_int(idx)))
        else:
            libcosy.da_reset(byref(c_int(idx)))
        return idx

    @classmethod
    def release(cls, idx, is_complex=False):
        if idx is None:
            return
        # Only pool if outside scope; inside scope, indices are invalid after rewind
        if CosyScope._depth == 0:
            local_pool = cls._get_thread_local_pool(is_complex)
            local_pool.append(idx)

            # If local pool size exceeds threshold, return a chunk to global pool
            if len(local_pool) >= 2 * cls._chunk_size:
                chunk_to_return = [local_pool.pop() for _ in range(cls._chunk_size)]
                with cls._lock:
                    global_pool = cls._free_indices_cda if is_complex else cls._free_indices_da
                    global_pool.extend(chunk_to_return)


# Module-level lock removed because os.chdir swapping is no longer needed since DAINI.DAT is bypassed.


class CosyBackend:
    _initialized = False
    _order = 1
    _dim = 1

    @staticmethod
    def is_initialized():
        return CosyBackend._initialized

    @staticmethod
    def initialize(order, dim):
        # Clear stale indices from the pool to prevent reuse across re-initializations
        CosyIndexPool._free_indices_da.clear()
        CosyIndexPool._free_indices_cda.clear()
        CosyIndexPool._epoch += 1

        CosyBackend._order = order
        CosyBackend._dim = dim
        c_order = c_int(order)
        c_dim = c_int(dim)
        c_nmmax = c_int(0)

        if not hasattr(libcosy, "setup_cosy"):
            raise RuntimeError(f"COSY library not found at {LIB_PATH}")

        libcosy.setup_cosy(byref(c_order), byref(c_dim), byref(c_nmmax))

        CosyBackend._initialized = True

    @staticmethod
    def reset():
        if CosyBackend._initialized:
            CosyBackend.initialize(CosyBackend._order, CosyBackend._dim)

    @staticmethod
    def var(var_index):
        return CosyDA(var_id=var_index)

    @staticmethod
    def biot_savart_batch_indices(
        pos_x, pos_y, pos_z, src_x, src_y, src_z, dl_x, dl_y, dl_z
    ):
        """
        Low-level batch calculation returning raw integer indices.
        Used for Parametric Mode (MTF/DA).
        """
        n_pts = len(pos_x)
        n_src = len(src_x)

        # Helper to get indices and keep temporaries alive
        def get_indices(da_list):
            indices = (c_int * len(da_list))()
            keep_alive = []
            for i, item in enumerate(da_list):
                if isinstance(item, CosyDA):
                    indices[i] = item.idx
                elif (
                    hasattr(item, "mtf_data")
                    and hasattr(item.mtf_data, "da")
                    and isinstance(item.mtf_data.da, CosyDA)
                ):
                    # Handle Sandalwood MultivariateTaylorFunction wrapper
                    indices[i] = item.mtf_data.da.idx
                else:
                    # Constants need to be created as DA
                    obj = CosyDA.from_const(item)
                    keep_alive.append(obj)
                    indices[i] = obj.idx
            return indices, keep_alive

        c_pos_x, k_pos_x = get_indices(pos_x)
        c_pos_y, k_pos_y = get_indices(pos_y)
        c_pos_z, k_pos_z = get_indices(pos_z)

        c_src_x, k_src_x = get_indices(src_x)
        c_src_y, k_src_y = get_indices(src_y)
        c_src_z, k_src_z = get_indices(src_z)

        c_dl_x, k_dl_x = get_indices(dl_x)
        c_dl_y, k_dl_y = get_indices(dl_y)
        c_dl_z, k_dl_z = get_indices(dl_z)

        c_b_x = (c_int * n_pts)()
        c_b_y = (c_int * n_pts)()
        c_b_z = (c_int * n_pts)()

        libcosy.compute_biot_savart_batch(
            byref(c_int(n_pts)),
            byref(c_int(n_src)),
            c_pos_x,
            c_pos_y,
            c_pos_z,
            c_src_x,
            c_src_y,
            c_src_z,
            c_dl_x,
            c_dl_y,
            c_dl_z,
            c_b_x,
            c_b_y,
            c_b_z,
        )

        # Return raw C-arrays of indices (caller must wrap them)
        return c_b_x, c_b_y, c_b_z

    @staticmethod
    def linear_combination(coeffs, da_list):
        """
        Computes sum(coeffs[i] * da_list[i]) efficiently using COSY kernel.
        """
        n = len(da_list)
        if n == 0:
            return CosyDA.from_const(0.0)

        # Prepare inputs
        c_n = c_int(n)
        c_coeffs = (c_double * n)(*coeffs)
        c_indices = (c_int * n)()

        # Handle both CosyDA and CosyMtfData/wrapper objects
        for i, item in enumerate(da_list):
            if hasattr(item, "idx"):
                c_indices[i] = item.idx
            elif hasattr(item, "da") and hasattr(item.da, "idx"):
                c_indices[i] = item.da.idx
            else:
                # Fallback for constant
                tmp = CosyDA.from_const(item)
                c_indices[i] = tmp.idx

        # Fallback to iterative addition/scaling because da_lin_comb seems unstable/broken
        # (getting "VARIABLE 2 HAS WRONG TYPE" errors).
        # We perform the loop using direct C calls for speed.

        # Allocate accumulator
        c_res_idx = c_int(CosyIndexPool.acquire())

        c_temp_idx = c_int(0)
        c_add_res = c_int(0)

        for i in range(n):
            idx = c_indices[i]
            coeff = c_coeffs[i]

            # If coeff is 0, skip
            if abs(coeff) < 1e-16:
                continue

            # Scale if needed
            if abs(coeff - 1.0) > 1e-16:
                # Multiply by scalar
                # Note: compute_da_mul_const allocates result in last arg?
                # compute_da_mul_const(idx_in, val, idx_out)
                c_scaled_idx = c_int(CosyIndexPool.acquire())
                libcosy.compute_da_mul_const(
                    byref(c_int(idx)), byref(c_double(coeff)), byref(c_scaled_idx)
                )
                term_idx = c_scaled_idx
            else:
                term_idx = c_int(idx)

            # Add to accumulator
            # compute_da_add(a, b, res) -> allocated new res usually
            # But we want to accumulate.
            # R = R + Term
            # result index changes at each step.

            c_next_res = c_int(CosyIndexPool.acquire())
            libcosy.compute_da_add(
                byref(c_res_idx),
                byref(
                    c_int(term_idx.value if hasattr(term_idx, "value") else term_idx)
                ),
                byref(c_next_res),
            )

            # Free old accumulator
            old_idx = c_res_idx.value
            CosyIndexPool.release(old_idx)

            c_res_idx = c_next_res

            # If we scaled, we created a temp, technically should free it.
            if hasattr(term_idx, "value") and term_idx.value != idx:
                CosyIndexPool.release(term_idx.value)

        return CosyDA(idx=c_res_idx.value, owned=True)

    @staticmethod
    def batch_arithmetic(op, list_a, list_b):
        """
        Performs element-wise batch arithmetic: res[i] = op(list_a[i], list_b[i]).
        op: 'add', 'sub', 'mul', 'div'
        list_a, list_b: lists or arrays of CosyDA/CosyMtfData
        """
        n = len(list_a)
        if len(list_b) != n:
            raise ValueError("Batch arithmetic lists must be same length")

        c_n = c_int(n)
        c_idx_a = (c_int * n)()
        c_idx_b = (c_int * n)()
        c_idx_res = (c_int * n)()

        def get_idx(item):
            if hasattr(item, "idx"):
                return item.idx
            if hasattr(item, "da"):
                return item.da.idx
            return CosyDA.from_const(item).idx

        for i in range(n):
            c_idx_a[i] = get_idx(list_a[i])
            c_idx_b[i] = get_idx(list_b[i])

        func_map = {
            "add": libcosy.compute_da_add_batch,
            "sub": libcosy.compute_da_sub_batch,
            "mul": libcosy.compute_da_mul_batch,
            "div": libcosy.compute_da_div_batch,
        }

        for i in range(n):
            c_idx_res[i] = CosyIndexPool.acquire()

        func_map[op](byref(c_n), c_idx_a, c_idx_b, c_idx_res)

        return [CosyDA(idx=c_idx_res[i], owned=True) for i in range(n)]

    @staticmethod
    def biot_savart_batch(pos_x, pos_y, pos_z, src_x, src_y, src_z, dl_x, dl_y, dl_z):
        """
        Batch Biot-Savart calculation with Hybrid Dispatch.
        - If inputs are floats: Uses Fast Path (scalars), returns numpy arrays of floats.
        - If inputs are DAs: Uses General Path, returns lists of CosyDA objects.
        """
        # Check input type availability
        is_float_src = len(src_x) > 0 and isinstance(
            src_x[0], (float, np.floating, int, np.integer)
        )

        if is_float_src:
            # FAST PATH: Floats
            n_pts = len(pos_x)
            n_src = len(src_x)

            # Helper: Cast to contiguous doubles
            def to_doubles(arr):
                return np.ascontiguousarray(arr, dtype=np.float64)

            c_pos_x = to_doubles(pos_x)
            c_pos_y = to_doubles(pos_y)
            c_pos_z = to_doubles(pos_z)
            c_src_x = to_doubles(src_x)
            c_src_y = to_doubles(src_y)
            c_src_z = to_doubles(src_z)
            c_dl_x = to_doubles(dl_x)
            c_dl_y = to_doubles(dl_y)
            c_dl_z = to_doubles(dl_z)

            # Outputs
            out_x = np.zeros(n_pts, dtype=np.float64)
            out_y = np.zeros(n_pts, dtype=np.float64)
            out_z = np.zeros(n_pts, dtype=np.float64)

            libcosy.compute_biot_savart_batch_fast(
                c_int(n_pts),
                c_int(n_src),
                c_pos_x.ctypes.data_as(POINTER(c_double)),
                c_pos_y.ctypes.data_as(POINTER(c_double)),
                c_pos_z.ctypes.data_as(POINTER(c_double)),
                c_src_x.ctypes.data_as(POINTER(c_double)),
                c_src_y.ctypes.data_as(POINTER(c_double)),
                c_src_z.ctypes.data_as(POINTER(c_double)),
                c_dl_x.ctypes.data_as(POINTER(c_double)),
                c_dl_y.ctypes.data_as(POINTER(c_double)),
                c_dl_z.ctypes.data_as(POINTER(c_double)),
                out_x.ctypes.data_as(POINTER(c_double)),
                out_y.ctypes.data_as(POINTER(c_double)),
                out_z.ctypes.data_as(POINTER(c_double)),
            )
            return out_x, out_y, out_z

        else:
            # GENERAL PATH: DAs
            c_b_x, c_b_y, c_b_z = CosyBackend.biot_savart_batch_indices(
                pos_x, pos_y, pos_z, src_x, src_y, src_z, dl_x, dl_y, dl_z
            )
            n_pts = len(pos_x)

            # Wrap results (Legacy behavior used by tests/direct callers)
            res_x = [CosyDA(idx=c_b_x[i], owned=True) for i in range(n_pts)]
            res_y = [CosyDA(idx=c_b_y[i], owned=True) for i in range(n_pts)]
            res_z = [CosyDA(idx=c_b_z[i], owned=True) for i in range(n_pts)]

            return res_x, res_y, res_z


class CosyDA:
    is_complex = False

    def __init__(
        self, create_new=False, idx=None, var_id=None, create_mode=None, owned=False
    ):
        self.owned = owned
        if create_mode == "new":
            create_new = True
        if idx is not None:
            self.idx = idx
        elif create_new:
            self.owned = True
            self.idx = CosyIndexPool.acquire()
        elif var_id is not None:
            self.owned = True
            # Use safe NDA creation to avoid NST issues
            res_idx = c_int(0)
            libcosy.create_nda_var(
                byref(res_idx), byref(c_double(0.0)), byref(c_int(var_id + 1))
            )
            self.idx = res_idx.value
        else:
            raise ValueError("Must provide idx, create_new=True, or var_id")

    def __repr__(self):
        return f"<CosyDA idx={self.idx}>"

    def __del__(self):
        if hasattr(self, "idx") and self.owned:
            CosyIndexPool.release(self.idx)

    def transfer_ownership(self) -> int:
        """
        Transfers ownership of the underlying Fortran DA index to the caller.

        After calling this method, this CosyDA object is marked as non-owning
        (``self.owned = False``) and will NOT free the index when it is garbage
        collected. The caller receives the raw integer index and is responsible
        for wrapping it in a new CosyDA (or CosyMtfData) with ``owned=True``.

        Returns
        -------
        int
            The raw Fortran DA index previously owned by this object.

        Example
        -------
        res_da = some_operation()        # owned=True
        idx = res_da.transfer_ownership()  # res_da.owned is now False
        new_wrapper = CosyDA(idx=idx, owned=True)
        """
        self.owned = False
        return self.idx

    def get_all_terms(self):
        max_order = CosyBackend._order
        dim = CosyBackend._dim
        from math import comb

        n_coeffs_est = comb(max_order + dim, dim)

        # Use batch getter
        max_len = c_int(n_coeffs_est + 100)  # Buffer
        c_vals = (c_double * max_len.value)()
        c_exps = (c_int * (max_len.value * CosyBackend._dim))()
        actual_len = c_int(0)

        libcosy.get_all_coeffs_flat(
            byref(c_int(self.idx)), c_vals, c_exps, byref(max_len), byref(actual_len)
        )

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
        libcosy.get_da_coeff(byref(c_int(self.idx)), c_exponents, byref(c_val))
        return c_val.value

    @classmethod
    def from_const(cls, val):
        if isinstance(val, (CosyDA, CosyCDA)):
            # If we are already a DA object, and we are being called via a subclass (like CosyCDA),
            # we might need to promote.
            if cls is CosyCDA and not val.is_complex:
                return val.to_complex()
            return val
        res_idx = c_int(0)
        libcosy.create_da_const(byref(res_idx), byref(c_double(float(val))))
        return cls(idx=res_idx.value, owned=True)

    def _should_promote(self, other):
        return (
            self.is_complex
            or isinstance(other, complex)
            or (isinstance(other, CosyDA) and other.is_complex)
        )

    def __add__(self, other):
        if self._should_promote(other):
            return self.to_complex() + other
        res_idx = c_int(CosyIndexPool.acquire())
        if isinstance(other, CosyDA):
            libcosy.compute_da_add(
                byref(c_int(self.idx)), byref(c_int(other.idx)), byref(res_idx)
            )
        elif isinstance(other, (int, float, np.number)):
            libcosy.compute_da_add_const(
                byref(c_int(self.idx)), byref(c_double(float(other))), byref(res_idx)
            )
        else:
            con = CosyDA.from_const(other)
            libcosy.compute_da_add(
                byref(c_int(self.idx)), byref(c_int(con.idx)), byref(res_idx)
            )
        return CosyDA(idx=res_idx.value, owned=True)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if self._should_promote(other):
            return self.to_complex() - other
        res_idx = c_int(CosyIndexPool.acquire())
        if isinstance(other, CosyDA):
            libcosy.compute_da_sub(
                byref(c_int(self.idx)), byref(c_int(other.idx)), byref(res_idx)
            )
        elif isinstance(other, (int, float, np.number)):
            libcosy.compute_da_sub_const(
                byref(c_int(self.idx)), byref(c_double(float(other))), byref(res_idx)
            )
        else:
            con = CosyDA.from_const(other)
            libcosy.compute_da_sub(
                byref(c_int(self.idx)), byref(c_int(con.idx)), byref(res_idx)
            )
        return CosyDA(idx=res_idx.value, owned=True)

    def __rsub__(self, other):
        if self._should_promote(other):
            return CosyCDA.from_const(other) - self
        res_idx = c_int(CosyIndexPool.acquire())
        if isinstance(other, (int, float, np.number)):
            libcosy.compute_da_sub_r_const(
                byref(c_int(self.idx)), byref(c_double(float(other))), byref(res_idx)
            )
        else:
            con = CosyDA.from_const(other)
            libcosy.compute_da_sub(
                byref(c_int(con.idx)), byref(c_int(self.idx)), byref(res_idx)
            )
        return CosyDA(idx=res_idx.value, owned=True)

    def __neg__(self):
        if self.is_complex:
            return self.to_complex() * -1.0
        res_idx = c_int(CosyIndexPool.acquire())
        con = CosyDA.from_const(0.0)
        libcosy.compute_da_sub(
            byref(c_int(con.idx)), byref(c_int(self.idx)), byref(res_idx)
        )
        return CosyDA(idx=res_idx.value, owned=True)

    def to_complex(self):
        res_idx = c_int(0)
        libcosy.compute_da_to_cd(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def __mul__(self, other):
        if self._should_promote(other):
            return self.to_complex() * other
        res_idx = c_int(CosyIndexPool.acquire())
        if isinstance(other, CosyDA):
            libcosy.compute_da_mul(
                byref(c_int(self.idx)), byref(c_int(other.idx)), byref(res_idx)
            )
        elif isinstance(other, (int, float, np.number)):
            # Optimized
            libcosy.compute_da_mul_const(
                byref(c_int(self.idx)), byref(c_double(float(other))), byref(res_idx)
            )
        else:
            return NotImplemented
        return CosyDA(idx=res_idx.value, owned=True)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if self._should_promote(other):
            return self.to_complex() / other
        res_idx = c_int(CosyIndexPool.acquire())
        if isinstance(other, CosyDA):
            libcosy.compute_da_div(
                byref(c_int(self.idx)), byref(c_int(other.idx)), byref(res_idx)
            )
        elif isinstance(other, (int, float, np.number)):
            libcosy.compute_da_div_const(
                byref(c_int(self.idx)), byref(c_double(float(other))), byref(res_idx)
            )
        else:
            con = CosyDA.from_const(other)
            libcosy.compute_da_div(
                byref(c_int(self.idx)), byref(c_int(con.idx)), byref(res_idx)
            )
        return CosyDA(idx=res_idx.value, owned=True)

    def __rtruediv__(self, other):
        if self._should_promote(other):
            return CosyCDA.from_const(other) / self
        res_idx = c_int(CosyIndexPool.acquire())
        if isinstance(other, (int, float, np.number)):
            libcosy.compute_da_div_r_const(
                byref(c_int(self.idx)), byref(c_double(float(other))), byref(res_idx)
            )
        else:
            con = CosyDA.from_const(other)
            libcosy.compute_da_div(
                byref(c_int(con.idx)), byref(c_int(self.idx)), byref(res_idx)
            )
        return CosyDA(idx=res_idx.value, owned=True)

    def deriv(self, var_id):
        res_idx = c_int(CosyIndexPool.acquire())
        c_var = c_int(var_id + 1)
        c_idx = c_int(self.idx)
        libcosy.da_deriv_safe(byref(c_var), byref(c_idx), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def integral(self, var_id):
        res_idx = c_int(CosyIndexPool.acquire())
        c_var = c_int(var_id + 1)
        c_idx = c_int(self.idx)
        libcosy.da_integ(byref(c_var), byref(c_idx), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def poisson_bracket(self, other):
        res_idx = c_int(CosyIndexPool.acquire())
        if isinstance(other, CosyDA):
            c_idx_self = c_int(self.idx)
            c_idx_other = c_int(other.idx)
            libcosy.da_poisson(byref(c_idx_self), byref(c_idx_other), byref(res_idx))
        else:
            # Constant 0 reset
            libcosy.da_reset(byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def sin(self):
        res_idx = c_int(CosyIndexPool.acquire())
        libcosy.compute_da_sin(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def cos(self):
        res_idx = c_int(CosyIndexPool.acquire())
        libcosy.compute_da_cos(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def exp(self):
        res_idx = c_int(CosyIndexPool.acquire())
        libcosy.compute_da_exp(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def log(self):
        res_idx = c_int(CosyIndexPool.acquire())
        libcosy.compute_da_log(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def sinh(self):
        res_idx = c_int(CosyIndexPool.acquire())
        libcosy.compute_da_sinh(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def cosh(self):
        res_idx = c_int(CosyIndexPool.acquire())
        libcosy.compute_da_cosh(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def tanh(self):
        res_idx = c_int(CosyIndexPool.acquire())
        libcosy.compute_da_tanh(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def tan(self):
        res_idx = c_int(CosyIndexPool.acquire())
        libcosy.compute_da_tan(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def sqrt(self):
        res_idx = c_int(CosyIndexPool.acquire())
        libcosy.compute_da_sqrt(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def arcsin(self):
        res_idx = c_int(CosyIndexPool.acquire())
        libcosy.compute_da_asin(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def arccos(self):
        res_idx = c_int(CosyIndexPool.acquire())
        libcosy.compute_da_acos(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def arctan(self):
        res_idx = c_int(CosyIndexPool.acquire())
        libcosy.compute_da_atan(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def inv_sqrt(self):
        res_idx = c_int(CosyIndexPool.acquire())
        libcosy.compute_da_isrt(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def inv_cbrt(self):
        res_idx = c_int(CosyIndexPool.acquire())
        libcosy.compute_da_isrt3(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def coth(self):
        res_idx = c_int(CosyIndexPool.acquire())
        libcosy.compute_da_coth(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def cot(self):
        res_idx = c_int(CosyIndexPool.acquire())
        libcosy.compute_da_cot(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def norm(self):
        val = c_double(0.0)
        libcosy.compute_da_norm(byref(c_int(self.idx)), byref(val))
        return val.value

    def weighted_norm(self, weight_da):
        val = c_double(0.0)
        libcosy.compute_da_wnorm(
            byref(c_int(self.idx)), byref(c_int(weight_da.idx)), byref(val)
        )
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
        libcosy.compute_da_daest(
            byref(c_int(self.idx)),
            byref(c_int(var_id)),
            byref(c_int(order)),
            byref(val),
        )
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

        is_complex = getattr(self, "is_complex", False) or any(
            getattr(arg, "is_complex", False) for arg in args_da_list
        )

        res_idx = c_int(CosyIndexPool.acquire(is_complex=is_complex))
        libcosy.compute_da_polval(
            byref(res_idx), byref(c_int(self.idx)), args_indices, byref(c_int(n_args))
        )
        if is_complex:
            return CosyCDA(idx=res_idx.value, owned=True)
        return CosyDA(idx=res_idx.value, owned=True)

    def inverse(self):
        res_idx = c_int(CosyIndexPool.acquire())
        libcosy.compute_da_mui(byref(c_int(self.idx)), byref(res_idx))
        return CosyDA(idx=res_idx.value, owned=True)

    def __pow__(self, exponent):
        res_idx = c_int(CosyIndexPool.acquire())
        if isinstance(exponent, int):
            libcosy.compute_da_pep(
                byref(c_int(self.idx)), byref(c_int(exponent)), byref(res_idx)
            )
        elif isinstance(exponent, float):
            libcosy.compute_da_pkp(
                byref(c_int(self.idx)), byref(c_double(exponent)), byref(res_idx)
            )
        else:
            CosyIndexPool.release(res_idx.value)
            return NotImplemented
        return CosyDA(idx=res_idx.value, owned=True)


class CosyCDA(CosyDA):
    is_complex = True

    @classmethod
    def from_const(cls, val):
        if isinstance(val, (CosyDA, CosyCDA)):
            if not val.is_complex:
                return val.to_complex()
            return val
        re_val = float(val.real) if hasattr(val, "real") else float(val)
        im_val = float(val.imag) if hasattr(val, "imag") else 0.0
        return cls(from_const=(re_val, im_val))

    def __init__(
        self,
        create_new=False,
        idx=None,
        from_var=None,
        from_const=None,
        create_mode=None,
        owned=False,
    ):
        self.owned = owned
        if create_mode == "new":
            create_new = True
        if idx is not None:
            self.idx = idx
        elif create_new:
            self.owned = True
            self.idx = CosyIndexPool.acquire(is_complex=True)
        elif from_var:
            var_id, real_val = from_var
            self.owned = True
            res_idx = c_int(0)
            libcosy.create_cda_var(
                byref(res_idx), byref(c_int(var_id)), byref(c_double(real_val))
            )
            self.idx = res_idx.value
        elif from_const:
            re_val, im_val = from_const
            self.owned = True
            res_idx = c_int(0)
            libcosy.create_cda_const(
                byref(res_idx), byref(c_double(re_val)), byref(c_double(im_val))
            )
            self.idx = res_idx.value
        else:
            raise ValueError(
                "Must provide idx, create_new=True, from_var, or from_const"
            )

    def __repr__(self):
        return f"<CosyCDA idx={self.idx}>"

    def __del__(self):
        if hasattr(self, "idx") and self.owned:
            CosyIndexPool.release(self.idx, is_complex=True)

    def get_constant(self):
        re_da = CosyDA(create_new=True)
        im_da = CosyDA(create_new=True)
        libcosy.get_cda_re(byref(c_int(self.idx)), byref(c_int(re_da.idx)))
        libcosy.get_cda_im(byref(c_int(self.idx)), byref(c_int(im_da.idx)))
        return complex(re_da.get_constant(), im_da.get_constant())

    def to_complex(self):
        """
        Returns a copy of this Complex DA.
        Since it is already complex, we just add 0 to create a new copy.
        """
        return self + 0.0

    def _ensure_cd(self, other):
        if isinstance(other, CosyCDA):
            return other
        if isinstance(other, CosyDA):
            return other.to_complex()
        if isinstance(other, (int, float, complex, np.number)):
            return CosyCDA.from_const(other)
        return NotImplemented

    def __truediv__(self, other):
        b = self._ensure_cd(other)
        if b is NotImplemented:
            return NotImplemented
        res_idx = c_int(0)
        libcosy.compute_cd_div(
            byref(c_int(self.idx)), byref(c_int(b.idx)), byref(res_idx)
        )
        return CosyCDA(idx=res_idx.value, owned=True)

    def __rtruediv__(self, other):
        a = self._ensure_cd(other)
        if a is NotImplemented:
            return NotImplemented
        return a / self

    def inverse(self):
        res_idx = c_int(0)
        libcosy.compute_cd_mui(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def __pow__(self, exponent):
        res_idx = c_int(0)
        if isinstance(exponent, int):
            libcosy.compute_cd_pei(
                byref(c_int(self.idx)), byref(c_int(exponent)), byref(res_idx)
            )
        elif isinstance(exponent, float):
            libcosy.compute_cd_pkp(
                byref(c_int(self.idx)), byref(c_double(exponent)), byref(res_idx)
            )
        else:
            return NotImplemented
        return CosyCDA(idx=res_idx.value, owned=True)

    def exp(self):
        res_idx = c_int(0)
        libcosy.compute_cd_exp(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def log(self):
        res_idx = c_int(0)
        libcosy.compute_cd_log(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def sin(self):
        res_idx = c_int(0)
        libcosy.compute_cd_sin(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def cos(self):
        res_idx = c_int(0)
        libcosy.compute_cd_cos(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def tan(self):
        res_idx = c_int(0)
        libcosy.compute_cd_tan(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def sinh(self):
        res_idx = c_int(0)
        libcosy.compute_cd_sinh(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def cosh(self):
        res_idx = c_int(0)
        libcosy.compute_cd_cosh(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def tanh(self):
        res_idx = c_int(0)
        libcosy.compute_cd_tanh(byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def sqrt(self):
        res_idx = c_int(0)
        libcosy.compute_cd_sqrt(byref(c_int(self.idx)), byref(res_idx))
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
            libcosy.get_cda_coeff_by_index(
                byref(c_int(self.idx)),
                byref(c_int(i)),
                c_exps,
                byref(c_val_re),
                byref(c_val_im),
            )
            re, im = c_val_re.value, c_val_im.value
            if abs(re) > 1e-15 or abs(im) > 1e-15:
                exps = tuple(c_exps[k] for k in range(dim))
                coeffs.append((exps, complex(re, im)))
        return coeffs

    def __add__(self, other):
        res_idx = c_int(0)
        if isinstance(other, CosyCDA):
            libcosy.compute_cd_add(
                byref(c_int(self.idx)), byref(c_int(other.idx)), byref(res_idx)
            )
        elif isinstance(other, CosyDA):
            other_cd = other.to_complex()
            libcosy.compute_cd_add(
                byref(c_int(self.idx)), byref(c_int(other_cd.idx)), byref(res_idx)
            )
        else:
            re_v = float(other.real) if hasattr(other, "real") else float(other)
            im_v = float(other.imag) if hasattr(other, "imag") else 0.0
            con = CosyCDA(from_const=(re_v, im_v))
            libcosy.compute_cd_add(
                byref(c_int(self.idx)), byref(c_int(con.idx)), byref(res_idx)
            )
        return CosyCDA(idx=res_idx.value, owned=True)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        res_idx = c_int(0)
        if isinstance(other, CosyCDA):
            libcosy.compute_cd_sub(
                byref(c_int(self.idx)), byref(c_int(other.idx)), byref(res_idx)
            )
        elif isinstance(other, CosyDA):
            other_cd = other.to_complex()
            libcosy.compute_cd_sub(
                byref(c_int(self.idx)), byref(c_int(other_cd.idx)), byref(res_idx)
            )
        else:
            re_v = float(other.real) if hasattr(other, "real") else float(other)
            im_v = float(other.imag) if hasattr(other, "imag") else 0.0
            con = CosyCDA(from_const=(re_v, im_v))
            libcosy.compute_cd_sub(
                byref(c_int(self.idx)), byref(c_int(con.idx)), byref(res_idx)
            )
        return CosyCDA(idx=res_idx.value, owned=True)

    def __rsub__(self, other):
        a = self._ensure_cd(other)
        if a is NotImplemented:
            return NotImplemented
        res_idx = c_int(0)
        libcosy.compute_cd_sub(
            byref(c_int(a.idx)), byref(c_int(self.idx)), byref(res_idx)
        )
        return CosyCDA(idx=res_idx.value, owned=True)

    def __mul__(self, other):
        res_idx = c_int(0)
        if isinstance(other, CosyCDA):
            libcosy.compute_cd_mul(
                byref(c_int(self.idx)), byref(c_int(other.idx)), byref(res_idx)
            )
        elif isinstance(other, CosyDA):
            other_cd = other.to_complex()
            libcosy.compute_cd_mul(
                byref(c_int(self.idx)), byref(c_int(other_cd.idx)), byref(res_idx)
            )
        else:
            re_v = float(other.real) if hasattr(other, "real") else float(other)
            im_v = float(other.imag) if hasattr(other, "imag") else 0.0
            con = CosyCDA(from_const=(re_v, im_v))
            libcosy.compute_cd_mul(
                byref(c_int(self.idx)), byref(c_int(con.idx)), byref(res_idx)
            )
        return CosyCDA(idx=res_idx.value, owned=True)

    def __rmul__(self, other):
        return self.__mul__(other)

    def deriv(self, var_id):
        res_idx = c_int(0)
        c_var = c_int(var_id + 1)
        libcosy.compute_cd_der(byref(c_var), byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def integral(self, var_id):
        res_idx = c_int(0)
        c_var = c_int(var_id + 1)
        libcosy.compute_cd_int(byref(c_var), byref(c_int(self.idx)), byref(res_idx))
        return CosyCDA(idx=res_idx.value, owned=True)

    def poisson_bracket(self, other):
        b = self._ensure_cd(other)
        if b is NotImplemented:
            return NotImplemented
        res_idx = c_int(0)
        libcosy.compute_cd_poi(
            byref(c_int(self.idx)), byref(c_int(b.idx)), byref(res_idx)
        )
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
    def __init__(
        self, dimension: int = None, is_complex: bool = False, idx=None, owned=False
    ):
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
                raise ValueError(
                    f"Requested dimension {dimension} > COSY backend dimension {self.physical_dimension}"
                )
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
        if len(coeffs) == 0:
            return

        current_dim = exponents.shape[1]
        # Pad exponents if necessary to match global COSY dimension
        global_dim = self.physical_dimension

        if current_dim < global_dim:
            padded_exponents = np.zeros(
                (exponents.shape[0], global_dim), dtype=np.int32
            )
            padded_exponents[:, :current_dim] = exponents
            flat_exps = padded_exponents.flatten()
        elif current_dim > global_dim:
            raise ValueError(
                f"Exponents dimension {current_dim} exceeds COSY global dimension {global_dim}"
            )
        else:
            flat_exps = exponents.flatten().astype(np.int32)

        c_exps = (c_int * len(flat_exps))(*flat_exps)

        if isinstance(self.da, CosyCDA) or np.iscomplexobj(coeffs):
            # Ensure we have complex DA
            if not isinstance(self.da, CosyCDA):
                old_idx = self.da.idx
                self.da = CosyCDA(create_mode="new")
                # Note: We lose old real data here if we don't copy,
                # but from_numpy usually overwrites.

            flat_coeffs = coeffs.astype(np.complex128).flatten()
            flat_re = flat_coeffs.real.astype(np.float64)
            flat_im = flat_coeffs.imag.astype(np.float64)

            c_re = (c_double * len(flat_re))(*flat_re)
            c_im = (c_double * len(flat_im))(*flat_im)

            # Use specific complex setter if available, otherwise manual split
            if hasattr(libcosy, "cosy_set_cd_coeffs_"):
                # Fast Path
                bind_cosy_func(
                    "cosy_set_cd_coeffs",
                    [
                        POINTER(c_int),
                        POINTER(c_double),
                        POINTER(c_double),
                        POINTER(c_int),
                        POINTER(c_int),
                    ],
                )
                libcosy.cosy_set_cd_coeffs(
                    byref(c_int(self.da.idx)),
                    c_re,
                    c_im,
                    c_exps,
                    byref(c_int(len(coeffs))),
                )
            else:
                # Robust Fallback: Set Real and Imag parts separately
                # 1. Create temporary Real DAs
                re_da = CosyDA(create_new=True)
                im_da = CosyDA(create_new=True)

                # 2. Set coefficients for them
                libcosy.cosy_set_coeffs(
                    byref(c_int(re_da.idx)), c_re, c_exps, byref(c_int(len(coeffs)))
                )
                libcosy.cosy_set_coeffs(
                    byref(c_int(im_da.idx)), c_im, c_exps, byref(c_int(len(coeffs)))
                )

                # 3. Merge into Complex DA
                libcosy.set_cd_parts(
                    byref(c_int(self.da.idx)),
                    byref(c_int(re_da.idx)),
                    byref(c_int(im_da.idx)),
                )

        else:
            # Real case
            flat_coeffs = coeffs.astype(np.float64).flatten()
            c_vals = (c_double * len(flat_coeffs))(*flat_coeffs)
            libcosy.cosy_set_coeffs(
                byref(c_int(self.da.idx)), c_vals, c_exps, byref(c_int(len(coeffs)))
            )

    def get_constant(self):
        return self.da.get_constant()

    def inverse(self):
        c0 = self.get_constant()
        if abs(c0) < 1e-14:
            raise ValueError(
                "Inversion of near-zero constant term (constant part is effectively zero)."
            )
        res_da = self.da.inverse()
        idx = res_da.transfer_ownership()
        is_complex = isinstance(res_da, CosyCDA)
        return CosyMtfData(self.dimension, is_complex=is_complex, idx=idx, owned=True)

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            res_da = self.da**other
            if hasattr(res_da, "owned"):
                res_da.owned = False
            is_complex = isinstance(res_da, CosyCDA)
            return CosyMtfData(
                self.dimension, is_complex=is_complex, idx=res_da.idx, owned=True
            )
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
            data["exponents"] = all_exponents[:, : self.dimension]
        else:
            data["exponents"] = all_exponents

        data["coeffs"] = coeffs
        return data

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
                point_flat = point.flatten()  # Assumes dim == NVMAX

            if isinstance(self.da, CosyCDA):
                # Create temporary real DAs
                re_da = CosyDA(create_new=True)
                im_da = CosyDA(create_new=True)
                libcosy.get_cda_re(byref(c_int(self.da.idx)), byref(c_int(re_da.idx)))
                libcosy.get_cda_im(byref(c_int(self.da.idx)), byref(c_int(im_da.idx)))

                re_vals = np.zeros(n_points, dtype=np.float64)
                im_vals = np.zeros(n_points, dtype=np.float64)
                max_terms = 100000
                c_max_terms = c_int(max_terms)
                nvmax = 40
                temp_exps = np.zeros(max_terms * nvmax, dtype=np.int32)
                temp_coeffs = np.zeros(max_terms, dtype=np.float64)

                c_temp_exps = (c_int * len(temp_exps)).from_buffer(temp_exps)
                c_temp_coeffs = (c_double * len(temp_coeffs)).from_buffer(temp_coeffs)
                c_points = (c_double * len(point_flat)).from_buffer(point_flat)
                c_n_points = c_int(n_points)

                c_re_vals = (c_double * len(re_vals)).from_buffer(re_vals)
                libcosy.eval_da_batch(
                    byref(c_int(re_da.idx)),
                    c_points,
                    byref(c_n_points),
                    c_re_vals,
                    c_temp_exps,
                    c_temp_coeffs,
                    byref(c_max_terms),
                )

                # Reset temp arrays before evaluating imaginary part
                temp_exps.fill(0)
                temp_coeffs.fill(0.0)
                c_im_vals = (c_double * len(im_vals)).from_buffer(im_vals)
                libcosy.eval_da_batch(
                    byref(c_int(im_da.idx)),
                    c_points,
                    byref(c_n_points),
                    c_im_vals,
                    c_temp_exps,
                    c_temp_coeffs,
                    byref(c_max_terms),
                )

                return re_vals + 1j * im_vals

            vals = np.zeros(n_points, dtype=np.float64)
            max_terms = 100000
            c_max_terms = c_int(max_terms)

            # Allocating workspace arrays
            # TEMP_EXPS: (MAX_TERMS, 40) integers
            # TEMP_COEFFS: (MAX_TERMS) doubles
            nvmax = 40  # Standard COSY limit
            temp_exps = np.zeros(max_terms * nvmax, dtype=np.int32)
            temp_coeffs = np.zeros(max_terms, dtype=np.float64)

            c_temp_exps = (c_int * len(temp_exps)).from_buffer(temp_exps)
            c_temp_coeffs = (c_double * len(temp_coeffs)).from_buffer(temp_coeffs)

            c_points = (c_double * len(point_flat)).from_buffer(point_flat)
            c_vals = (c_double * len(vals)).from_buffer(vals)
            c_n_points = c_int(n_points)

            libcosy.eval_da_batch(
                byref(c_int(self.da.idx)),
                c_points,
                byref(c_n_points),
                c_vals,
                c_temp_exps,
                c_temp_coeffs,
                byref(c_max_terms),
            )

            return vals
        # Handle Single Point Evaluation (1D input)
        elif point.ndim == 1:
            points_reshaped = point.reshape(1, -1)
            # Recursively call with 2D array and extract result
            return self.eval(points_reshaped)[0]
        else:
            raise ValueError(f"Invalid input shape {point.shape}")

    def add(self, other):
        if isinstance(other, (int, float, complex, np.number)):
            # Create constant CosyMtfData? Or handle in underlying CosyDA
            # CosyDA handles scalars in __add__.
            # We need to extract .da if it's CosyMtfData
            return self._create_res(self.da + other)
        return self._create_res(self.da + other.da)

    def subtract(self, other):
        if isinstance(other, (int, float, complex, np.number)):
            return self._create_res(self.da - other)
        return self._create_res(self.da - other.da)

    def multiply(self, other):
        if isinstance(other, (int, float, complex, np.number)):
            return self._create_res(self.da * other)
        return self._create_res(self.da * other.da)

    def multiply_inplace(self, other):
        if isinstance(other, (int, float, complex, np.number)):
            self.da = self.da * other
        else:
            self.da = self.da * other.da

    def divide(self, other):
        if isinstance(other, (int, float, complex, np.number)):
            return self._create_res(self.da / other)

        c0 = other.get_constant()
        if abs(c0) < 1e-14:
            raise ValueError(
                "Division by near-zero constant term (constant part is effectively zero)."
            )
        return self._create_res(self.da / other.da)

    def negate(self):
        return self._create_res(-self.da)

    # --- Operator Overloading for CosyMtfData ---
    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.subtract(other)

    def __rsub__(self, other):
        if isinstance(other, (int, float, complex, np.number)):
            return self._create_res(other - self.da)
        return NotImplemented  # Should be handled by other.__sub__

    def __mul__(self, other):
        return self.multiply(other)

    def __rmul__(self, other):
        return self.multiply(other)

    def __truediv__(self, other):
        return self.divide(other)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, complex, np.number)):
            return self._create_res(other / self.da)
        return NotImplemented

    def __neg__(self):
        return self.negate()

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
        is_complex = isinstance(res_da, (CosyCDA, complex))
        if not is_complex and hasattr(res_da, "is_complex"):
            is_complex = res_da.is_complex
        if isinstance(res_da, CosyCDA):
            is_complex = True

        # Use transfer_ownership() to safely move the Fortran index from the
        # temporary res_da to the new CosyMtfData, preventing double-free.
        if hasattr(res_da, "transfer_ownership"):
            idx = res_da.transfer_ownership()
        elif hasattr(res_da, "idx"):
            # Fallback for objects without the method (should not normally occur)
            idx = res_da.idx
            if hasattr(res_da, "owned"):
                res_da.owned = False
        else:
            idx = None

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
                raise ValueError(
                    "Square root undefined for negative constant part (Real domain)."
                )
        return self._create_res(self.da.sqrt())

    def asin(self):
        return self._create_res(
            self.da.asin() if hasattr(self.da, "asin") else self.da.arcsin()
        )

    def acos(self):
        return self._create_res(
            self.da.acos() if hasattr(self.da, "acos") else self.da.arccos()
        )

    def atan(self):
        return self._create_res(
            self.da.atan() if hasattr(self.da, "atan") else self.da.arctan()
        )

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

    libcosy.da_mat_inv(mat_arr, inv_arr, byref(c_int(n)))

    return [inv_arr[i] for i in range(n * n)]


def da_lin_comb(da1, c1, da2, c2):
    """
    Compute Linear Combination: res = c1*da1 + c2*da2
    """
    res_idx = c_int(0)
    libcosy.create_da_var(byref(res_idx), byref(c_double(0.0)), byref(c_int(0)))

    libcosy.da_lin_comb(
        byref(c_int(da1.idx)),
        byref(c_double(c1)),
        byref(c_int(da2.idx)),
        byref(c_double(c2)),
        byref(res_idx),
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

    # Acquire pooled indices for results
    res_indices = [CosyIndexPool.acquire() for _ in range(n)]
    res_ptr = np.array(res_indices, dtype=np.int32)

    return n, a_ptr, b_ptr, res_ptr


def batch_add(idx_arr_a, idx_arr_b):
    n, a_ptr, b_ptr, res_ptr = _prepare_batch_args(idx_arr_a, idx_arr_b)
    libcosy.compute_da_add_batch(
        byref(c_int(n)),
        a_ptr.ctypes.data_as(POINTER(c_int)),
        b_ptr.ctypes.data_as(POINTER(c_int)),
        res_ptr.ctypes.data_as(POINTER(c_int)),
    )
    return res_ptr


def batch_sub(idx_arr_a, idx_arr_b):
    n, a_ptr, b_ptr, res_ptr = _prepare_batch_args(idx_arr_a, idx_arr_b)
    libcosy.compute_da_sub_batch(
        byref(c_int(n)),
        a_ptr.ctypes.data_as(POINTER(c_int)),
        b_ptr.ctypes.data_as(POINTER(c_int)),
        res_ptr.ctypes.data_as(POINTER(c_int)),
    )
    return res_ptr


def batch_mul(idx_arr_a, idx_arr_b):
    n, a_ptr, b_ptr, res_ptr = _prepare_batch_args(idx_arr_a, idx_arr_b)
    libcosy.compute_da_mul_batch(
        byref(c_int(n)),
        a_ptr.ctypes.data_as(POINTER(c_int)),
        b_ptr.ctypes.data_as(POINTER(c_int)),
        res_ptr.ctypes.data_as(POINTER(c_int)),
    )
    return res_ptr


def batch_div(idx_arr_a, idx_arr_b):
    n, a_ptr, b_ptr, res_ptr = _prepare_batch_args(idx_arr_a, idx_arr_b)
    libcosy.compute_da_div_batch(
        byref(c_int(n)),
        a_ptr.ctypes.data_as(POINTER(c_int)),
        b_ptr.ctypes.data_as(POINTER(c_int)),
        res_ptr.ctypes.data_as(POINTER(c_int)),
    )
    return res_ptr
