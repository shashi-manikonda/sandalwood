
"""
Numba-accelerated kernels for Taylor Function operations.
"""
import numpy as np

try:
    from numba import njit, prange, config
    _NUMBA_AVAILABLE = True
    # config.THREADING_LAYER = 'workqueue' # Optional: sometimes helps stability
except ImportError:
    _NUMBA_AVAILABLE = False
    # Dummy decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(n):
        return range(n)

@njit(fastmath=True, cache=True, parallel=True)
def evaluate_dense_kernel(points, exponents, coeffs, result):
    """
    Parallelized evaluation kernel.
    Avoiding memory allocation of (N_points, N_terms, Dim).
    
    Parameters
    ----------
    points : (N_pts, Dim) array
    exponents : (N_terms, Dim) array (int32)
    coeffs : (N_terms) array (float/complex)
    result : (N_pts) array (output)
    """
    n_pts = points.shape[0]
    n_terms = len(coeffs)
    dim = points.shape[1]
    
    # Parallelize over points (independent)
    for i in prange(n_pts):
        sum_val = 0.0
        # Iterate terms
        for j in range(n_terms):
            term_val = coeffs[j]
            # Compute monomial value: x1^p1 * x2^p2 ...
            for d in range(dim):
                power = exponents[j, d]
                if power > 0:
                    # Optimization: Precomputing powers outside this loop 
                    # for small orders is faster, but this is memory efficient.
                    term_val *= points[i, d] ** power
            sum_val += term_val
        result[i] = sum_val

@njit(fastmath=True, cache=True)
def multiply_dense_parallel(idx_a, coeffs_a, idx_b, coeffs_b, table, result_coeffs):
    """
    Dense multiplication using thread-private accumulation (Serial for now).
    
    Strategy:
    1. Iterate over idx_a.
    2. Iterate over idx_b.
    3. Accumulate into result_coeffs.
    
    Parallelization note: 
    Numba's parallel reduction is tricky with arrays. 
    Race conditions on result_coeffs[res_idx] are fatal.
    Falling back to optimized serial loop for safety.
    """
    n_a = len(idx_a)
    n_b = len(idx_b)
    
    for i in range(n_a):
        idx_i = idx_a[i]
        if idx_i == -1:
            continue
        c_i = coeffs_a[i]
        
        # Inner loop is where the work is.
        # This access pattern table[idx_i, ...] is row-sequential(ish).
        for j in range(n_b):
            idx_j = idx_b[j]
            if idx_j == -1:
                continue
            res_idx = table[idx_i, idx_j]
            
            if res_idx != -1:
                result_coeffs[res_idx] += c_i * coeffs_b[j]
