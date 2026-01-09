"""
Numba-accelerated kernels for Taylor Function operations.
"""
import numpy as np

try:
    from numba import njit, prange, get_num_threads
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    # Dummy decorators/functions for fallback
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(n):
        return range(n)
    def get_num_threads():
        return 1

@njit(fastmath=True, cache=True, parallel=True)
def evaluate_dense_kernel(points, exponents, coeffs, result):
    """
    Parallelized evaluation with power caching.
    Avoids expensive pow() calls in the inner loop.
    """
    n_pts = points.shape[0]
    n_terms = len(coeffs)
    dim = points.shape[1]
    
    # Pre-calculate max order to size the power cache
    # Assumes exponents are int32. 
    # We can find max order by scanning or passing it in. 
    # For optimization, we assume a reasonable static cap or scan quickly.
    max_order = 0
    for j in range(n_terms):
        for d in range(dim):
            if exponents[j, d] > max_order:
                max_order = exponents[j, d]
    
    # Parallelize over points
    for i in prange(n_pts):
        # 1. Thread-local Power Cache: cache[d, p] = points[i, d]**p
        # Size is small: Dim * (MaxOrder+1)
        # We manually manage this "stack" array
        pow_cache = np.ones((dim, max_order + 1), dtype=points.dtype)
        
        for d in range(dim):
            val = points[i, d]
            current_pow = 1.0
            # pow_cache[d, 0] is already 1.0
            for p in range(1, max_order + 1):
                current_pow *= val
                pow_cache[d, p] = current_pow
        
        # 2. Compute Sum
        sum_val = 0.0
        for j in range(n_terms):
            term_val = coeffs[j]
            for d in range(dim):
                exp = exponents[j, d]
                if exp > 0:
                    term_val *= pow_cache[d, exp]
            sum_val += term_val
            
        result[i] = sum_val

@njit(fastmath=True, cache=True, parallel=True)
def multiply_dense_parallel(idx_a, coeffs_a, idx_b, coeffs_b, table, result_size):
    """
    True Parallel Multiplication using Map-Reduce.
    
    Args:
        result_size (int): The size of the full dense coefficient vector.
    Returns:
        accumulated_result (array): The final dense coefficients.
    """
    n_a = len(idx_a)
    n_b = len(idx_b)
    
    # 1. Allocate thread-local buffers
    # shape: (n_threads, result_size)
    num_threads = get_num_threads()
    thread_buffers = np.zeros((num_threads, result_size), dtype=coeffs_a.dtype)
    
    # 2. Parallel Accumulation
    # We parallelize the OUTER loop (idx_a)
    for i in prange(n_a):
        tid = np.uint32(i % num_threads) # Simple round-robin thread ID mapping
        
        idx_i = idx_a[i]
        c_i = coeffs_a[i]
        
        # Inner loop: Iterate over B
        for j in range(n_b):
            res_idx = table[idx_i, idx_b[j]]
            if res_idx != -1:
                # Accumulate into thread-private buffer
                thread_buffers[tid, res_idx] += c_i * coeffs_b[j]

    # 3. Reduction (Sum buffers)
    # Collapse the thread dimension
    final_result = np.sum(thread_buffers, axis=0)
    return final_result
