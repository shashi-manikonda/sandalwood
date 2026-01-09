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
    def evaluate_dense_kernel(*args): pass
    def multiply_dense_parallel(*args): pass

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
    max_order = 0
    for j in range(n_terms):
        for d in range(dim):
            if exponents[j, d] > max_order:
                max_order = exponents[j, d]
    
    # Parallelize over points
    for i in prange(n_pts):
        # 1. Thread-local Power Cache
        pow_cache = np.ones((dim, max_order + 1), dtype=points.dtype)
        
        for d in range(dim):
            val = points[i, d]
            current_pow = 1.0
            for p in range(1, max_order + 1):
                current_pow *= val
                pow_cache[d, p] = current_pow
        
        # 2. Compute Sum
        # Initialize accumulator with zero of the RESULT type (handling complex)
        sum_val = result[i] * 0
        
        for j in range(n_terms):
            term_val = coeffs[j]
            for d in range(dim):
                exp = exponents[j, d]
                if exp > 0:
                    term_val *= pow_cache[d, exp]
            sum_val += term_val
            
        result[i] = sum_val

@njit(fastmath=True, parallel=True)
def multiply_dense_parallel(idx_a, coeffs_a, idx_b, coeffs_b, table, result_size):
    """
    True Parallel Multiplication using Map-Reduce with Manual Chunking.
    """
    n_a = len(idx_a)
    n_b = len(idx_b)
    
    # Use the config to determine thread count
    num_threads = get_num_threads()
    
    # 1. Allocate buffers (Complex support depends on input dtype)
    thread_buffers = np.zeros((num_threads, result_size), dtype=coeffs_a.dtype)
    
    # 2. Manual Chunking to ensure Thread Isolation
    # We iterate over THREADS, not data indices directly
    chunk_size = (n_a + num_threads - 1) // num_threads
    
    for t in prange(num_threads):
        start = t * chunk_size
        end = min((t + 1) * chunk_size, n_a)
        
        # Each thread processes its exclusive chunk of A
        for i in range(start, end):
            idx_i = idx_a[i]
            c_i = coeffs_a[i]
            
            for j in range(n_b):
                res_idx = table[idx_i, idx_b[j]]
                if res_idx != -1:
                    # Safe: Only thread 't' writes to row 't'
                    thread_buffers[t, res_idx] += c_i * coeffs_b[j]

    # 3. Reduction
    final_result = np.sum(thread_buffers, axis=0)
    return final_result
