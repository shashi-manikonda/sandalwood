"""
Numba-accelerated kernels for Taylor Function operations.

See Also
--------
:ref:`optimization` : For a detailed explanation of the Dense Mode and Numba optimization strategy.
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
    Parallelized evaluation with block-based power caching and vectorization.
    Explicitly handles memory layout to maximize SIMD usage.
    
    Strategy:
    1. Process points in blocks to keep data in L1/L2 cache.
    2. Pre-compute powers for the block in a transposed layout (dim, max_order, block_size).
    3. Vectorize the summation over the block of points for each term.
    """
    n_pts = points.shape[0]
    n_terms = len(coeffs)
    dim = points.shape[1]
    
    # Pre-calculate max order
    max_order = 0
    for j in range(n_terms):
        for d in range(dim):
            if exponents[j, d] > max_order:
                max_order = exponents[j, d]
                
    # Block size optimization
    # 256 is a reasonable balance for cache locality and vector length
    BLOCK_SIZE = 256
    
    # Outer parallel loop over blocks of points
    # We round up the number of blocks
    num_blocks = (n_pts + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for b in prange(num_blocks):
        start_idx = b * BLOCK_SIZE
        end_idx = min(start_idx + BLOCK_SIZE, n_pts)
        actual_size = end_idx - start_idx
        
        if actual_size <= 0:
            continue
            
        # 1. Compute Power Cache for this block
        # Layout: (dim, max_order + 1, actual_size) -> Contiguous in 'actual_size' (SIMD friendly)
        # Note: Numba is smart enough to stack-allocate small arrays or heap-allocate efficiently
        pow_cache = np.ones((dim, max_order + 1, actual_size), dtype=points.dtype)
        
        for d in range(dim):
            # Base values (order 1)
            # Copy points slice to contiguous memory if needed, but here we just read
            current_vals = points[start_idx:end_idx, d]
            
            # Fill powers
            # pow_cache[d, 0, :] is already 1.0
            
            # Fill Order 1
            for k in range(actual_size):
                pow_cache[d, 1, k] = current_vals[k]
                
            # Fill Higher Orders recursively
            for p in range(2, max_order + 1):
                for k in range(actual_size):
                    pow_cache[d, p, k] = pow_cache[d, p-1, k] * current_vals[k]

        # 2. Accumulate Terms
        # Initialize block accumulator with zeros
        # We infer type from result array to handle complex support
        block_result = np.zeros(actual_size, dtype=result.dtype)
        
        for j in range(n_terms):
            coeff = coeffs[j]
            
            # Start with coefficient
            term_values = np.empty(actual_size, dtype=result.dtype)
            for k in range(actual_size):
                term_values[k] = coeff
            
            # Multiply by powers of each variable
            for d in range(dim):
                exp = exponents[j, d]
                if exp > 0:
                    # SIMD multiplication
                    # term_values *= pow_cache[d, exp]
                    for k in range(actual_size):
                        term_values[k] *= pow_cache[d, exp, k]
            
            # Add to accumulator
            for k in range(actual_size):
                block_result[k] += term_values[k]
                
        # 3. Write back
        for k in range(actual_size):
            result[start_idx + k] = block_result[k]

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
