"""
Numba-accelerated kernels for Taylor Function operations.

See Also
--------
:ref:`optimization` : For a detailed explanation of the Dense Mode and Numba optimization strategy.
"""

import numpy as np

try:
    from numba import get_num_threads, njit, prange

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

    def evaluate_dense_kernel(*args):
        pass

    def multiply_dense_parallel(*args):
        pass


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
                    pow_cache[d, p, k] = pow_cache[d, p - 1, k] * current_vals[k]

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


@njit(fastmath=True)
def dense_mul(A, B, table, out):
    """
    Multiplies dense polynomials A and B into out using precomputed table.
    A, B, out are 1D arrays of size n_terms.
    """
    # Reset out
    out[:] = 0.0

    n_a = len(A)
    n_b = len(B)

    # Iterate over A and B
    for i in range(n_a):
        val_a = A[i]
        if abs(val_a) < 1e-16:
            continue

        for j in range(n_b):
            val_b = B[j]
            if abs(val_b) < 1e-16:
                continue

            idx = table[i, j]
            if idx != -1:
                out[idx] += val_a * val_b


@njit(parallel=True)
def compose_dense_kernel(outer_exps, outer_coeffs, inner_powers, table, n_terms):
    """
    Optimized kernel for TaylorMap composition using dense arrays.

    Parameters
    ----------
    outer_exps : (N, dim) int array
        Exponents of the terms in the outer map component.
    outer_coeffs : (N,) float/complex array
        Coefficients of the outer map component.
    inner_powers : (dim, max_order+1, n_terms) float/complex array
        Precomputed powers of the inner map components.
        inner_powers[d, p, :] is the dense array for (component_d)^p.
    table : (n_terms, n_terms) int array
        The multiplication table.
    n_terms : int
        The size of the dense vector space.

    Returns
    -------
    final_result : (n_terms,) float/complex array
        Dense coefficients of the composition result.
    """
    n_outer_terms = len(outer_coeffs)
    dim = outer_exps.shape[1]

    # Thread-local storage for accumulation
    # Shape: (num_threads, n_terms)
    num_threads = get_num_threads()

    # We infer dtype from the coefficients
    result_dtype = outer_coeffs.dtype
    thread_accumulators = np.zeros((num_threads, n_terms), dtype=result_dtype)

    # Parallel loop over outer terms
    for i in prange(n_outer_terms):
        tid = 0  # Default for single thread
        if num_threads > 1:
            # Get thread ID (requires OpenMP backend usually, or we use explicit chunking to avoid race)
            # Numba prange automatic reduction is safer, but we are doing complex logic.
            # We will use manual reduction into thread_accumulators using chunk logic implies we need 't'
            # But prange doesn't give 't'.
            # Pattern: Use a simple manual loop chunking strategy similar to multiply_dense_parallel
            # if we want explicit buffers, OR use Numba's automatic reduction if possible.
            # However, automatic reduction for array operations is tricky.
            pass

    # Better Strategy for Parallelism compatible with Numba:
    # We split the work manually into chunks based on thread ID, just like multiply_dense_parallel

    chunk_size = (n_outer_terms + num_threads - 1) // num_threads

    for t in prange(num_threads):
        # Determine range for this thread
        start = t * chunk_size
        end = min((t + 1) * chunk_size, n_outer_terms)

        if start >= end:
            continue

        # Thread-specific scratchpads
        # We need a 'current_poly' accumulator for the product term
        # And a temporary buffer for intermediate multiplications
        current_term = np.zeros(n_terms, dtype=result_dtype)
        temp_buffer = np.zeros(n_terms, dtype=result_dtype)

        for k in range(start, end):
            coeff = outer_coeffs[k]

            # Initialize current_term = coeff (scalar constant)
            # In dense representation, constant term is at index 0 (assuming order logic)
            # BUT we prefer to construct it: product starts as Identity (1.0) * coeff

            # Reset current_term to represent just the scalar 'coeff'
            # We assume constant index is 0.
            # Let's double check standard ordering: (0,0,...) is usually first.
            current_term[:] = 0.0
            current_term[0] = coeff

            # Multiply by powers of each variable
            for d in range(dim):
                p = outer_exps[k, d]
                if p > 0:
                    # Multiply current_term * inner_powers[d, p]
                    # Store in temp_buffer
                    dense_mul(current_term, inner_powers[d, p], table, temp_buffer)

                    # Swap buffers: copy temp back to current
                    # Or just copy. For simplicity/clarity: copy.
                    current_term[:] = temp_buffer[:]

            # Add computed term contribution to thread accumulator
            thread_accumulators[t] += current_term

    # Reduction across threads
    final_result = np.sum(thread_accumulators, axis=0)
    return final_result
