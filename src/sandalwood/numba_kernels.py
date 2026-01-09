
"""
Numba-accelerated kernels for Taylor Function operations.
"""
try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    # Dummy decorator to allow import even if numba is missing (though we shouldn't use this file then)
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@njit(fastmath=True, cache=True)
def multiply_dense_kernel(idx_a, coeffs_a, idx_b, coeffs_b, table, result_coeffs):
    """
    Performs multiplication of two sparse-indexed polynomials using a dense lookup table.
    
    Parameters
    ----------
    idx_a : array(int32)
        Indices of terms in the first polynomial.
    coeffs_a : array(complex128/float64)
        Coefficients of the first polynomial.
    idx_b : array(int32)
        Indices of terms in the second polynomial.
    coeffs_b : array(complex128/float64)
        Coefficients of the second polynomial.
    table : 2D array(int32)
        Precomputed multiplication table. table[i, j] gives the index of the result term.
        -1 indicates truncation.
    result_coeffs : array(complex128/float64)
        Accumulator array for the result coefficients. Must be zero-initialized and
        of size equal to the total number of terms in the table.
        
    Returns
    -------
    None (result is accumulated in result_coeffs)
    """
    # Simply iterate over all pairs
    # Since we are memory-bound, we want to iterate linearly through result if possible,
    # but the table is random access.
    # Iterating through A and B is the standard way.
    
    n_a = len(idx_a)
    n_b = len(idx_b)
    
    for i in range(n_a):
        idx_i = idx_a[i]
        c_i = coeffs_a[i]
        for j in range(n_b):
            idx_j = idx_b[j]
            
            # Lookup result index
            res_idx = table[idx_i, idx_j]
            
            if res_idx != -1:
                result_coeffs[res_idx] += c_i * coeffs_b[j]
