
import numpy as np
import pytest

import sandalwood.taylor_function as taylor

# Common test parameters
MAX_ORDER = 4
MAX_DIMENSION = 2
ETOL = 1e-12

@pytest.fixture(scope="function", autouse=True)
def setup_mtf():
    # Force COSY backend if available, else skip
    if not taylor._COSY_BACKEND_AVAILABLE:
        pytest.skip("COSY backend not available")
    
    taylor.MultivariateTaylorFunction.initialize_mtf(max_order=MAX_ORDER, max_dimension=MAX_DIMENSION, implementation="cosy")
    taylor.MultivariateTaylorFunction.set_etol(ETOL)
    yield
    # Cleanup
    taylor.MultivariateTaylorFunction._INITIALIZED = False
    taylor.MultivariateTaylorFunction._MAX_ORDER = None
    taylor.MultivariateTaylorFunction._MAX_DIMENSION = None


def test_inverse_method():
    # A = 1 + x
    A = taylor.MultivariateTaylorFunction.var(1, dimension=MAX_DIMENSION) + 1.0
    
    # B = A.inverse() (Explicit method)
    B = A.inverse()
    
    # Check A * B approx 1
    Prod = A * B
    
    assert Prod.get_constant() == pytest.approx(1.0, abs=ETOL)
    # Ensure higher order terms are small/zero
    poly_coeffs = Prod.coeffs[1:] # Skip constant
    if len(poly_coeffs) > 0:
         assert np.all(np.abs(poly_coeffs) < 1e-10)

def test_truediv_inverse():
    # A = 1 + x
    A = taylor.MultivariateTaylorFunction.var(1, dimension=MAX_DIMENSION) + 1.0
    
    # B = 1 / A (Using __rtruediv__ -> division via backend)
    B = 1.0 / A
    
    # Check consistency with explicit inverse
    Inv = A.inverse()
    
    # Check difference
    Diff = B - Inv
    assert Diff.is_zero_mtf(Diff, zero_tolerance=1e-10)

def test_integer_power_positive():
    # A = 1 + x
    A = taylor.MultivariateTaylorFunction.var(1, dimension=MAX_DIMENSION) + 1.0
    
    # B = A^2
    B = A ** 2
    
    # Expected: 1 + 2x + x^2
    # Check coefficients
    # Constant
    assert B.get_constant() == pytest.approx(1.0)
    
    # x coeff (order 1, var 1)
    # We can reconstruct check: (1+x)*(1+x)
    A_sq_mul = A * A
    
    Diff = B - A_sq_mul
    assert Diff.is_zero_mtf(Diff, zero_tolerance=1e-10)

def test_integer_power_negative():
    # A = 1 + x
    A = taylor.MultivariateTaylorFunction.var(1, dimension=MAX_DIMENSION) + 1.0
    
    # B = A^-1
    B = A ** -1
    
    # Should match inverse
    Inv = A.inverse()
    
    Diff = B - Inv
    assert Diff.is_zero_mtf(Diff, zero_tolerance=1e-10)
    
    # B = A^-2
    C = A ** -2
    # Should be (A^-1)^2
    Inv_sq = Inv ** 2
    
    Diff2 = C - Inv_sq
    assert Diff2.is_zero_mtf(Diff2, zero_tolerance=1e-10)

def test_real_power_sqrt():
    # A = 1 + x
    A = taylor.MultivariateTaylorFunction.var(1, dimension=MAX_DIMENSION) + 1.0
    
    # B = A^0.5
    B = A ** 0.5
    
    # Should match A.sqrt() (which uses separate DAISRT wrapper, aka 1/sqrt, or DASQRT?
    # Backend sqrt uses compute_da_sqrt_. 
    # A**0.5 uses DAPKP.
    
    C = A.sqrt()
    
    Diff = B - C
    assert Diff.is_zero_mtf(Diff, zero_tolerance=1e-10)

def test_real_power_arbitrary():
    # A = 1 + x
    A = taylor.MultivariateTaylorFunction.var(1, dimension=MAX_DIMENSION) + 1.0
    
    # B = A^1.5
    B = A ** 1.5
    
    # Check B*B approx A^3
    B2 = B * B
    A3 = A ** 3
    
    Diff = B2 - A3
    assert Diff.is_zero_mtf(Diff, zero_tolerance=1e-9)

