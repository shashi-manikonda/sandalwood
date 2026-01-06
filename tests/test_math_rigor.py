import math
import sys
import numpy as np
import pytest

from sandalwood.taylor_function import MultivariateTaylorFunction as mtf
from sandalwood.complex_taylor_function import ComplexMultivariateTaylorFunction as cmtf

# --- Fixtures & Setup ---

@pytest.fixture(params=["python", "cosy"])
def implementation(request):
    """Fixture to run tests against both Python and COSY backends."""
    impl = request.param
    impl = request.param
    mtf._INITIALIZED = False
    mtf._MAX_ORDER = None
    mtf._MAX_DIMENSION = None
    mtf._ETOL = 1e-16
    try:
        mtf.initialize_mtf(max_order=2, max_dimension=2, implementation=impl)
    except RuntimeError as e:
        if impl == "cosy" and "not initialized" in str(e):
             pytest.skip("COSY backend failed to initialize")
        raise e
    yield impl
    mtf._INITIALIZED = False

# --- 1. Mathematical Invariants ---

def test_arithmetic_commutativity(implementation):
    """
    Verify Commutativity: a + b == b + a and a * b == b * a
    """
    x = mtf.var(1)
    y = mtf.var(2)
    
    # Addition
    sum1 = x + y
    sum2 = y + x
    diff_sum = sum1 - sum2
    assert np.allclose(diff_sum.get_max_coefficient(), 0.0, atol=1e-15), "Addition not commutative"

    # Multiplication
    prod1 = x * y
    prod2 = y * x
    diff_prod = prod1 - prod2
    assert np.allclose(diff_prod.get_max_coefficient(), 0.0, atol=1e-15), "Multiplication not commutative"

def test_arithmetic_associativity(implementation):
    """
    Verify Associativity: (a + b) + c == a + (b + c)
    """
    x = mtf.var(1)
    y = mtf.var(2)
    z = mtf.from_constant(3.0) # Constant term
    
    # Addition
    lhs = (x + y) + z
    rhs = x + (y + z)
    diff = lhs - rhs
    assert np.allclose(diff.get_max_coefficient(), 0.0, atol=1e-15), "Addition not associative"

    # Multiplication
    lhs_mul = (x * y) * z
    rhs_mul = x * (y * z)
    diff_mul = lhs_mul - rhs_mul
    assert np.allclose(diff_mul.get_max_coefficient(), 0.0, atol=1e-15), "Multiplication not associative"

def test_arithmetic_distributivity(implementation):
    """
    Verify Distributivity: a * (b + c) == a * b + a * c
    """
    a = mtf.var(1)
    b = mtf.var(2)
    c = mtf.from_constant(2.0)
    
    lhs = a * (b + c)
    rhs = (a * b) + (a * c)
    diff = lhs - rhs
    assert np.allclose(diff.get_max_coefficient(), 0.0, atol=1e-15), "Distributivity arithmetic failed"

def test_calculus_fundamental_theorem(implementation):
    """
    Verify Fundamental Theorem of Calculus: int( d(f)/dx ) dx = f + C
    We check this for a polynomial f = x^3
    """
    x = mtf.var(1)
    f = x * x * x # x^3
    
    df_dx = f.deriv(1) # 3x^2
    int_df = df_dx.integrate(1) # x^3 + C (C=0 usually in this lib)
    
    # The integration constant is 0 by default implementation, so int_df should match f
    diff = int_df - f
    assert np.allclose(diff.get_max_coefficient(), 0.0, atol=1e-15), "Calculus identity failed"

def test_calculus_trig_derivatives(implementation):
    """
    Verify d(sin(x))/dx = cos(x) and d(cos(x))/dx = -sin(x)
    """
    x = mtf.var(1)
    try:
        s = x.sin()
        c = x.cos()
    except NotImplementedError:
        pytest.skip(f"Trig functions not implemented for {implementation}")

    # d/dx sin(x) should be cos(x)
    # TPSA Rule: deriv(order N) is accurate only to order N-1
    max_order = x.get_max_order()
    ds_dx = s.deriv(1)
    diff_s = ds_dx - c
    # Check that coefficients are zero up to max_order - 1
    assert np.allclose(diff_s.truncate(max_order - 1).get_max_coefficient(), 0.0, atol=1e-12), "d(sin)/dx != cos (truncated)"
    
    # d/dx cos(x) should be -sin(x)
    dc_dx = c.deriv(1)
    diff_c = dc_dx + s 
    assert np.allclose(diff_c.truncate(max_order - 1).get_max_coefficient(), 0.0, atol=1e-12), "d(cos)/dx != -sin (truncated)"


# --- 2. Edge Case Matrix ---

@pytest.mark.parametrize("zero_val", [0, 0.0, -0.0])
def test_edge_zeros(implementation, zero_val):
    """Test handling of different zero representations."""
    x = mtf.var(1)
    res = x + zero_val
    assert np.allclose(res.extract_coefficient((0,0)), 0.0)
    assert np.allclose(res.extract_coefficient((1,0)), 1.0)
    
    res_mul = x * zero_val
    assert np.allclose(res_mul.get_max_coefficient(), 0.0)

@pytest.mark.parametrize("epsilon", [1e-12, -1e-12, 1e-15])
def test_edge_epsilons(implementation, epsilon):
    """Test very small numbers."""
    x = mtf.var(1)
    res = x + epsilon
    assert np.allclose(res.extract_coefficient((0,0)), epsilon, atol=1e-16)

@pytest.mark.parametrize("large_val", [1e15, -1e15, 1e20])
def test_edge_large_magnitude(implementation, large_val):
    """Test handling of large magnitude coefficients."""
    x = mtf.var(1)
    res = x * large_val
    coeff = res.extract_coefficient((1,0))
    # Python handles large floats. COSY might have limits but 1e20 is usually fine for double.
    assert np.isclose(coeff, large_val, rtol=1e-10)

def test_edge_singularities(implementation):
    """
    Test behavior with inf and NaN. 
    Python backend should handle or propagate. 
    COSY backend handling depends on F77/C interaction, usually best to avoid crashing.
    """
    if implementation == "cosy":
        pytest.skip("COSY backend behavior with NaN/Inf is undefined/unsafe")

    x = mtf.var(1)
    
    # NaN propagation
    nan_val = float('nan')
    res_nan = x + nan_val
    assert np.isnan(res_nan.extract_coefficient((0,0)))
    
    # Inf propagation
    inf_val = float('inf')
    res_inf = x + inf_val
    assert np.isinf(res_inf.extract_coefficient((0,0)))

def test_domain_error_log_zero(implementation):
    """Test log(0) raises ValueError or RuntimeError."""
    zero_func = mtf.from_constant(0.0)
    with pytest.raises((ValueError, RuntimeError)):
        zero_func.log()

def test_domain_error_sqrt_negative(implementation):
    """Test sqrt(-1) raises ValueError for Real MTF."""
    neg_func = mtf.from_constant(-1.0)
    with pytest.raises((ValueError, RuntimeError)):
        neg_func.sqrt()

def test_domain_error_division_zero(implementation):
    """Test 1/0 raises ZeroDivisionError or similar."""
    one = mtf.from_constant(1.0)
    zero = mtf.from_constant(0.0)
    with pytest.raises((ZeroDivisionError, ValueError, RuntimeError)):
        # Depending on backend: Python might raise ZeroDivisionError, 
        # COSY might raise RuntimeError or return Inf (if safe).
        # We accept standard error types.
        res = one / zero

# --- 3. Type Safety ---

def test_type_safety_invalid_inputs(implementation):
    """Ensure proper errors for invalid types."""
    x = mtf.var(1)
    
    with pytest.raises(TypeError):
        x + "string"
        
    with pytest.raises(TypeError):
        x * None

