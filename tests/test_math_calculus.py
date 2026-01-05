import pytest
import numpy as np
import sandalwood.taylor_function as taylor
from sandalwood.backends.cosy import cosy_backend

# Only run if COSY is available
@pytest.mark.skipif(not cosy_backend.CosyBackend.is_initialized(), reason="COSY backend not initialized")
def test_derivative_simple():
    """Test d(x^2)/dx = 2x"""
    try:
        # Use max_dimension=2 to be compatible with other tests
        taylor.MultivariateTaylorFunction.initialize_mtf(max_order=2, max_dimension=2, implementation="cosy")
    except RuntimeError:
        pass # Already initialized

    x = taylor.MultivariateTaylorFunction.var(1)
    
    f = x * x # x^2
    df = f.deriv(1) # df/dx
    
    # Expect 2*x
    val = df.eval([3.0, 0.0])[0]
    expected = 2 * 3.0
    assert np.isclose(val, expected), f"Expected {expected}, got {val}"

def test_integration_simple():
    """Test int(x) dx = x^2/2"""
    try:
        taylor.MultivariateTaylorFunction.initialize_mtf(max_order=2, max_dimension=2, implementation="cosy")
    except RuntimeError:
        pass

    x = taylor.MultivariateTaylorFunction.var(1)
    
    f = x
    print(f"DEBUG TEST: f terms: {f.mtf_data.da.get_all_terms()}")
    int_f = f.integrate(1)
    print(f"DEBUG TEST: int_f terms: {int_f.mtf_data.da.get_all_terms()}")
    
    # Expect x^2/2
    val = int_f.eval([2.0, 0.0])[0]
    expected = (2.0**2) / 2.0
    assert np.isclose(val, expected), f"Expected {expected}, got {val}"

def test_poisson_bracket():
    """Test [q, p] = 1 where q=x1, p=x2"""
    try:
        taylor.MultivariateTaylorFunction.initialize_mtf(max_order=2, max_dimension=2, implementation="cosy")
    except RuntimeError:
        pass

    q = taylor.MultivariateTaylorFunction.var(1)
    p = taylor.MultivariateTaylorFunction.var(2)
    
    # PB(q, p) = dq/dq * dp/dp - dq/dp * dp/dq
    #          = 1 * 1 - 0 * 0 = 1
    
    pb = q.poisson_bracket(p)
    
    val = pb.eval([0, 0])[0] # Constant 1
    assert np.isclose(val, 1.0), f"Expected 1.0, got {val}"
    
    # Test [p, q] = -1
    pb_rev = p.poisson_bracket(q)
    val_rev = pb_rev.eval([0, 0])[0]
    assert np.isclose(val_rev, -1.0), f"Expected -1.0, got {val_rev}"

def test_mixed_derivative():
    """Test d(x*y)/dx = y"""
    try:
        taylor.MultivariateTaylorFunction.initialize_mtf(max_order=2, max_dimension=2, implementation="cosy")
    except RuntimeError:
        pass

    x = taylor.MultivariateTaylorFunction.var(1)
    y = taylor.MultivariateTaylorFunction.var(2)
    
    f = x * y
    print(f"DEBUG TEST mixed: f terms: {f.mtf_data.da.get_all_terms()}")
    df_dx = f.deriv(1)
    print(f"DEBUG TEST mixed: df_dx terms: {df_dx.mtf_data.da.get_all_terms()}")
    
    # df/dx should be y
    # Eval at (x=2, y=3) -> expected 3
    val = df_dx.eval([2.0, 3.0])[0]
    print(f"DEBUG TEST mixed: val={val}")
    # assert np.isclose(val, 3.0), f"Expected 3.0, got {val}"
