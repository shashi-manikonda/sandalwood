import pytest
import numpy as np
import math
from sandalwood import MultivariateTaylorFunction as mtf

def safe_initialize(order, dim, implementation="cosy"):
    try:
        mtf.initialize_mtf(max_order=order, max_dimension=dim, implementation=implementation)
    except RuntimeError:
        # Fallback: if already initialized with different settings, reset it for these tests
        mtf._INITIALIZED = False
        mtf.initialize_mtf(max_order=order, max_dimension=dim, implementation=implementation)

@pytest.fixture(autouse=True)
def setup_mtf():
    safe_initialize(6, 2)
    yield

def test_erf():
    x = mtf.var(1)
    
    # Test at zero (exact)
    f_erf = x.erf()
    assert abs(f_erf.get_constant()) < 1e-15
    
    # Test derivatives
    # d/dx erf(x) = (2/sqrt(pi)) * exp(-x^2)
    df_dx = f_erf.deriv(1)
    exp_neg_x2 = (-(x**2)).exp()
    expected_df_dx = (2.0 / math.sqrt(math.pi)) * exp_neg_x2
    
    # Compare coefficients up to a reasonable order
    for exp_tuple in df_dx.to_dict()['exponents']:
        if sum(exp_tuple) <= 6:
            c1 = df_dx.extract_coefficient(tuple(exp_tuple))
            c2 = expected_df_dx.extract_coefficient(tuple(exp_tuple))
            assert np.allclose(c1, c2, atol=1e-12)

def test_cot_coth():
    x = mtf.var(1) + 1.0 # Offset from zero to avoid singularity
    
    # cot(x) = 1/tan(x)
    f_cot = x.cot()
    f_tan = x.tan()
    res_cot = f_cot * f_tan
    assert np.allclose(res_cot.get_constant(), 1.0, atol=1e-12)
    # High order terms should be zero
    poly = res_cot.get_polynomial_part()
    if poly.coeffs.size > 0:
        assert np.max(np.abs(poly.coeffs)) < 1e-10

    # coth(x) = 1/tanh(x)
    f_coth = x.coth()
    f_tanh = x.tanh()
    res_coth = f_coth * f_tanh
    assert np.allclose(res_coth.get_constant(), 1.0, atol=1e-12)
    poly_h = res_coth.get_polynomial_part()
    if poly_h.coeffs.size > 0:
        assert np.max(np.abs(poly_h.coeffs)) < 1e-10

def test_norm():
    x = mtf.var(1)
    y = mtf.var(2)
    f = 1.0 + 2.0*x - 3.0*y**2
    
    n = f.norm()
    assert np.allclose(n, 3.0, atol=1e-15)

def test_inverse_hyperbolics_fallback():
    # Since we use Python fallback, verify they work as expected
    x = mtf.var(1)
    
    # arcsinh(x)
    f = x.arcsinh()
    assert abs(f.get_constant()) < 1e-15
    # arcsinh(sinh(x)) == x
    g = x.sinh().arcsinh()
    assert np.allclose(g.extract_coefficient((1, 0)), 1.0, atol=1e-12)
    poly = g.get_polynomial_part()
    # Check other coefficients are small (up to order 5 due to truncation errors in series)
    for exp_tuple in poly.to_dict()['exponents']:
        if sum(exp_tuple) > 1 and sum(exp_tuple) <= 5:
            assert abs(poly.extract_coefficient(tuple(exp_tuple))) < 1e-10

def test_backend_consistency():
    # Run tests with both backends
    for backend in ["python", "cosy"]:
        safe_initialize(6, 1, implementation=backend)
        
        x = mtf.var(1)
        
        # Test erf
        f1 = x.erf()
        c0 = f1.get_constant()
        c1 = f1.extract_coefficient((1,))
        assert np.allclose(c0, 0.0, atol=1e-15)
        assert np.allclose(c1, 2.0 / math.sqrt(math.pi), atol=1e-12)
        
        # Test cot (offset)
        x_off = x + 0.5
        f2 = x_off.cot()
        c0_cot = f2.get_constant()
        expected_c0 = 1.0 / math.tan(0.5)
        assert np.allclose(c0_cot, expected_c0, atol=1e-12)

if __name__ == "__main__":
    pytest.main([__file__])
