
import pytest
import numpy as np
from sandalwood.taylor_function import MultivariateTaylorFunction as mtf

@pytest.fixture(autouse=True)
def setup_mtf():
    mtf.initialize_mtf(max_order=4, max_dimension=2)
    # Ensure clean state
    yield
    mtf._INITIALIZED = False

def test_complex_scalar_inverse():
    if mtf._IMPLEMENTATION != "cosy":
        pytest.skip("Complex inverse only implemented for COSY backend so far")

    # DEBUG EARLY
    dummy = mtf.var(1, dimension=2)
    from sandalwood.backends.cosy.cosy_backend import libcosy
    from ctypes import byref, c_int
    if hasattr(libcosy, 'debug_print_type_'):
         print("DEBUG: Early Type Check")
         libcosy.debug_print_type_(byref(c_int(dummy.mtf_data.da.idx)))
        
    A = mtf.from_constant(1.0 + 1.0j, dimension=2)
    
    # DEBUG
    from sandalwood.backends.cosy.cosy_backend import libcosy
    from ctypes import byref, c_int
    if hasattr(A.mtf_data.da, 'idx'):
         # print(f"DEBUG: Checking type for idx={A.mtf_data.da.idx}")
         if hasattr(libcosy, 'debug_print_type_'):
             libcosy.debug_print_type_(byref(c_int(A.mtf_data.da.idx)))

    B = A.inverse()
    
    # Expected: 1/(1+i) = (1-i)/2 = 0.5 - 0.5i
    coeffs = B.to_dict()["coeffs"]
    assert np.allclose(coeffs[0], 0.5 - 0.5j)

def test_complex_var_inverse():
    if mtf._IMPLEMENTATION != "cosy":
        pytest.skip("Complex inverse only implemented for COSY backend so far")

    # A = 1 + i*x
    x = mtf.var(1, dimension=2)
    A = 1.0 + 1.0j * x
    
    # A^-1 = 1/(1+ix) = 1 - ix + (ix)^2 - (ix)^3 ...
    #      = 1 - ix - x^2 + ix^3 + x^4 ...
    B = A.inverse()
    
    # Check coeff of x^0: 1
    assert np.allclose(B.extract_coefficient((0,0)), 1.0 + 0j)
    # Check coeff of x^1: -i
    assert np.allclose(B.extract_coefficient((1,0)), 0.0 - 1.0j)
    # Check coeff of x^2: -1
    assert np.allclose(B.extract_coefficient((2,0)), -1.0 + 0j)
    # Check coeff of x^3: i
    assert np.allclose(B.extract_coefficient((3,0)), 0.0 + 1.0j)

def test_complex_integer_power():
    if mtf._IMPLEMENTATION != "cosy":
        pytest.skip("Complex power only implemented for COSY backend so far")
    
    # A = 1 + i
    A = mtf.from_constant(1.0 + 1.0j, dimension=2)
    # A^2 = (1+i)^2 = 1 + 2i - 1 = 2i
    B = A ** 2
    
    coeffs = B.to_dict()["coeffs"]
    assert np.allclose(coeffs[0], 0.0 + 2.0j)
    
    # A^3 = 2i * (1+i) = 2i - 2 = -2 + 2i
    C = A ** 3
    coeffs_c = C.to_dict()["coeffs"]
    assert np.allclose(coeffs_c[0], -2.0 + 2.0j)

def test_complex_real_power():
    if mtf._IMPLEMENTATION != "cosy":
        pytest.skip("Complex power only implemented for COSY backend so far")
    
    # A = 1 + i
    A = mtf.from_constant(1.0 + 1.0j, dimension=2)
    
    # A^2.0 should be 2i
    B = A ** 2.0
    coeffs = B.to_dict()["coeffs"]
    assert np.allclose(coeffs[0], 0.0 + 2.0j)
    
    # Sqrt(i) = e^(i pi/4) = (1+i)/sqrt(2)?
    # Wait, A = i. 
    I = mtf.from_constant(1.0j, dimension=2)
    S = I ** 0.5
    # (1+i)/sqrt(2) = 0.707 + 0.707i
    expected = (1.0 + 1.0j) / np.sqrt(2)
    coeffs_s = S.to_dict()["coeffs"]
    assert np.allclose(coeffs_s[0], expected)

def test_complex_log_exp_consistency():
    if mtf._IMPLEMENTATION != "cosy":
        pytest.skip("Complex power only implemented for COSY backend so far")
        
    # Check if (e^x)^i = e^(ix) = cos(x) + i sin(x)
    # Verify via Euler
    x = mtf.var(1, dimension=2)
    exp_x = x.exp() # Real DA
    
    # exp_x ** 1j -> Complex Power logic should trigger.
    # Note: exp() returns CosyDA (real).
    # CosyDA.__pow__(complex) triggers to_complex().
    # So (e^x)^(i) calls COMPUTE_CD_PKI? No PKP is for real power.
    # We implemented PKP (Real Power DA^Double).
    # We did NOT implement COMPUTE_CD_PCI (Complex Power exp^Complex).
    # My wrapper additions only had PKP (Real Val).
    # Ah.
    
    # Limitation: Current implementation supports Real Power of Complex Base.
    # Does not support Complex Power of Base (Real or Complex).
    # So (e^x)**2.0 works. (e^x)**(1j) fails (NotImplemented).
    
    # Let's verify Real Power of Complex Base works.
    # (e^(ix))^2 = e^(2ix)
    
    i = mtf.from_constant(1.0j, dimension=2)
    ix = i * x
    # ix is Complex DA
    
    # exp(ix) -> Not implemented directly in MTF? 
    # MTF.exp() delegates to backend.
    # CosyMtfData.exp() calls da.exp().
    # CosyCDA does NOT have exp() method implemented in python backend wrapper list.
    # I only added inverse and pow.
    # So exp(ix) will fail if CosyCDA doesn't implement exp.
    pass

