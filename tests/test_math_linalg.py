import pytest
import numpy as np
from sandalwood.backends.cosy import cosy_backend

@pytest.mark.skipif(not cosy_backend.CosyBackend.is_initialized(), reason="COSY backend not initialized")
def test_linear_combination():
    """Test da_lin_comb: res = 2*DA1 + 3*DA2"""
    # Setup
    cosy_backend.CosyBackend.initialize(order=1, dim=1)
    
    # Create x and constant 1
    da_x = cosy_backend.CosyDA(var_id=0) # x1
    da_c = cosy_backend.CosyDA.from_const(1.0)
    
    # Compute 2*x + 3*1 using lin_comb
    res_da = cosy_backend.da_lin_comb(da_x, 2.0, da_c, 3.0) # 2x + 3
    
    # Wrap in MTF to eval? Or assume CosyDA works?
    # CosyDA is low level. Use CosyMtfData to eval?
    # Or implement simple eval for testing
    
    # We can use CosyMtfData wrapper for easy eval
    mtf_data = cosy_backend.CosyMtfData(1)
    mtf_data.da = res_da
    
    val = mtf_data.eval([2.0])[0] # 2(2) + 3 = 7
    assert np.isclose(val, 7.0), f"Expected 7.0, got {val}"

@pytest.mark.skipif(not cosy_backend.CosyBackend.is_initialized(), reason="COSY backend not initialized")
def test_matrix_inversion():
    """Test inversion of a 2x2 matrix using da_mat_inv"""
    # Matrix A = [[4, 7], [2, 6]]
    # Det = 24 - 14 = 10
    # Inv = 1/10 * [[6, -7], [-2, 4]] = [[0.6, -0.7], [-0.2, 0.4]]
    
    matrix = [4.0, 7.0, 2.0, 6.0]
    n = 2
    
    inv_matrix = cosy_backend.da_mat_inv(matrix, n)
    
    expected = [0.6, -0.7, -0.2, 0.4]
    
    assert np.allclose(inv_matrix, expected), f"Expected {expected}, got {inv_matrix}"
