
import sandalwood.taylor_function
from sandalwood.taylor_function import MultivariateTaylorFunction as MTF
import numpy as np

def test_cosy_basics():
    print(f"File: {sandalwood.taylor_function.__file__}")
    print(f"COSY Available: {sandalwood.taylor_function._COSY_BACKEND_AVAILABLE}")
    
    print("Initializing MTF with COSY backend...")
    MTF.initialize_mtf(max_order=2, max_dimension=2, implementation="cosy")
    
    print("Creating variables...")
    x = MTF.var(1)
    y = MTF.var(2)
    
    print("Performing arithmetic...")
    f = x + y
    g = x * y
    
    print("Result of x + y:")
    print(f.exponents)
    print(f.coeffs)
    
    # Expected: (1,0)->1, (0,1)->1
    expected_coeffs_f = { (1,0): 1.0, (0,1): 1.0 }
    
    # Extract dict from f
    f_dict = f.mtf_data.to_dict()
    f_exps = f_dict['exponents']
    f_vals = f_dict['coeffs']
    
    # Verify
    for i in range(len(f_vals)):
        exp = tuple(f_exps[i])
        val = f_vals[i]
        if abs(val) > 1e-10:
            print(f"Term: {exp}, Coeff: {val}")
            
    print("Result of x * y:")
    # Expected: (1,1)->1
    g_dict = g.mtf_data.to_dict()
    g_exps = g_dict['exponents']
    g_vals = g_dict['coeffs']
    
    for i in range(len(g_vals)):
        exp = tuple(g_exps[i])
        val = g_vals[i]
        if abs(val) > 1e-10:
            print(f"Term: {exp}, Coeff: {val}")

    print("COSY backend verification complete.")

if __name__ == "__main__":
    test_cosy_basics()
