
import time
import numpy as np
from sandalwood.taylor_function import MultivariateTaylorFunction
from sandalwood.taylor_map import TaylorMap
import sandalwood.taylor_map

def run_benchmark(dim, order, n_repeats=5):
    print(f"\n--- Benchmarking Composition (Dim={dim}, Order={order}) ---")
    
    # Setup Maps
    # We use 'python' impl to ensure we are testing the Python-side logic (accelerated vs not)
    # rather than COSY backend.
    MultivariateTaylorFunction._INITIALIZED = False
    MultivariateTaylorFunction.initialize_mtf(max_order=order, max_dimension=dim, implementation="python")
    
    # Create random maps
    # F: R^n -> R^n
    # G: R^n -> R^n
    
    # We construct them manually to ensure they are "full" enough to stress test
    vars = [MultivariateTaylorFunction.var(i+1, dim) for i in range(dim)]
    
    # simple dense polynomials
    components_f = []
    components_g = []
    
    for d in range(dim):
        poly_f = MultivariateTaylorFunction.from_constant(0.0, dim)
        poly_g = MultivariateTaylorFunction.from_constant(0.0, dim)
        
        # Add some terms
        for v in vars:
            poly_f = poly_f + v + 0.5*v**2
            poly_g = poly_g + v - 0.5*v**2
            
        components_f.append(poly_f)
        components_g.append(poly_g)
        
    F = TaylorMap(components_f)
    G = TaylorMap(components_g)
    
    # warmup
    F.compose(G)
    
    # --- Measure Numba (Dense) ---
    # Ensure dense mode is active
    if MultivariateTaylorFunction._MULT_TABLE is None:
        print("WARNING: Dense tables not initialized!")
        
    start_time = time.time()
    for _ in range(n_repeats):
        F.compose(G)
    end_time = time.time()
    numba_time = (end_time - start_time) / n_repeats
    print(f"Numba (Dense) Average Time: {numba_time:.6f} s")
    
    # --- Measure Sparse (Python Fallback) ---
    # Force disable dense mode by temporarily hiding the table
    real_table = MultivariateTaylorFunction._MULT_TABLE
    MultivariateTaylorFunction._MULT_TABLE = None
    
    # Verify it's using sparse (sanity check: accessing internal cache or something? 
    # Just trust the switch based on my implementation knowledge)
    
    start_time = time.time()
    for _ in range(n_repeats):
        F.compose(G)
    end_time = time.time()
    sparse_time = (end_time - start_time) / n_repeats
    print(f"Python (Sparse) Average Time: {sparse_time:.6f} s")
    
    # Restore table
    MultivariateTaylorFunction._MULT_TABLE = real_table
    
    speedup = sparse_time / numba_time if numba_time > 0 else 0.0
    print(f"Speedup: {speedup:.2f}x")
    return speedup

if __name__ == "__main__":
    # Test cases: (dim, order)
    cases = [
        (2, 4),
        (2, 8),
        (3, 4),
        (3, 6), # High load
    ]
    
    for d, o in cases:
        run_benchmark(d, o)
