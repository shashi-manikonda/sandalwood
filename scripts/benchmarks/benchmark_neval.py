
import time
import numpy as np
from sandalwood.taylor_function import MultivariateTaylorFunction as MTF

def benchmark_neval():
    # Force Python backend
    # Order 10, Dim 6 -> ~8000 terms
    ORDER = 10
    DIM = 6
    N_POINTS = 200_000

    print(f"Benchmarking Neval: Order={ORDER}, Dim={DIM}, N_Points={N_POINTS}")

    MTF._INITIALIZED = False
    MTF.initialize_mtf(ORDER, DIM, implementation="python")
    
    # Create a dense polynomial
    x = MTF.var(1)
    # Just sum variables to power
    poly = sum(MTF.var(i+1) for i in range(DIM))**ORDER
    # This might be too huge?
    # Dim 6, Order 10 -> C(16, 6) = 8008 terms.
    print(f"Terms: {len(poly.coeffs)}")
    
    points = np.random.rand(N_POINTS, DIM)
    
    start = time.time()
    res = poly.neval(points)
    end = time.time()
    
    print(f"Time: {end - start:.4f}s")
    throughput = N_POINTS / (end - start)
    print(f"Throughput: {throughput:,.0f} pts/s")

if __name__ == "__main__":
    benchmark_neval()
