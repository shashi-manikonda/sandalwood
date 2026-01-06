
import gc
import os
import time

import psutil

from sandalwood.backends.cosy.cosy_backend import (
    CosyBackend,
    CosyDA,
    CosyMtfData,
)


def print_memory(label):
    process = psutil.Process(os.getpid())
    print(f"[{label}] RSS: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def get_cosy_internals():
    # Helper to peek at IVAR/IMEM if wrapper exposes them (not yet)
    # Since we can't see them yet, we infer from RSS or failure
    pass

def benchmark_batch_access():
    print("\n--- Benchmarking Coefficient Access ---")
    order = 8
    dim = 4
    print(f"Initializing COSY (Order={order}, Dim={dim})...")
    CosyBackend.initialize(order, dim)
    
    # Create a polynomial with many terms
    mtf = CosyMtfData(dimension=dim)
    # (1+x)^order will have many terms
    # Create a sum of variables
    s = CosyMtfData(dimension=dim) # 0
    for i in range(dim):
        var = CosyMtfData(dimension=dim)
        # Manually set var to x_i
        # We can't easily do that with high level API without creating objects
        # accessing low level DA
        pass
    
    # Just use built-in ops which we know are safe-ish for now
    x1 = CosyMtfData(dimension=dim)
    # Set x1 to variable 1:
    # Hack: CosyMtfData init calls create_new=True -> constant 0.
    # We want var.
    # CosyDA(var_id=0) gives x_1
    x1.da = CosyDA(var_id=0) 
    
    # Create constant 1.0
    one = CosyMtfData(dimension=dim)
    one.da = CosyDA.from_const(1.0)
    poly = x1.add(one)
    
    for i in range(1, dim):
        xi = CosyMtfData(dimension=dim)
        xi.da = CosyDA(var_id=i)
        poly = poly.add(xi)
        
    print("Computing power (creating many terms)...")
    # (1 + x1 + ... + xn)^order
    # This might be huge.
    # Let's try power 6 first.
    large_poly = poly.exp() # Exp creates infinite series, truncated. Good.
    
    print("Getting all terms (current implementation)...")
    start = time.time()
    terms = large_poly.da.get_all_terms()
    end = time.time()
    print(f"Retrieved {len(terms)} terms in {end - start:.4f} seconds.")
    
    return end - start, len(terms)

def benchmark_memory_leak():
    print("\n--- Benchmarking Memory Leak ---")
    # Loop creation of objects
    
    print_memory("Start")
    for i in range(100):
        # Create temporary objects
        a = CosyMtfData(dimension=2)
        one = CosyMtfData(dimension=2)
        one.da = CosyDA.from_const(1.0)
        
        b = a.add(one)
        c = b.multiply(CosyMtfData(dimension=2)) # Just multiply by 0 to keep it simple but allocate
        # Let them go out of scope
    
    gc.collect()
    print_memory("After 100 iters")
    
    # Because of the wrapper.f leak (Stack vs Heap), we expect COSY internal memory to fill up.
    # We won't see RSS grow infinitely if COSY reuses the same buffer, but IVAR will grow.
    # Eventually COSY crashes or errors.
    
    print("Running 10,000 iters WITH Scope (expect stable memory)...")
    from sandalwood.backends.cosy.cosy_backend import CosyScope
    try:
        for i in range(10000):
            with CosyScope():
                a = CosyMtfData(dimension=2)
                one = CosyMtfData(dimension=2)
                one.da = CosyDA.from_const(1.0)
                b = a.add(one)
                # b goes out of scope here, CosyScope rewinds stack
            
            if i % 1000 == 0:
                print(f"Iter {i}...")
    except Exception as e:
        print(f"Crashed at iter {i}: {e}")
        return False
        
    print("Survived 10,000 iters. (Maybe memory is large enough?)")
    return True

if __name__ == "__main__":
    t, n = benchmark_batch_access()
    benchmark_memory_leak()
