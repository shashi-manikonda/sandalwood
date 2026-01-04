import sys
import os
import time
import argparse
import numpy as np

# Ensure we can import sandalwood
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sandalwood import mtf

def run_benchmark(backend, operation, dims, order, iterations):
    try:
        if backend == "cosy":
            mtf.initialize_mtf(max_order=order, max_dimension=dims, implementation="cosy")
        else:
            mtf.initialize_mtf(max_order=order, max_dimension=dims, implementation="cpp") # Default python/cpp
            
        # Setup variables
        vars = [mtf.var(i+1) for i in range(dims)]
        
        # Pre-compute some dense polynomials for testing
        poly1 = vars[0]
        poly2 = vars[0]
        for i in range(1, dims):
            poly1 = poly1 + vars[i]
            poly2 = poly2 - vars[i]
            
        # Start timing
        start_time = time.time()
        
        for _ in range(iterations):
            if operation == "add":
                res = poly1 + poly2
            elif operation == "mul":
                res = poly1 * poly2
            elif operation == "pow":
                # Create dense polynomial by high power
                # Power that fits within max_order: e.g. (order - 1)
                p = max(1, order - 1)
                res = poly1 ** p
            elif operation == "eval":
                res = poly1.eval([0.5] * dims)
            elif operation == "sin":
                res = poly1.sin()
            elif operation == "cos":
                res = poly1.cos()
            elif operation == "exp":
                res = poly1.exp()
                
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations
        print(f"RESULT: {backend}: {avg_time:.6f} s/iter")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["python", "cosy"])
    parser.add_argument("--op", required=True, choices=["add", "mul", "pow", "eval", "sin", "cos", "exp"])
    parser.add_argument("--dims", type=int, default=4)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    
    args = parser.parse_args()
    run_benchmark(args.backend, args.op, args.dims, args.order, args.iters)
