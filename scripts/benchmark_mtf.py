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
            mtf.initialize_mtf(max_order=order, max_dimension=dims, implementation="python")
            
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
            elif operation == "sub":
                res = poly1 - poly2
            elif operation == "div":
                res = poly1 / (poly2 + 10.0) # Avoid division by zero
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
            elif operation == "tan":
                res = poly1.tan()
            elif operation == "exp":
                res = poly1.exp()
            elif operation == "log":
                res = (poly1 + 10.0).log()
            elif operation == "sqrt":
                res = (poly1 + 10.0).sqrt()
            elif operation == "asin":
                res = (poly1 * 0.1).asin()
            elif operation == "acos":
                res = (poly1 * 0.1).acos()
            elif operation == "atan":
                res = poly1.atan()
            elif operation == "sinh":
                res = poly1.sinh()
            elif operation == "cosh":
                res = poly1.cosh()
            elif operation == "tanh":
                res = poly1.tanh()
            elif operation == "derivative":
                res = poly1.derivative(1)
            elif operation == "integrate":
                res = poly1.integrate(1)
                
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations
        print(f"RESULT: {backend}: {avg_time:.6f} s/iter")
        
    except Exception as e:
        print(f"Error: {e}")
        # traceback
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["python", "cosy"])
    parser.add_argument("--op", required=True, choices=["add", "sub", "mul", "div", "pow", "eval", "sin", "cos", "tan", "exp", "log", "sqrt", "asin", "acos", "atan", "sinh", "cosh", "tanh", "derivative", "integrate"])
    parser.add_argument("--dims", type=int, default=4)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    
    args = parser.parse_args()
    run_benchmark(args.backend, args.op, args.dims, args.order, args.iters)
