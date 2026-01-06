import argparse
import json
import os
import sys
import time

import numpy as np

# Ensure we can import sandalwood
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sandalwood import mtf


def run_benchmark(backend, operation, dims, order, iterations, use_complex):
    try:
        mtf_impl = "cosy" if backend == "cosy" else "python"
        mtf.initialize_mtf(max_order=order, max_dimension=dims, implementation=mtf_impl)
            
        # Setup variables
        vars = [mtf.var(i+1) for i in range(dims)]
        
        # Pre-compute some dense polynomials for testing
        poly1 = vars[0]
        poly2 = vars[0]
        # To verify Phase 2 (Complex DA), we mix in complex constants if requested
        if use_complex:
             poly1 = poly1 + 1j
             poly2 = poly2 - 0.5j

        for i in range(1, dims):
            poly1 = poly1 + vars[i]
            poly2 = poly2 - vars[i]
            
        # Batching for statistics
        num_batches = 5
        batch_size = max(1, iterations // num_batches)
        
        times = []

        for _ in range(num_batches):
            # Start timing
            start_time = time.perf_counter()

            for _ in range(batch_size):
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
                elif operation == "cot":
                    res = (poly1 + 0.1).cot()
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
                elif operation == "coth":
                    res = (poly1 + 0.1).coth()
                elif operation == "erf":
                    res = poly1.erf()
                elif operation == "derivative":
                    res = poly1.derivative(1)
                elif operation == "integrate":
                    res = poly1.integrate(1)

            end_time = time.perf_counter()
            times.append((end_time - start_time) / batch_size)

        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        result = {
            "backend": backend,
            "mean": avg_time,
            "std": std_time,
            "min": min_time,
            "max": max_time,
            "iterations": iterations,
            "batch_size": batch_size,
            "complex": use_complex
        }

        print(f"RESULT_JSON: {json.dumps(result)}")
        
    except Exception as e:
        # If it's the specific COSY error, just exit with code 1, caller handles it.
        # But we print something to stderr for debugging
        sys.stderr.write(f"Error in benchmark_mtf: {e}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["python", "cosy"])
    parser.add_argument("--op", required=True, choices=["add", "sub", "mul", "div", "pow", "eval", "sin", "cos", "tan", "cot", "exp", "log", "sqrt", "asin", "acos", "atan", "sinh", "cosh", "tanh", "coth", "erf", "derivative", "integrate"])
    parser.add_argument("--dims", type=int, default=4)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--complex", action="store_true", help="Use complex polynomials")
    
    args = parser.parse_args()
    run_benchmark(args.backend, args.op, args.dims, args.order, args.iters, args.complex)
