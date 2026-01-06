
import os
import sys
import time

import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from sandalwood.backends.cosy.cosy_backend import CosyBackend, CosyDA, CosyMtfData


def benchmark_batch_eval():
    print("Initializing COSY backend...")
    try:
        CosyBackend.initialize(order=10, dim=3)
    except RuntimeError:
        pass
        
    print("Constructing polynomial (Order 10, Dim 3)...")
    # Create a dense polynomial
    da = CosyMtfData(dimension=3)
    da.da = CosyDA.from_const(1.0)
    vars = [CosyMtfData(dimension=3) for i in range(3)]
    for i in range(3):
        vars[i].da = CosyDA(var_id=i)
        
    # Add some high order terms
    term = vars[0].add(vars[1]).add(vars[2])
    poly = term
    for i in range(9): # Power 10
        poly = poly.multiply(term)
        
    # Generate points
    n_points = 100000
    print(f"Generating {n_points} random points...")
    points = np.random.rand(n_points, 3)
    
    print("\n--- Starting Benchmark ---")
    
    # 1. Serial Evaluation (Sample)
    # Only run for a subset to estimate time, as full 100k would be too slow
    n_sample = 1000
    print(f"Running Serial Evaluation on {n_sample} points...")
    start_time = time.time()
    for i in range(n_sample):
        poly.eval(points[i])
    end_time = time.time()
    serial_time_ms = (end_time - start_time) * 1000
    serial_per_point = serial_time_ms / n_sample
    print(f"Serial Time: {serial_time_ms:.2f} ms")
    print(f"Serial Per Point: {serial_per_point:.4f} ms")
    estimated_serial_total = serial_per_point * n_points / 1000.0 # seconds
    print(f"Estimated Serial Total for {n_points}: {estimated_serial_total:.2f} s")
    
    # 2. Batch Evaluation
    print(f"\nRunning Batch Evaluation on {n_points} points...")
    start_time = time.time()
    poly.eval(points)
    end_time = time.time()
    batch_time_s = end_time - start_time
    print(f"Batch Time: {batch_time_s:.4f} s")
    print(f"Batch Per Point: {(batch_time_s * 1000 / n_points):.6f} ms")
    
    # Speedup
    speedup = estimated_serial_total / batch_time_s
    print(f"\nSpeedup Factor: {speedup:.2f}x")

if __name__ == "__main__":
    benchmark_batch_eval()
