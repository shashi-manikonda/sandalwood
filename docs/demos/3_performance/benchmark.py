import time

import numpy as np
import torch
from mtflib import mtf


def old_eval_loop(mtf_instance, points):
    """
    Evaluates the MTF at each point in a loop using the old eval logic.
    """
    results = np.empty(points.shape[0])
    for i, point in enumerate(points):
        term_values = np.prod(np.power(point, mtf_instance.exponents), axis=1)
        results[i] = np.einsum("j,j->", mtf_instance.coeffs, term_values)
    return results


def main():
    """
    Main function to run the performance demo.
    """
    # --- Setup ---
    print("Setting up the performance demo...")
    mtf.initialize_mtf(max_order=5, max_dimension=3)

    # Create a sample MTF
    x = mtf.var(1)
    y = mtf.var(2)
    z = mtf.var(3)
    mtf_instance = mtf.sin(x) + mtf.cos(y) + mtf.exp(z)

    # Generate a large number of random points
    n_points = 100_000
    dimension = 3
    points = np.random.rand(n_points, dimension)
    print(f"Generated {n_points} random points in {dimension} dimensions.")
    print("-" * 30)

    # --- Benchmark old eval loop ---
    print("Benchmarking the old eval method (looping)...")
    start_time = time.time()
    results_old = old_eval_loop(mtf_instance, points)
    end_time = time.time()
    time_old = end_time - start_time
    print(f"Time taken: {time_old:.4f} seconds")
    print("-" * 30)

    # --- Benchmark new neval method ---
    print("Benchmarking the new neval method (vectorized)...")
    start_time = time.time()
    results_new = mtf_instance.neval(points)
    end_time = time.time()
    time_new = end_time - start_time
    print(f"Time taken: {time_new:.4f} seconds")
    print("-" * 30)

    # --- Benchmark PyTorch neval method ---
    print("Benchmarking the new neval method with PyTorch tensors...")
    points_torch = torch.from_numpy(points).to(torch.float64)
    start_time = time.time()
    results_torch = mtf_instance.neval(points_torch)
    end_time = time.time()
    time_torch = end_time - start_time
    print(f"Time taken: {time_torch:.4f} seconds")
    print("-" * 30)

    # --- Comparison ---
    print("Performance Comparison:")
    if time_new > 0:
        speedup_numpy = time_old / time_new
        print(
            f"The numpy `neval` method is {speedup_numpy:.2f}x faster "
            "than the old `eval` loop."
        )
    else:
        print("Could not calculate numpy speedup (neval was too fast).")

    if time_torch > 0:
        speedup_torch = time_old / time_torch
        print(
            f"The torch `neval` method is {speedup_torch:.2f}x faster "
            "than the old `eval` loop."
        )
    else:
        print("Could not calculate torch speedup (neval was too fast).")

    # Verify that the results are consistent
    assert np.allclose(
        results_old, results_new
    ), "Results from numpy and old methods do not match."
    assert np.allclose(
        results_old, results_torch.numpy()
    ), "Results from torch and old methods do not match."
    print("\nResults from all methods are consistent.")


if __name__ == "__main__":
    main()
