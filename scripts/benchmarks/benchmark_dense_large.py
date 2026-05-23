import time

from sandalwood.taylor_function import MultivariateTaylorFunction as MTF


def benchmark_dense_large():
    # Order 10, Dim 6 -> ~8000 terms
    ORDER = 10
    DIM = 6
    ITERS = 50

    print(f"Benchmarking Large Dense Mul: Order={ORDER}, Dim={DIM}, Iters={ITERS}")

    MTF._INITIALIZED = False
    MTF.initialize_mtf(ORDER, DIM, implementation="python")

    x = MTF.var(1)
    # Sum variables
    poly1 = sum(MTF.var(i + 1) for i in range(DIM)) ** 5
    poly2 = sum(MTF.var(i + 1) for i in range(DIM)) ** 5

    # Force materialization to ensure we have dense indices cached
    _ = poly1.coeffs
    _ = poly2.coeffs

    print(f"Terms: {len(poly1.coeffs)}")

    start = time.time()
    for _ in range(ITERS):
        res = poly1 * poly2
    end = time.time()

    print(f"Time: {end - start:.4f}s")
    print(f"Avg Time: {(end - start) / ITERS * 1000:.2f} ms")


if __name__ == "__main__":
    benchmark_dense_large()
