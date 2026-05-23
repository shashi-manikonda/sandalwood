import time

from sandalwood.taylor_function import MultivariateTaylorFunction as MTF


def benchmark_dense():
    # Force Python backend
    # Order 12, Dim 3 -> ~455 terms
    ORDER = 8
    DIM = 4
    ITERS = 200

    print(f"Benchmarking Order={ORDER}, Dim={DIM}, Iters={ITERS}")

    # 1. Run without Dense Mode (simulate by setting table to None)
    print("\n--- Sparse Mode (Void View) ---")
    MTF._INITIALIZED = False
    MTF.initialize_mtf(ORDER, DIM, implementation="python")
    MTF._MULT_TABLE = None  # Disable table

    x = MTF.var(1)
    y = MTF.var(2)
    z = MTF.var(3)
    u = MTF.var(4)

    poly1 = (x + y + z + u) ** 6
    poly2 = (x - y + z - u) ** 6
    print(f"Terms: {len(poly1.coeffs)}")

    start = time.time()
    for _ in range(ITERS):
        res = poly1 * poly2
    end = time.time()
    print(f"Time: {end - start:.4f}s")
    sparse_time = end - start

    # 2. Run with Dense Mode
    print("\n--- Dense Mode ---")
    MTF._INITIALIZED = False
    MTF.initialize_mtf(ORDER, DIM, implementation="python")
    # Table is computed automatically

    x = MTF.var(1)
    y = MTF.var(2)
    z = MTF.var(3)
    u = MTF.var(4)

    poly1 = (x + y + z + u) ** 6
    poly2 = (x - y + z - u) ** 6

    start = time.time()
    for _ in range(ITERS):
        res = poly1 * poly2
    end = time.time()
    print(f"Time: {end - start:.4f}s")
    dense_time = end - start

    print(f"\nSpeedup: {sparse_time / dense_time:.2f}x")


if __name__ == "__main__":
    benchmark_dense()
