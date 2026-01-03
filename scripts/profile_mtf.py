import cProfile
import io
import pstats
import time

from mtflib import TaylorMap, mtf


def profile_operations():
    # Setup
    order = 10
    dim = 4
    mtf.initialize_mtf(max_order=order, max_dimension=dim)

    x = mtf.var(1)
    y = mtf.var(2)
    z = mtf.var(3)
    u = mtf.var(4)

    print(f"Profiling Order {order}, Dim {dim}")

    # 1. Arithmetic Intensive
    start = time.perf_counter()
    _ = (x + y + z + u) ** 3 * mtf.exp(x)
    end = time.perf_counter()
    print(f"Complex Arithmetic: {end - start:.4f}s")

    # 2. Composition (Heavy)
    # Create a simple map
    m1 = TaylorMap([x + 0.1 * y * y, y + 0.1 * x, z, u])
    m2 = TaylorMap([x, y + 0.1 * z * z, z + 0.1 * y, u])

    start = time.perf_counter()
    _ = m1.compose(m2)
    end = time.perf_counter()
    print(f"Map Composition (4D): {end - start:.4f}s")


def run_profiler():
    pr = cProfile.Profile()
    pr.enable()
    profile_operations()
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())


if __name__ == "__main__":
    run_profiler()
