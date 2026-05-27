import threading
from unittest.mock import patch

import numpy as np
import pytest

import sandalwood.taylor_function as tf
from sandalwood import mtf
from sandalwood.taylor_map import TaylorMap

_COSY_AVAILABLE = tf._COSY_BACKEND_AVAILABLE

@pytest.mark.skipif(not _COSY_AVAILABLE, reason="COSY backend not available")
def test_cosy_index_pool_thread_local_stress():
    """Stress test the thread-local pool to verify no lost/duplicate indices and no deadlocks."""
    from sandalwood.backends.cosy.cosy_backend import CosyIndexPool

    mtf._INITIALIZED = False
    mtf.initialize_mtf(max_order=3, max_dimension=2, implementation="cosy")

    N_THREADS = 16
    ACQUIRES_PER_THREAD = 100
    barrier = threading.Barrier(N_THREADS)
    errors = []

    def worker():
        try:
            barrier.wait()
            acquired_indices = []
            for _ in range(ACQUIRES_PER_THREAD):
                idx = CosyIndexPool.acquire(is_complex=False)
                acquired_indices.append(idx)
            
            # Release all of them
            for idx in acquired_indices:
                CosyIndexPool.release(idx, is_complex=False)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors in threads: {errors}"


def test_sub_dimensional_dense_composition(monkeypatch):
    """Verify that composition on sub-dimensional maps (new_dimension < max_dimension) uses dense mode and is correct."""
    from sandalwood import numba_kernels
    # Force Numba availability
    monkeypatch.setattr(numba_kernels, "_NUMBA_AVAILABLE", True)

    mtf._INITIALIZED = False
    # Max dimension is 3
    mtf.initialize_mtf(max_order=4, max_dimension=3, implementation="python")

    # Create maps with dimension 2 (sub-dimensional!)
    x = mtf.var(1, dimension=2)
    y = mtf.var(2, dimension=2)

    # component functions F(x, y) = [x + y^2, y - x^2]
    # dimension of components is 2
    f1 = x + y**2
    f2 = y - x**2
    F = TaylorMap([f1, f2])

    # G(u, v) = [u + v, u - v]
    u = mtf.var(1, dimension=2)
    v = mtf.var(2, dimension=2)
    g1 = u + v
    g2 = u - v
    G = TaylorMap([g1, g2])

    # We want to patch compose_dense_kernel to verify it is called
    from sandalwood import numba_kernels
    called = []
    original_compose_dense_kernel = numba_kernels.compose_dense_kernel

    def mock_compose_dense_kernel(*args, **kwargs):
        called.append(True)
        return original_compose_dense_kernel(*args, **kwargs)

    monkeypatch.setattr(numba_kernels, "compose_dense_kernel", mock_compose_dense_kernel)

    # Compose H = G(F)
    H = G.compose(F)

    # Verify that the dense kernel was called
    assert len(called) > 0, "compose_dense_kernel was not called for sub-dimensional composition"

    # Verify correctness of composition
    # component 1: g1(f1, f2) = f1 + f2 = x + y^2 + y - x^2 = x + y + y^2 - x^2
    # component 2: g2(f1, f2) = f1 - f2 = x + y^2 - y + x^2 = x - y + x^2 + y^2
    h1 = H.components[0]
    h2 = H.components[1]

    assert h1.dimension == 2
    assert h2.dimension == 2

    assert np.isclose(h1.extract_coefficient((1, 0)), 1.0) # x
    assert np.isclose(h1.extract_coefficient((0, 1)), 1.0) # y
    assert np.isclose(h1.extract_coefficient((2, 0)), -1.0) # -x^2
    assert np.isclose(h1.extract_coefficient((0, 2)), 1.0) # y^2

    assert np.isclose(h2.extract_coefficient((1, 0)), 1.0) # x
    assert np.isclose(h2.extract_coefficient((0, 1)), -1.0) # -y
    assert np.isclose(h2.extract_coefficient((2, 0)), 1.0) # x^2
    assert np.isclose(h2.extract_coefficient((0, 2)), 1.0) # y^2
