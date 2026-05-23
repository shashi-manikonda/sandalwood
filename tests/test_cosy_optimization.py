import os
import sys

import numpy as np
import pytest

# Ensure we can import sandalwood
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sandalwood.backends.cosy.cosy_backend import CosyBackend, CosyDA, CosyMtfData
from sandalwood.taylor_function import _COSY_BACKEND_AVAILABLE


@pytest.mark.skipif(
    sys.platform == "win32", reason="Batch eval optimization primarily for Linux/OpenMP"
)
def test_batch_eval_correctness():
    """
    Verifies that the batch evaluation (EVAL_DA_BATCH) produces identical results
    to the single-point evaluation loop.
    """
    if not _COSY_BACKEND_AVAILABLE:
        pytest.skip("COSY backend not available")

    try:
        CosyBackend.initialize(order=5, dim=2)
    except RuntimeError:
        pass  # Already initialized

    # specific coefficient setup
    # f(x, y) = 1.0 + 2.0*x + 3.0*y + 0.5*x^2 + 0.5*y^2 + x*y

    da = CosyMtfData(dimension=2)
    da.da = CosyDA.from_const(1.0)

    x = CosyMtfData(dimension=2)
    x.da = CosyDA(var_id=0)  # x

    y = CosyMtfData(dimension=2)
    y.da = CosyDA(var_id=1)  # y

    # Construct polynomial
    # Construct polynomial
    def make_const(val):
        d = CosyMtfData(dimension=2)
        d.da = CosyDA.from_const(val)
        return d

    poly = (
        da.add(x.multiply(make_const(2.0)))
        .add(y.multiply(make_const(3.0)))
        .add(x.multiply(x).multiply(make_const(0.5)))
        .add(y.multiply(y).multiply(make_const(0.5)))
        .add(x.multiply(y))
    )

    # Generate random points
    np.random.seed(42)
    n_points = 100
    points = np.random.rand(n_points, 2)

    # 1. Python Loop Evaluation (Gold Standard for Correctness, albeit slow)
    expected = []
    for p in points:
        val = poly.eval(p)
        expected.append(val)
    expected = np.array(expected)

    # 2. Batch Evaluation
    actual = poly.eval(points)

    # Check shape
    assert actual.shape == (n_points,)

    # Check values
    np.testing.assert_allclose(
        actual, expected, rtol=1e-12, atol=1e-12, err_msg="Batch evaluation mismatch"
    )

    print("Batch evaluation correctness verified!")


if __name__ == "__main__":
    test_batch_eval_correctness()
