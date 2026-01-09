import os
import sys

import numpy as np
import pytest

# Ensure we can import sandalwood
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from sandalwood import MultivariateTaylorFunction as MTF

try:
    from sandalwood.backends.cosy import cosy_backend

    _COSY_AVAILABLE = True
except ImportError:
    _COSY_AVAILABLE = False


@pytest.fixture(scope="module")
def setup_cosy():
    if not _COSY_AVAILABLE:
        pytest.skip("COSY backend not available")
    try:
        MTF.initialize_mtf(max_order=4, max_dimension=2, implementation="cosy")
    except RuntimeError:
        pass  # Already initialized
    yield
    # Cleanup if needed? Global state persists.


@pytest.mark.skipif(not _COSY_AVAILABLE, reason="COSY backend required")
def test_batch_add(setup_cosy):
    """Test vectorized addition: vec + vec."""
    x = MTF.var(1)
    y = MTF.var(2)

    vec_a = np.array([x + i for i in range(10)])
    vec_b = np.array([y + i * 0.1 for i in range(10)])

    # Vectorized
    res_vec = vec_a + vec_b

    # Loop
    res_loop = np.array([a + b for a, b in zip(vec_a, vec_b)])

    assert isinstance(res_vec, np.ndarray)
    assert len(res_vec) == 10

    # Check coefficients
    for i in range(10):
        # res[i] should be (x + i) + (y + 0.1*i) = x + y + 1.1*i
        r = res_vec[i]
        l = res_loop[i]

        # Compare raw coefficients
        # Or compare eval
        pt = [0.1, 0.2]
        val_r = r.eval(pt)
        val_l = l.eval(pt)
        assert np.isclose(val_r, val_l, atol=1e-12)

        # Use extract
        c_x = r.extract_coefficient((1, 0))
        c_y = r.extract_coefficient((0, 1))
        c_0 = r.extract_coefficient((0, 0))

        assert np.isclose(c_x, 1.0)
        assert np.isclose(c_y, 1.0)
        assert np.isclose(c_0, i * 1.1)


@pytest.mark.skipif(not _COSY_AVAILABLE, reason="COSY backend required")
def test_batch_broadcasting(setup_cosy):
    """Test vectorized addition: vec + scalar."""
    x = MTF.var(1)
    vec = np.array([x + i for i in range(5)])
    scalar = 10.0

    # Scalar + Vec
    res = scalar + vec

    for i in range(5):
        # res[i] = 10 + x + i
        r = res[i]
        assert np.isclose(r.extract_coefficient((0, 0)), 10.0 + i)
        assert np.isclose(r.extract_coefficient((1, 0)), 1.0)

    # Vec + Scalar
    res2 = vec + scalar
    for i in range(5):
        r = res2[i]
        assert np.isclose(r.extract_coefficient((0, 0)), 10.0 + i)


@pytest.mark.skipif(not _COSY_AVAILABLE, reason="COSY backend required")
def test_batch_mul(setup_cosy):
    """Test vectorized multiplication."""
    x = MTF.var(1)
    vec_a = np.array([x + i for i in range(5)])
    vec_b = np.array([x - i for i in range(5)])

    # (x + i)*(x - i) = x^2 - i^2
    res = vec_a * vec_b

    for i in range(5):
        r = res[i]
        c_x2 = r.extract_coefficient((2, 0))
        c_0 = r.extract_coefficient((0, 0))
        c_x = r.extract_coefficient((1, 0))  # should be 0

        assert np.isclose(c_x2, 1.0)
        assert np.isclose(c_0, -(i**2))
        assert np.isclose(c_x, 0.0)
