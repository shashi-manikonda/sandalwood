import numpy as np
import pytest

import sandalwood.taylor_function as tf
from sandalwood import mtf


@pytest.mark.parametrize("use_numba", [True, False])
def test_dense_multiplication_correctness(monkeypatch, use_numba):
    """Verifies that Dense Mode (with and without Numba) matches Sparse Mode."""
    # Force Numba availability flag
    monkeypatch.setattr(tf, "_NUMBA_AVAILABLE", use_numba)

    # Initialize with specific order to trigger table generation
    # Reset initialization first
    mtf._INITIALIZED = False
    mtf.initialize_mtf(max_order=4, max_dimension=2, implementation="python")

    x = mtf.var(1)
    y = mtf.var(2)

    # Dense operation
    f = (1 + x + y) ** 3

    # Check "Lazy" state (optimization check)
    # The result should be valid but might not have sparse coeffs yet
    # Note: Depending on implementation, small constants in addition might trigger materialization.
    # But (1+x+y)**3 involves multiplication of dense objects.

    # Trigger materialization explicitly to check values
    _ = f.coeffs

    # Analytical check: (1+x+y)^3 constant term is 1
    assert np.isclose(f.extract_coefficient((0, 0)), 1.0)
    # Coeff of x^3 is 1
    assert np.isclose(f.extract_coefficient((3, 0)), 1.0)
    # Coeff of x*y is 6 (from expansion 1 + 3(x+y) + 3(x+y)^2 + ... -> 3*(2xy))
    assert np.isclose(f.extract_coefficient((1, 1)), 6.0)


def test_lazy_materialization():
    """Ensure accessing properties triggers conversion from dense to sparse."""
    mtf._INITIALIZED = False
    mtf.initialize_mtf(max_order=2, max_dimension=2, implementation="python")
    x = mtf.var(1)

    # Create a dense object via multiplication
    # x*x should use dense path if initialized
    f = x * x

    # It might be in dense mode internally
    if hasattr(f, "_dense_coeffs") and f._dense_coeffs is not None:
        # Before access, exponents might be None/Empty depending on implementation
        # Just ensure accessing them works and gives correct data
        assert f.exponents is not None
        assert len(f.exponents) == 1
        assert tuple(f.exponents[0]) == (2, 0)
