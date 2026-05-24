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
    """Ensure accessing .exponents triggers conversion from dense to sparse.

    The original test guarded its assertions behind
    ``if hasattr(f, '_dense_coeffs') and f._dense_coeffs is not None``
    which was never True, making it a no-op. This version always asserts.
    """
    mtf._INITIALIZED = False
    mtf.initialize_mtf(max_order=2, max_dimension=2, implementation="python")
    x = mtf.var(1)

    # x*x must produce a valid MTF with one term: x^2
    f = x * x

    # Accessing .exponents triggers materialisation from any internal dense form
    assert f.exponents is not None, "f.exponents should not be None after x*x"
    assert len(f.exponents) >= 1, "Expected at least one term in x*x"

    # The x^2 term must exist and have the correct shape
    found_x2 = any(tuple(e) == (2, 0) for e in f.exponents)
    assert found_x2, "Coefficient for x^2 not found in x*x"


def test_kernel_etol_constant_exists():
    """_KERNEL_ETOL must be present in numba_kernels as a module constant.

    Guards the named tolerance constant introduced in feat/backend-hardening
    to document and centralise the early-exit threshold in dense_mul.
    """
    import sandalwood.numba_kernels as nk

    assert hasattr(nk, "_KERNEL_ETOL"), (
        "_KERNEL_ETOL constant is missing from sandalwood.numba_kernels"
    )
    assert nk._KERNEL_ETOL == 1e-16, (
        f"Expected _KERNEL_ETOL == 1e-16, got {nk._KERNEL_ETOL}"
    )


def test_compose_kernel_correctness():
    """compose_dense_kernel must produce correct coefficients for a 2-variable map.

    This is an integration-level guard for the compose_dense_kernel fix in
    feat/backend-hardening (dead prange loop removed, chunk_size restored).
    Verifies that composition of identity maps gives identity.
    """
    mtf._INITIALIZED = False
    mtf.initialize_mtf(max_order=4, max_dimension=2, implementation="python")

    x = mtf.var(1)
    y = mtf.var(2)

    # f(x,y) = x^2 + y composed with identity map {1: x, 2: y} = x^2 + y
    f = x**2 + y
    result = f.compose({1: x, 2: y})

    # Must equal the original
    diff = result - f
    assert diff.get_max_coefficient() < 1e-12, (
        "compose(identity) did not reproduce f; compose_dense_kernel may be broken"
    )
