import math

import numpy as np
import pytest

import sandalwood.taylor_function as taylor


def safe_initialize(implementation):
    """Robust initialization for tests."""
    try:
        taylor.MultivariateTaylorFunction.initialize_mtf(
            max_order=2, max_dimension=2, implementation=implementation
        )
    except RuntimeError:
        pass


@pytest.mark.parametrize("implementation", ["python", "cosy"])
def test_inv_sqrt(implementation):
    """Test inv_sqrt(x) = 1/sqrt(x)"""
    safe_initialize(implementation)

    # x = 4 + delta_x1
    x = 4.0 + taylor.MultivariateTaylorFunction.var(1)

    res = x.inv_sqrt()

    # Expected constant part: 1/sqrt(4) = 0.5
    val = res.eval([0, 0])[0]
    assert np.isclose(val, 0.5), f"Expected 0.5, got {val}"

    # Expected derivative for 1/sqrt(x) at 4: -0.5 * x^(-1.5) = -0.5 * (1/8) = -0.0625
    df = res.deriv(1)
    val_deriv = df.eval([0, 0])[0]
    assert np.isclose(val_deriv, -0.0625), f"Expected -0.0625, got {val_deriv}"


@pytest.mark.parametrize("implementation", ["python", "cosy"])
def test_inv_cbrt(implementation):
    """Test inv_cbrt (currently unimplemented)"""
    safe_initialize(implementation)
    x = 8.0 + taylor.MultivariateTaylorFunction.var(1)

    with pytest.raises(NotImplementedError):
        x.inv_cbrt()


@pytest.mark.parametrize("implementation", ["python", "cosy"])
def test_inv_pow_3_2(implementation):
    """Test inv_pow_3_2(x) = 1/x^1.5"""
    safe_initialize(implementation)

    if implementation == "cosy" and not taylor._COSY_BACKEND_AVAILABLE:
        pytest.skip("COSY backend not available")

    x = 4.0 + taylor.MultivariateTaylorFunction.var(1)

    if implementation == "python":
        with pytest.raises(NotImplementedError):
            x.inv_pow_3_2()
        return

    res = x.inv_pow_3_2()

    # Expected constant: 1/4^1.5 = 1/8 = 0.125
    val = res.eval([0, 0])[0]
    assert np.isclose(val, 0.125), f"Expected 0.125, got {val}"

    # Expected derivative: d/dx x^-1.5 = -1.5 * x^-2.5
    # At x=4: -1.5 * 4^-2.5 = -1.5 * (1/32) = -1.5 * 0.03125 = -0.046875
    df = res.deriv(1)
    val_deriv = df.eval([0, 0])[0]
    assert np.isclose(val_deriv, -0.046875), f"Expected -0.046875, got {val_deriv}"


@pytest.mark.parametrize("implementation", ["python", "cosy"])
def test_erf(implementation):
    """Test erf(x)"""
    safe_initialize(implementation)

    if implementation == "cosy" and not taylor._COSY_BACKEND_AVAILABLE:
        pytest.skip("COSY backend not available")

    x = 0.0 + taylor.MultivariateTaylorFunction.var(1)

    res = x.erf()

    # erf(0) = 0
    val = res.eval([0, 0])[0]
    assert np.isclose(val, 0.0), f"Expected 0.0, got {val}"

    # d/dx erf(x) = 2/sqrt(pi) * exp(-x^2)
    # At x=0, derivative is 2/sqrt(pi) approx 1.128379
    df = res.deriv(1)
    val_deriv = df.eval([0, 0])[0]
    expected_deriv = 2.0 / math.sqrt(math.pi)
    assert np.isclose(
        val_deriv, expected_deriv
    ), f"Expected {expected_deriv}, got {val_deriv}"


@pytest.mark.parametrize("implementation", ["python", "cosy"])
def test_coth(implementation):
    """Test coth(x)"""
    safe_initialize(implementation)

    if implementation == "cosy" and not taylor._COSY_BACKEND_AVAILABLE:
        pytest.skip("COSY backend not available")

    # coth(x) = 1/tanh(x). Need non-zero constant for tanh?
    # No, coth(x) blows up at x=0. Use x = 1 + delta
    x = 1.0 + taylor.MultivariateTaylorFunction.var(1)

    res = x.coth()

    # Expected constant: coth(1) = 1/tanh(1)
    val = res.eval([0, 0])[0]
    expected = 1.0 / math.tanh(1.0)
    assert np.isclose(val, expected), f"Expected {expected}, got {val}"


@pytest.mark.parametrize("implementation", ["python", "cosy"])
def test_estimate_stability(implementation):
    """Test estimate_stability"""
    safe_initialize(implementation)

    if implementation == "cosy" and not taylor._COSY_BACKEND_AVAILABLE:
        pytest.skip("COSY backend not available")

    x = taylor.MultivariateTaylorFunction.var(1)
    f = x * x

    if implementation == "python":
        with pytest.raises(
            NotImplementedError,
            match="estimate_stability is only available for the COSY backend",
        ):
            f.estimate_stability()
        return

    # COSY estimate_stability
    est = f.estimate_stability()
    assert isinstance(est, float)
    # We just check it runs and returns a float.
    # Detailed numerical validation of DAEST is beyond this quick check.
