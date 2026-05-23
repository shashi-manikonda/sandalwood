import numpy as np
import pytest
import sandalwood.taylor_function as taylor


@pytest.fixture(autouse=True)
def cleanup_mtf():
    """Reset MTF initialization after each test."""
    yield
    taylor.MultivariateTaylorFunction._INITIALIZED = False


def safe_initialize(implementation):
    """Robust initialization for tests."""
    try:
        taylor.MultivariateTaylorFunction.initialize_mtf(
            max_order=2, max_dimension=2, implementation=implementation
        )
    except RuntimeError:
        pass


@pytest.mark.parametrize("implementation", ["python", "cosy"])
def test_derivative_simple(implementation):
    """Test d(x^2)/dx = 2x"""
    safe_initialize(implementation)
    x = taylor.MultivariateTaylorFunction.var(1)
    f = x * x  # x^2
    df = f.deriv(1)  # df/dx

    # Expect 2*x
    val = df.eval([3.0, 0.0])[0]
    expected = 2 * 3.0
    assert np.isclose(val, expected), f"Expected {expected}, got {val}"


@pytest.mark.parametrize("implementation", ["python", "cosy"])
def test_integration_simple(implementation):
    """Test int(x) dx = x^2/2"""
    safe_initialize(implementation)
    x = taylor.MultivariateTaylorFunction.var(1)
    f = x
    int_f = f.integrate(1)

    # Expect x^2/2
    val = int_f.eval([2.0, 0.0])[0]
    expected = (2.0**2) / 2.0
    assert np.isclose(val, expected), f"Expected {expected}, got {val}"


@pytest.mark.parametrize("implementation", ["python", "cosy"])
def test_poisson_bracket(implementation):
    """Test [q, p] = 1 where q=x1, p=x2"""
    safe_initialize(implementation)

    if implementation == "cosy" and not taylor._COSY_BACKEND_AVAILABLE:
        pytest.skip("COSY backend not available")

    q = taylor.MultivariateTaylorFunction.var(1)
    p = taylor.MultivariateTaylorFunction.var(2)

    if implementation == "python":
        with pytest.raises(NotImplementedError):
            q.poisson_bracket(p)
        return

    # PB(q, p) = dq/dq * dp/dp - dq/dp * dp/dq
    #          = 1 * 1 - 0 * 0 = 1

    pb = q.poisson_bracket(p)

    val = pb.eval([0, 0])[0]  # Constant 1
    assert np.isclose(val, 1.0), f"Expected 1.0, got {val}"

    # Test [p, q] = -1
    pb_rev = p.poisson_bracket(q)
    val_rev = pb_rev.eval([0, 0])[0]
    assert np.isclose(val_rev, -1.0), f"Expected -1.0, got {val_rev}"


@pytest.mark.parametrize("implementation", ["python", "cosy"])
def test_mixed_derivative(implementation):
    """Test d(x*y)/dx = y"""
    safe_initialize(implementation)
    x = taylor.MultivariateTaylorFunction.var(1)
    y = taylor.MultivariateTaylorFunction.var(2)

    f = x * y
    df_dx = f.deriv(1)

    # df/dx should be y
    # Eval at (x=2, y=3) -> expected 3
    val = df_dx.eval([2.0, 3.0])[0]
    assert np.isclose(val, 3.0), f"Expected 3.0, got {val}"
