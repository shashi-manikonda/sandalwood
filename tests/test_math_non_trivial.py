import math

import numpy as np
import pytest

from sandalwood import mtf


@pytest.fixture(autouse=True)
def implementation(request):
    """Fixture to run tests against both Python and COSY backends."""
    # Parameterize is handled at function level, but we use this to setup
    pass


def safe_init(impl, order=4, dim=3):
    mtf._INITIALIZED = False
    mtf._MAX_ORDER = None
    mtf._MAX_DIMENSION = None
    mtf._ETOL = 1e-16
    try:
        mtf.initialize_mtf(max_order=order, max_dimension=dim, implementation=impl)
    except RuntimeError:
        pass

    if mtf._IMPLEMENTATION != impl:
        pytest.skip(
            f"Implementation {impl} not available, fell back to {mtf._IMPLEMENTATION}"
        )


@pytest.mark.parametrize("implementation", ["python", "cosy"])
def test_multivariate_polynomial_arithmetic(implementation):
    """
    Test (1 + x + y^2 + 3xy) * (x - 2y + z)
    = x - 2y + z + x^2 - 2xy + xz + xy^2 - 2y^3 + y^2z + 3x^2y - 6xy^2 + 3xyz
    = z - 2y + x + x^2 + xz - 2xy + 3x^2y - 5xy^2 - 2y^3 + y^2z + 3xyz
    """
    safe_init(implementation, order=4, dim=3)
    x = mtf.var(1)
    y = mtf.var(2)
    z = mtf.var(3)

    p1 = 1.0 + x + y**2 + 3.0 * x * y
    p2 = x - 2.0 * y + z

    res = p1 * p2

    # Evaluate at a non-trivial point
    pt = np.array([0.5, -0.3, 0.1])
    val = res.eval(pt)[0]

    # Expected manually:
    # p1(pt) = 1 + 0.5 + (-0.3)^2 + 3*(0.5)*(-0.3)
    #        = 1 + 0.5 + 0.09 - 0.45 = 1.14
    # p2(pt) = 0.5 - 2*(-0.3) + 0.1 = 0.5 + 0.6 + 0.1 = 1.2
    # res(pt) = 1.14 * 1.2 = 1.368

    assert np.isclose(val, 1.368, atol=1e-12)

    # Check specific high-order coefficient: 3x^2y (exponents (2, 1, 0))
    coeff_3x2y = res.extract_coefficient((2, 1, 0))
    assert np.isclose(coeff_3x2y, 3.0, atol=1e-12)

    # Check coefficient of xy^2: 1*(-2y) from y^2 and 3xy*(-2y) => -2y^3 - 6xy^2?
    # Wait: y^2 * (-2y) = -2y^3
    # 3xy * (-2y) = -6xy^2
    # x * (-2y) = -2xy
    # y^2 * x = xy^2
    # Total xy^2: -6 + 1 = -5
    coeff_xy2 = res.extract_coefficient((1, 2, 0))
    assert np.isclose(coeff_xy2, -5.0, atol=1e-12)


@pytest.mark.parametrize("implementation", ["python", "cosy"])
def test_composition_chain_non_trivial(implementation):
    """
    Test f(x, y) = exp(sin(x + y))
    """
    safe_init(implementation, order=5, dim=2)
    x = mtf.var(1)
    y = mtf.var(2)

    # Composition sin(x+y)
    s = (x + y).sin()
    # Composition exp(s)
    f = s.exp()

    # Evaluate at a very small point to minimize truncation error
    pt = np.array([0.01, 0.01])
    val = f.eval(pt)[0]

    expected = math.exp(math.sin(0.02))
    # Approximation error is O(pt^order)
    assert np.isclose(val, expected, atol=1e-10)


@pytest.mark.parametrize("implementation", ["python", "cosy"])
def test_high_order_calculus_identities(implementation):
    """
    Test mixed partial derivatives: d2(x^2 y^3)/dxdy = 6xy^2
    And triple derivative: d3(x^3 y^2)/dx^2dy = 6x*2y = 12xy
    """
    safe_init(implementation, order=5, dim=2)
    x = mtf.var(1)
    y = mtf.var(2)

    f = (x**3) * (y**2)
    # d/dy f = 2 x^3 y
    # d/dx (d/dy f) = 6 x^2 y
    # d/dx (6 x^2 y) = 12 x y

    d3f = f.deriv(1).deriv(1).deriv(2)

    pt = np.array([2.0, 3.0])
    val = d3f.eval(pt)[0]
    expected = 12.0 * 2.0 * 3.0  # 72
    assert np.isclose(val, 72.0, atol=1e-12)


@pytest.mark.parametrize("implementation", ["cosy"])
def test_advanced_poisson_bracket(implementation):
    """
    Test [q^2, p^2] = 4qp
    """
    safe_init(implementation, order=4, dim=2)
    q = mtf.var(1)
    p = mtf.var(2)

    f = q**2
    g = p**2

    pb = f.poisson_bracket(g)

    # Expect 4 * q * p
    pt = np.array([1.5, 2.0])
    val = pb.eval(pt)[0]
    expected = 4.0 * 1.5 * 2.0  # 12.0
    assert np.isclose(val, 12.0, atol=1e-12)

    # Check coefficient of qp (1, 1)
    assert np.isclose(pb.extract_coefficient((1, 1)), 4.0, atol=1e-12)


@pytest.mark.parametrize("implementation", ["cosy"])
def test_da_weighted_matrix_inversion(implementation):
    """
    Test inversion property: M * M^-1 = I
    """
    safe_init(implementation, order=4, dim=2)
    x = mtf.var(1)
    y = mtf.var(2)

    # M = [[ 1+x, y ],
    #      [ y, 1-x ]]
    a, b, c, d = 1.0 + x, y, y, 1.0 - x
    det = a * d - b * c  # 1 - x^2 - y^2

    # Manual inverse components
    m_inv_00 = d / det
    m_inv_01 = -b / det
    m_inv_10 = -c / det
    m_inv_11 = a / det

    # Check I = M * M_inv
    i_00 = a * m_inv_00 + b * m_inv_10
    i_01 = a * m_inv_01 + b * m_inv_11

    # i_00 should be exactly 1 up to truncation!
    # a*m_inv_00 + b*m_inv_10 = (1+x)(1-x)/det + y(-y)/det = (1-x^2 - y^2)/det = 1

    max_order = x.get_max_order()
    # Check identity property (I_00 should be 1.0)
    assert np.isclose(i_00.extract_coefficient((0, 0)), 1.0, atol=1e-12)
    # Check higher coefficients are zero
    diff_i00 = i_00 - 1.0
    assert np.allclose(
        diff_i00.truncate(max_order).get_max_coefficient(), 0.0, atol=1e-12
    )

    # Check off-diagonal is zero
    assert np.isclose(i_01.truncate(max_order).get_max_coefficient(), 0.0, atol=1e-12)


@pytest.mark.parametrize("implementation", ["python", "cosy"])
def test_non_trivial_integration(implementation):
    """
    Test int(x*y^2 + sin(x)) dx = 0.5*x^2*y^2 - cos(x) + C
    We verify by differentiating the result to get back the integrand.
    """
    safe_init(implementation, order=5, dim=2)
    x = mtf.var(1)
    y = mtf.var(2)

    integrand = x * (y**2) + x.sin()

    # Integrate wrt x (dim 1)
    integral = integrand.integrate(1)

    # Derivative wrt x should match integrand (up to truncation)
    deriv = integral.deriv(1)

    # Compare deriv vs integrand
    diff = deriv - integrand

    max_order = x.get_max_order()
    # TPSA rule: Integral increases accuracy order, Derivative decreases.
    # So deriv(int(f)) should be accurate to order(f).
    # However x.sin() is truncated.
    # Let's check coefficients are zero.
    assert np.allclose(
        diff.truncate(max_order - 1).get_max_coefficient(), 0.0, atol=1e-12
    )


@pytest.mark.parametrize("implementation", ["python", "cosy"])
def test_advanced_special_functions(implementation):
    """
    Test composition chains:
    1. log(sqrt(1+x)) vs 0.5 * log(1+x)
    2. tan(asin(x)) vs x/sqrt(1-x^2)
    """
    safe_init(implementation, order=6, dim=1)
    x = mtf.var(1)

    # 1. log(sqrt(1+x)) = log((1+x)^0.5) = 0.5 * log(1+x)
    f1 = (1.0 + x).sqrt().log()
    f2 = 0.5 * (1.0 + x).log()

    diff1 = f1 - f2
    # Should be essentially zero
    assert np.allclose(diff1.truncate(5).get_max_coefficient(), 0.0, atol=1e-12)

    # 2. tan(asin(x))
    # Domain check: asin(x) defined for |x| < 1.
    # Use small x.
    if implementation == "cosy":
        # tan/asin might be supported in COSY
        t = x.asin().tan()

        # Analytic: x / sqrt(1-x^2)
        # = x * (1 - x^2)^(-0.5)
        analytic = x * (1.0 - x**2) ** (-0.5)

        diff2 = t - analytic
        # Evaluate at small point
        val = diff2.eval([0.1])[0]
        assert np.isclose(val, 0.0, atol=1e-10)


@pytest.mark.parametrize("implementation", ["python", "cosy"])
def test_complex_arithmetic_non_trivial(implementation):
    """
    Test (1+i)x * (1-i)x = (1 - i^2)x^2 = 2x^2
    """
    safe_init(implementation, order=4, dim=1)

    # We need to manually construct complex MTFs if not supported by var directly?
    # var(1) is proper Real MTF.
    x = mtf.var(1)

    # If implementation allows complex scalars:
    c1 = 1.0 + 1j
    c2 = 1.0 - 1j

    if implementation == "cosy":
        # Check if complex scalars work in COSY backend wrapper
        # Currently cosymtfdata tries to handle it.
        pass

    p1 = x * c1  # (1+i)x
    p2 = x * c2  # (1-i)x

    res = p1 * p2

    # Should be 2*x^2 + 0j
    # Coefficient of x^2 should be 2.0

    coeff = res.extract_coefficient((2,))
    assert np.isclose(coeff.real, 2.0, atol=1e-12)
    assert np.isclose(coeff.imag, 0.0, atol=1e-12)
