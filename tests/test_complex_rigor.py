import cmath

import pytest

from sandalwood import ComplexMultivariateTaylorFunction as CMTF
from sandalwood import MultivariateTaylorFunction as MTF


def safe_initialize(order, dim, backend):
    try:
        MTF.initialize_mtf(max_order=order, max_dimension=dim, implementation=backend)
    except RuntimeError:
        # Already initialized, that's fine for these tests
        pass


@pytest.mark.parametrize("backend", ["python", "cosy"])
def test_complex_arithmetic_parity(backend):
    order = 4
    dim = 2
    safe_initialize(order, dim, backend)

    z1 = CMTF.var(1, dim) * (1 + 2j) + (3 - 4j)
    z2 = CMTF.var(2, dim) * (0.5 + 0.5j) + (1 + 1j)

    # Addition
    res_add = z1 + z2
    c0 = res_add.get_constant()
    assert cmath.isclose(c0, (3 - 4j) + (1 + 1j))

    # Multiplication
    res_mul = z1 * z2
    c0 = res_mul.get_constant()
    assert cmath.isclose(c0, (3 - 4j) * (1 + 1j))

    # Division
    res_div = z1 / z2
    c0 = res_div.get_constant()
    assert cmath.isclose(c0, (3 - 4j) / (1 + 1j))

    # Power
    res_pow = z1**2
    c0 = res_pow.get_constant()
    assert cmath.isclose(c0, (3 - 4j) ** 2)


@pytest.mark.parametrize("backend", ["python", "cosy"])
def test_euler_identity(backend):
    order = 5
    dim = 1
    safe_initialize(order, dim, backend)

    x = CMTF.var(1, dim)
    ix = x * 1j

    e_ix = ix.exp()
    cos_x = x.cos()
    sin_x = x.sin()

    res = e_ix - (cos_x + sin_x * 1j)

    # Check coefficients
    for exp, coeff in res.get_all_terms():
        assert abs(coeff) < 1e-12


@pytest.mark.parametrize("backend", ["python", "cosy"])
def test_hyperbolic_identities(backend):
    order = 4
    dim = 1
    safe_initialize(order, dim, backend)

    z = CMTF.var(1, dim) * (1 + 1j) + (0.5 - 0.2j)

    # cosh^2(z) - sinh^2(z) = 1
    res = z.cosh() ** 2 - z.sinh() ** 2 - 1.0
    for exp, coeff in res.get_all_terms():
        assert abs(coeff) < 1e-12

    # sin(iz) = i sinh(z)
    res = (z * 1j).sin() - (z.sinh() * 1j)
    for exp, coeff in res.get_all_terms():
        assert abs(coeff) < 1e-12


@pytest.mark.parametrize("backend", ["python", "cosy"])
def test_complex_log_sqrt_identities(backend):
    order = 3
    dim = 1
    safe_initialize(order, dim, backend)

    z = CMTF.var(1, dim) * (0.1 + 0.1j) + (1.0 + 1j)

    # exp(log(z)) = z
    res = z.log().exp() - z
    for exp, coeff in res.get_all_terms():
        assert abs(coeff) < 1e-10

    # sqrt(z)^2 = z
    res = z.sqrt() ** 2 - z
    for exp, coeff in res.get_all_terms():
        assert abs(coeff) < 1e-10


@pytest.mark.parametrize("backend", ["python", "cosy"])
def test_complex_tan_tanh(backend):
    order = 4
    dim = 1
    safe_initialize(order, dim, backend)

    z = CMTF.var(1, dim) * (0.2 + 0.3j) + (0.1 - 0.1j)

    # tan(z) = sin(z)/cos(z)
    res = z.tan() - z.sin() / z.cos()
    for exp, coeff in res.get_all_terms():
        assert abs(coeff) < 1e-12

    # tanh(z) = sinh(z)/cosh(z)
    res = z.tanh() - z.sinh() / z.cosh()
    for exp, coeff in res.get_all_terms():
        assert abs(coeff) < 1e-12


@pytest.mark.parametrize("backend", ["python", "cosy"])
def test_complex_inverse(backend):
    order = 4
    dim = 1
    safe_initialize(order, dim, backend)

    z = CMTF.var(1, dim) * (0.5 + 0.5j) + (2.0 - 1j)

    # z * (1/z) = 1
    res = z * z.inverse() - 1.0
    for exp, coeff in res.get_all_terms():
        assert abs(coeff) < 1e-12


@pytest.mark.parametrize("backend", ["python", "cosy"])
def test_complex_differentiation(backend):
    order = 4
    dim = 2
    safe_initialize(order, dim, backend)

    z = CMTF.var(1, dim) * (1j) + CMTF.var(2, dim)

    # f(z) = exp(z) -> f'(z) = exp(z)
    f = z.exp()
    df_dz1 = f.deriv(1)
    df_dz2 = f.deriv(2)

    # df/dz1 = d/dz1 exp(i x1 + x2) = i * exp(i x1 + x2)
    # df/dz2 = d/dz2 exp(i x1 + x2) = 1 * exp(i x1 + x2)

    res1 = df_dz1 - f * 1j
    res2 = df_dz2 - f

    # Derivative of truncated series T_N(f) matches T_N(f') only up to order N-1.
    # The term of order N in T_N(f') corresponds to derivatives of terms of order N+1 in f,
    # which are not present in T_N(f).

    for exp, coeff in res1.get_all_terms():
        if sum(exp) < order:
            assert abs(coeff) < 1e-12
    for exp, coeff in res2.get_all_terms():
        if sum(exp) < order:
            assert abs(coeff) < 1e-12
