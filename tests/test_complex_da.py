import numpy as np
import pytest

from sandalwood.taylor_function import MultivariateTaylorFunction as mtf


@pytest.fixture(autouse=True)
def setup_mtf():
    mtf.initialize_mtf(max_order=4, max_dimension=2)
    yield
    mtf._INITIALIZED = False


def test_complex_scalar_inverse():
    if mtf._IMPLEMENTATION != "cosy":
        pytest.skip("Complex inverse only implemented for COSY backend so far")

    A = mtf.from_constant(1.0 + 1.0j, dimension=2)
    B = A.inverse()

    # Expected: 1/(1+i) = (1-i)/2 = 0.5 - 0.5i
    coeffs = B.to_dict()["coeffs"]
    assert np.allclose(coeffs[0], 0.5 - 0.5j)


def test_complex_var_inverse():
    if mtf._IMPLEMENTATION != "cosy":
        pytest.skip("Complex inverse only implemented for COSY backend so far")

    # A = 1 + i*x
    x = mtf.var(1, dimension=2)
    A = 1.0 + 1.0j * x

    # A^-1 = 1/(1+ix) = 1 - ix + (ix)^2 - (ix)^3 ...
    #      = 1 - ix - x^2 + ix^3 + x^4 ...
    B = A.inverse()

    # Check coeff of x^0: 1
    assert np.allclose(B.extract_coefficient((0, 0)), 1.0 + 0j)
    # Check coeff of x^1: -i
    assert np.allclose(B.extract_coefficient((1, 0)), 0.0 - 1.0j)
    # Check coeff of x^2: -1
    assert np.allclose(B.extract_coefficient((2, 0)), -1.0 + 0j)
    # Check coeff of x^3: i
    assert np.allclose(B.extract_coefficient((3, 0)), 0.0 + 1.0j)


def test_complex_integer_power():
    if mtf._IMPLEMENTATION != "cosy":
        pytest.skip("Complex power only implemented for COSY backend so far")

    # A = 1 + i
    A = mtf.from_constant(1.0 + 1.0j, dimension=2)
    # A^2 = (1+i)^2 = 1 + 2i - 1 = 2i
    B = A**2

    coeffs = B.to_dict()["coeffs"]
    assert np.allclose(coeffs[0], 0.0 + 2.0j)

    # A^3 = 2i * (1+i) = 2i - 2 = -2 + 2i
    C = A**3
    coeffs_c = C.to_dict()["coeffs"]
    assert np.allclose(coeffs_c[0], -2.0 + 2.0j)


def test_complex_real_power():
    if mtf._IMPLEMENTATION != "cosy":
        pytest.skip("Complex power only implemented for COSY backend so far")

    # A = 1 + i
    A = mtf.from_constant(1.0 + 1.0j, dimension=2)

    # A^2.0 should be 2i
    B = A**2.0
    coeffs = B.to_dict()["coeffs"]
    assert np.allclose(coeffs[0], 0.0 + 2.0j)

    # Sqrt(i) = e^(i pi/4) = (1+i)/sqrt(2)
    I = mtf.from_constant(1.0j, dimension=2)
    S = I**0.5
    expected = (1.0 + 1.0j) / np.sqrt(2)
    coeffs_s = S.to_dict()["coeffs"]
    assert np.allclose(coeffs_s[0], expected)


def test_complex_exp_euler():
    if mtf._IMPLEMENTATION != "cosy":
        pytest.skip("Complex exp only implemented for COSY backend so far")

    # Euler's formula: exp(ix) = cos(x) + i sin(x)
    x = mtf.var(1, dimension=2)
    i = mtf.from_constant(1j, dimension=2)

    ix = i * x
    f_exp = ix.exp()

    f_cos = x.cos()
    f_sin = x.sin()
    f_euler = f_cos + i * f_sin

    # Compare coefficients
    diff = f_exp - f_euler
    coeffs = diff.to_dict()["coeffs"]
    max_err = np.max(np.abs(coeffs)) if len(coeffs) > 0 else 0.0
    assert max_err < 1e-12


def test_complex_log_exp():
    if mtf._IMPLEMENTATION != "cosy":
        pytest.skip("Complex log/exp only implemented for COSY backend so far")

    # log(exp(ix)) = ix
    x = mtf.var(1, dimension=2)
    i = mtf.from_constant(1j, dimension=2)
    ix = i * x

    f = ix.exp().log()

    # Compare with ix
    diff = f - ix
    coeffs = diff.to_dict()["coeffs"]
    max_err = np.max(np.abs(coeffs)) if len(coeffs) > 0 else 0.0
    assert max_err < 1e-12


def test_complex_trig():
    if mtf._IMPLEMENTATION != "cosy":
        pytest.skip("Complex trig only implemented for COSY backend so far")

    # sin^2(z) + cos^2(z) = 1
    x = mtf.var(1, dimension=2)
    y = mtf.var(2, dimension=2)
    z = x + 1j * y

    res = z.sin() ** 2 + z.cos() ** 2

    # Constant term should be 1.0, others 0.0
    coeffs = res.to_dict()["coeffs"]
    assert np.allclose(coeffs[0], 1.0 + 0j)
    assert np.all(np.abs(coeffs[1:]) < 1e-12)
