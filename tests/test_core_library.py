# mtflib/test_taylor_functions.py

import mtflib
import numpy as np
import pandas as pd
import pytest
from mtflib import (
    ComplexMultivariateTaylorFunction,
    mtf,
)

# Global settings for tests
MAX_ORDER = 5
MAX_DIMENSION = 3
ETOL = 1e-10


@pytest.fixture(scope="function", autouse=True)
def setup_function():
    mtf.initialize_mtf(max_order=MAX_ORDER, max_dimension=MAX_DIMENSION)
    mtf.set_etol(ETOL)
    global_dim = mtf.get_max_dimension()
    exponent_zero = tuple([0] * global_dim)
    yield global_dim, exponent_zero
    mtflib.taylor_function.mtf._INITIALIZED = False


# --- Global Settings Tests ---


# --- Tests for Global Settings ---
def test_global_initialization_once(setup_function):
    global_dim, exponent_zero = setup_function
    assert mtf.get_max_order() == MAX_ORDER
    assert mtf.get_max_dimension() == MAX_DIMENSION
    # Re-initialization should not fail, but print a warning.
    mtf.initialize_mtf(max_order=MAX_ORDER, max_dimension=MAX_DIMENSION)


def test_global_max_order_setting(setup_function):
    global_dim, exponent_zero = setup_function
    mtf.set_max_order(MAX_ORDER)
    assert mtf.get_max_order() == MAX_ORDER
    with pytest.raises(ValueError):
        mtf.set_max_order(-1)  # Invalid order


def test_global_etol_setting(setup_function):
    global_dim, exponent_zero = setup_function
    mtf.set_etol(1e-6)
    assert mtf.get_etol() == 1e-6
    with pytest.raises(ValueError):
        mtf.set_etol(-1e-6)  # Invalid etol
    with pytest.raises(ValueError):
        mtf.set_etol(0.0)  # Invalid etol


# --- Var Function Tests ---
def test_var_creation(setup_function):
    global_dim, exponent_zero = setup_function
    x_var = mtf.var(1)
    assert isinstance(x_var, mtf)
    assert x_var.dimension == global_dim
    exponent_one = [0] * global_dim
    if global_dim > 0:
        exponent_one[0] = 1
    coeff_x1 = x_var.extract_coefficient(tuple(exponent_one))
    assert np.allclose(coeff_x1, 1.0)
    coeff_constant = x_var.extract_coefficient(exponent_zero)
    assert np.allclose(coeff_constant, 0.0)
    with pytest.raises(ValueError):
        mtf.var(0)
    with pytest.raises(ValueError):
        mtf.var(global_dim + 1)


# --- MultivariateTaylorFunction (Real MTF) Tests ---


def test_mtf_constant_creation(setup_function):
    global_dim, exponent_zero = setup_function
    const_mtf = mtf.from_constant(5.0)
    assert np.allclose(const_mtf.eval([0] * global_dim), 5.0)
    assert np.allclose(const_mtf.extract_coefficient(exponent_zero), 5.0)

    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    assert np.allclose(const_mtf.extract_coefficient(tuple(exponent_one)), 0.0)


def test_mtf_variable_evaluation(setup_function):
    global_dim, exponent_zero = setup_function
    evaluation_point_x1 = [0] * global_dim
    if global_dim > 0:
        evaluation_point_x1[0] = 2.0
    evaluation_point_x2 = [0] * global_dim
    if global_dim > 1:
        evaluation_point_x2[1] = 3.0

    x1_var = mtf.var(1)
    x2_var = mtf.var(2)
    assert np.allclose(x1_var.eval(evaluation_point_x1), 2.0)
    assert np.allclose(x2_var.eval(evaluation_point_x2), 3.0)


def test_mtf_truncate(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    exponent_two = list(exponent_zero)
    exponent_three = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
        exponent_two[0] = 2
        exponent_three[0] = 3
    exponent_one = tuple(exponent_one)
    exponent_two = tuple(exponent_two)
    exponent_three = tuple(exponent_three)

    coeffs = {
        exponent_zero: 1.0,
        exponent_one: 2.0,
        exponent_two: 3.0,
        exponent_three: 4.0,
    }
    mtf_instance = mtf(coefficients=coeffs, dimension=global_dim)
    truncated_mtf = mtf_instance.truncate(2)
    assert np.allclose(truncated_mtf.extract_coefficient(exponent_three), 0.0)
    assert np.allclose(truncated_mtf.extract_coefficient(exponent_two), 3.0)
    assert np.allclose(truncated_mtf.extract_coefficient(exponent_one), 2.0)
    assert np.allclose(truncated_mtf.extract_coefficient(exponent_zero), 1.0)


def test_mtf_extract_coefficient(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one_one_list = list(exponent_one_one)
        if global_dim > 1:
            exponent_one_one_list[0] = 1
            exponent_one_one_list[1] = 1
        elif global_dim == 1:
            exponent_one_one_list[0] = 2  # If dimension is 1, (1, 1) becomes (2,)
        exponent_one_one = tuple(exponent_one_one_list)

    mtf_instance = mtf(
        coefficients={exponent_one_one: 2.5, exponent_zero: 1.0},
        dimension=max(2, global_dim),
    )  # Dimension is set here to at least 2 for the test
    assert np.allclose(mtf_instance.extract_coefficient(exponent_one_one), 2.5)
    assert np.allclose(mtf_instance.extract_coefficient(exponent_zero), 1.0)

    exponent_two_zero = list(exponent_zero)
    if global_dim > 0:
        exponent_two_zero_list = list(exponent_two_zero)
        exponent_two_zero_list[0] = 2
        exponent_two_zero = tuple(exponent_two_zero_list)
    assert np.allclose(
        mtf_instance.extract_coefficient(exponent_two_zero), 0.0
    )  # Non-existent coefficient should be zero


def test_mtf_set_coefficient(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    mtf_instance = mtf.from_constant(0.0)
    mtf_instance.set_coefficient(exponent_one, 3.0)
    assert np.allclose(mtf_instance.extract_coefficient(exponent_one), 3.0)
    mtf_instance.set_coefficient(exponent_one, 0.0)  # Setting to zero
    assert np.allclose(mtf_instance.extract_coefficient(exponent_one), 0.0)
    with pytest.raises(TypeError):
        mtf_instance.set_coefficient(list(exponent_one), 2.0)  # Exponents must be tuple
    with pytest.raises(ValueError):
        if global_dim > 1:
            invalid_exponent = tuple(exponent_one[:-1])  # Shorter tuple
        else:
            invalid_exponent = (1, 0)  # Mismatch if global_dim is 1
        mtf_instance.set_coefficient(
            invalid_exponent, 2.0
        )  # Exponent dimension mismatch
    with pytest.raises(TypeError):
        mtf_instance.set_coefficient(exponent_one, "invalid")  # Value must be numeric


def test_mtf_get_max_coefficient(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one_zero = list(exponent_zero)
    exponent_zero_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one_zero[0] = 1
    if global_dim > 1:
        exponent_zero_one[1] = 1
    exponent_one_zero = tuple(exponent_one_zero)
    exponent_zero_one = tuple(exponent_zero_one)

    coeffs = {
        exponent_zero: 1.0,
        exponent_one_zero: -2.0,
        exponent_zero_one: 3.0,
    }
    mtf_instance = mtf(
        coefficients=coeffs, dimension=max(2, global_dim)
    )  # Dimension at least 2 for the test
    assert pytest.approx(mtf_instance.get_max_coefficient()) == 3.0


def test_mtf_get_min_coefficient(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one_zero = list(exponent_zero)
    exponent_zero_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one_zero[0] = 1
    if global_dim > 1:
        exponent_zero_one[1] = 1
    exponent_one_zero = tuple(exponent_one_zero)
    exponent_zero_one = tuple(exponent_zero_one)

    etol = mtf.get_etol()
    coeffs = {
        exponent_zero: 0.1,
        exponent_one_zero: 2.0,
        exponent_zero_one: 3.0,
    }
    mtf_instance = mtf(
        coefficients=coeffs, dimension=max(2, global_dim)
    )  # Dimension at least 2 for the test
    assert (
        pytest.approx(mtf_instance.get_min_coefficient(tolerance=0.5)) == 2.0
    )  # Min non-negligible coefficient (above 0.5)
    coeffs_negligible = {exponent_zero: etol}
    mtf_negligible = mtf(coefficients=coeffs_negligible, dimension=max(2, global_dim))
    assert (
        pytest.approx(mtf_negligible.get_min_coefficient()) == 0.0
    )  # All negligible, returns 0.0


# def test_mtf_to_complex_mtf(setup_function):
#     global_dim, exponent_zero = setup_function
#     mtf_real = MultivariateTaylorFunction.from_constant(2.5)
#     cmtf = mtf_real.to_complex_mtf()
#     assert isinstance(cmtf, ComplexMultivariateTaylorFunction)
#     assert np.allclose(cmtf.extract_coefficient(exponent_zero).real,
#  np.array([2.5]).reshape(1))
#     assert np.allclose(cmtf.extract_coefficient(exponent_zero).imag,
#  np.array([0.0]).reshape(1))

# --- MTF Arithmetic Operations Tests ---


def test_mtf_addition(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = [0] * global_dim
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    mtf1 = mtf({exponent_zero: 1.0, exponent_one: 2.0}, dimension=global_dim)  # 1 + 2x
    mtf2 = mtf({exponent_zero: 3.0, exponent_one: -1.0}, dimension=global_dim)  # 3 - x
    mtf_sum = mtf1 + mtf2  # (1+2x) + (3-x) = 4 + x
    assert np.allclose(mtf_sum.extract_coefficient(exponent_zero), 4.0)
    assert np.allclose(mtf_sum.extract_coefficient(exponent_one), 1.0)
    mtf_sum_const = mtf1 + 2.0  # (1+2x) + 2 = 3 + 2x
    assert np.allclose(mtf_sum_const.extract_coefficient(exponent_zero), 3.0)
    assert np.allclose(mtf_sum_const.extract_coefficient(exponent_one), 2.0)
    mtf_sum_rconst = 2.0 + mtf1  # 2 + (1+2x) = 3 + 2x (commutativity)
    assert np.allclose(mtf_sum_rconst.extract_coefficient(exponent_zero), 3.0)
    assert np.allclose(mtf_sum_rconst.extract_coefficient(exponent_one), 2.0)


def test_mtf_subtraction(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = [0] * global_dim
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    mtf1 = mtf({exponent_zero: 5.0, exponent_one: 3.0}, dimension=global_dim)  # 5 + 3x
    mtf2 = mtf({exponent_zero: 2.0, exponent_one: 1.0}, dimension=global_dim)  # 2 + x
    mtf_diff = mtf1 - mtf2  # (5+3x) - (2+x) = 3 + 2x
    assert np.allclose(mtf_diff.extract_coefficient(exponent_zero), 3.0)
    assert np.allclose(mtf_diff.extract_coefficient(exponent_one), 2.0)
    mtf_diff_const = mtf1 - 2.0  # (5+3x) - 2 = 3 + 3x
    assert np.allclose(mtf_diff_const.extract_coefficient(exponent_zero), 3.0)
    assert np.allclose(mtf_diff_const.extract_coefficient(exponent_one), 3.0)
    mtf_diff_rconst = 4.0 - mtf1  # 4 - (5+3x) = -1 - 3x
    assert np.allclose(mtf_diff_rconst.extract_coefficient(exponent_zero), -1.0)
    assert np.allclose(mtf_diff_rconst.extract_coefficient(exponent_one), -3.0)


def test_mtf_multiplication(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = [0] * global_dim
    exponent_two = [0] * global_dim
    if global_dim > 0:
        exponent_one[0] = 1
        exponent_two[0] = 2
    exponent_one = tuple(exponent_one)
    exponent_two = tuple(exponent_two)

    mtf1 = mtf({exponent_zero: 2.0, exponent_one: 1.0}, dimension=global_dim)  # 2 + x
    mtf2 = mtf({exponent_zero: 3.0, exponent_one: -2.0}, dimension=global_dim)  # 3 - 2x
    mtf_prod = mtf1 * mtf2  # (2+x) * (3-2x) = 6 - 4x + 3x - 2x^2 = 6 - x - 2x^2
    assert np.allclose(mtf_prod.extract_coefficient(exponent_zero), 6.0)
    assert np.allclose(mtf_prod.extract_coefficient(exponent_one), -1.0)
    assert np.allclose(mtf_prod.extract_coefficient(exponent_two), -2.0)
    mtf_prod_const = mtf1 * 3.0  # (2+x) * 3 = 6 + 3x
    assert np.allclose(mtf_prod_const.extract_coefficient(exponent_zero), 6.0)
    assert np.allclose(mtf_prod_const.extract_coefficient(exponent_one), 3.0)
    mtf_prod_rconst = 3.0 * mtf1  # 3 * (2+x) = 6 + 3x (commutativity)
    assert np.allclose(mtf_prod_rconst.extract_coefficient(exponent_zero), 6.0)
    assert np.allclose(mtf_prod_rconst.extract_coefficient(exponent_one), 3.0)


def test_mtf_power(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = [0] * global_dim
    exponent_two = [0] * global_dim
    exponent_three = [0] * global_dim
    exponent_four = [0] * global_dim
    if global_dim > 0:
        exponent_one[0] = 1
        exponent_two[0] = 2
        exponent_three[0] = 3
        exponent_four[0] = 4
    exponent_one = tuple(exponent_one)
    exponent_two = tuple(exponent_two)
    exponent_three = tuple(exponent_three)
    exponent_four = tuple(exponent_four)

    mtf_instance = mtf(
        {exponent_zero: 1.0, exponent_one: 1.0}, dimension=global_dim
    )  # 1 + x
    mtf_sq = mtf_instance**2  # (1+x)^2 = 1 + 2x + x^2
    assert np.allclose(mtf_sq.extract_coefficient(exponent_zero), 1.0)
    assert np.allclose(mtf_sq.extract_coefficient(exponent_one), 2.0)
    assert np.allclose(mtf_sq.extract_coefficient(exponent_two), 1.0)
    mtf_cube = mtf_instance**3  # (1+x)^3 = 1 + 3x + 3x^2 + x^3
    assert np.allclose(mtf_cube.extract_coefficient(exponent_zero), 1.0)
    assert np.allclose(mtf_cube.extract_coefficient(exponent_one), 3.0)
    assert np.allclose(mtf_cube.extract_coefficient(exponent_two), 3.0)
    # pytest.approx might not be directly comparable with numpy arrays
    coeff_three = mtf_cube.extract_coefficient(exponent_three)
    if coeff_three is not None:
        assert coeff_three == pytest.approx(1.0)
    else:
        # If max_order is less than 3, the coefficient might not exist
        assert mtf.get_max_order() < 3

    # Test negative power
    mtf_inv_sq = mtf_instance ** (-2)  # 1/(1+x)^2 = 1 - 2x + 3x^2 - 4x^3 + ...
    assert np.allclose(mtf_inv_sq.extract_coefficient(exponent_zero), 1.0)
    assert np.allclose(mtf_inv_sq.extract_coefficient(exponent_one), -2.0)
    assert np.allclose(mtf_inv_sq.extract_coefficient(exponent_two), 3.0)
    assert np.allclose(mtf_inv_sq.extract_coefficient(exponent_three), -4.0)
    assert np.allclose(mtf_inv_sq.extract_coefficient(exponent_four), 5.0)

    with pytest.raises(ValueError):
        mtf_instance**2.5  # Non-integer power not allowed


def test_mtf_negation(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    mtf_instance = mtf(
        {exponent_zero: 2.0, exponent_one: -1.0}, dimension=global_dim
    )  # 2 - x
    neg_mtf = -mtf_instance  # -(2 - x) = -2 + x
    assert np.allclose(neg_mtf.extract_coefficient(exponent_zero), -2.0)
    assert np.allclose(neg_mtf.extract_coefficient(exponent_one), 1.0)


def test_mtf_eval_shape_consistency(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    mtf_instance = mtf({exponent_zero: 2.0, exponent_one: -1.0}, dimension=global_dim)
    eval_point = [0.5] * global_dim if global_dim > 0 else [0.5]
    if global_dim > 1:
        eval_point = [0.5] + [0] * (
            global_dim - 1
        )  # Evaluate with the first variable as 0.5

    eval_result = mtf_instance.eval(
        eval_point[:global_dim]
    )  # Ensure the evaluation point matches the dimension
    assert eval_result.shape == (1,)  # Evaluation result should be shape (1,)

    mtf_sum_result = (mtf_instance + mtf_instance).eval(eval_point[:global_dim])
    assert mtf_sum_result.shape == (1,)

    mtf_mul_result = (mtf_instance * 2.0).eval(eval_point[:global_dim])
    assert mtf_mul_result.shape == (1,)


# --- ComplexMultivariateTaylorFunction (Complex CMTF) Tests ---
def test_cmtf_creation(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    coeffs = {exponent_zero: 1 + 1j, exponent_one: 2 - 1j}
    cmtf = ComplexMultivariateTaylorFunction(coefficients=coeffs, dimension=global_dim)
    assert np.allclose(cmtf.extract_coefficient(exponent_zero), 1 + 1j)
    assert np.allclose(cmtf.extract_coefficient(exponent_one), 2 - 1j)


def test_cmtf_variable_evaluation(setup_function):
    global_dim, exponent_zero = setup_function
    evaluation_point_x1 = [0] * global_dim
    if global_dim > 0:
        evaluation_point_x1[0] = 0.5
    evaluation_point_x2 = [0] * global_dim
    if global_dim > 1:
        evaluation_point_x2[1] = 0.0

    x1_var_c = ComplexMultivariateTaylorFunction.from_variable(1, dimension=global_dim)
    assert np.allclose(x1_var_c.eval(evaluation_point_x1), 0.5 + 0.0j)


def test_cmtf_truncate(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    exponent_two = list(exponent_zero)
    exponent_three = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
        exponent_two[0] = 2
        exponent_three[0] = 3
    exponent_one = tuple(exponent_one)
    exponent_two = tuple(exponent_two)
    exponent_three = tuple(exponent_three)

    coeffs = {
        exponent_zero: 1 + 0j,
        exponent_one: 2j,
        exponent_two: 3 - 3j,
        exponent_three: 4 + 4j,
    }
    cmtf = ComplexMultivariateTaylorFunction(coefficients=coeffs, dimension=global_dim)
    truncated_cmtf = cmtf.truncate(2)
    assert np.allclose(truncated_cmtf.extract_coefficient(exponent_three), 0.0j)
    assert np.allclose(truncated_cmtf.extract_coefficient(exponent_two), 3 - 3j)
    assert np.allclose(truncated_cmtf.extract_coefficient(exponent_one), 2j)
    assert np.allclose(truncated_cmtf.extract_coefficient(exponent_zero), 1 + 0j)


def test_cmtf_extract_coefficient(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one_zero = list(exponent_zero)
    if global_dim > 0:
        exponent_one_zero[0] = 1
    exponent_one_zero = tuple(exponent_one_zero)

    coeffs = {exponent_one_zero: 1 + 1j}
    cmtf = ComplexMultivariateTaylorFunction(
        coefficients=coeffs, dimension=max(2, global_dim)
    )  # Dimension at least 2 for the test
    assert np.allclose(cmtf.extract_coefficient(exponent_one_zero), 1 + 1j)
    assert np.allclose(
        cmtf.extract_coefficient(exponent_zero), 0.0j
    )  # Default complex zero


def test_cmtf_set_coefficient(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    cmtf = ComplexMultivariateTaylorFunction.from_constant(
        0.0j, dimension=MAX_DIMENSION
    )
    cmtf.set_coefficient(exponent_one, 2 + 2j)
    assert np.allclose(cmtf.extract_coefficient(exponent_one), 2 + 2j)
    cmtf.set_coefficient(exponent_one, 0.0j)  # Setting to zero
    assert np.allclose(cmtf.extract_coefficient(exponent_one), 0.0j)
    with pytest.raises(TypeError):
        cmtf.set_coefficient(list(exponent_one), "invalid")  # Value must be numeric


def test_cmtf_get_max_coefficient(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one_zero = list(exponent_zero)
    exponent_zero_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one_zero[0] = 1
    if global_dim > 1:
        exponent_zero_one[1] = 1
    exponent_one_zero = tuple(exponent_one_zero)
    exponent_zero_one = tuple(exponent_zero_one)

    coeffs = {
        exponent_zero: 1 + 0j,
        exponent_one_zero: 1j,
        exponent_zero_one: -2 + 0j,
    }
    cmtf = ComplexMultivariateTaylorFunction(
        coefficients=coeffs, dimension=max(2, global_dim)
    )  # Dimension at least 2 for the test
    assert pytest.approx(cmtf.get_max_coefficient()) == 2.0  # Max magnitude is 2.0


def test_cmtf_get_min_coefficient(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one_zero = list(exponent_zero)
    exponent_zero_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one_zero[0] = 1
    if global_dim > 1:
        exponent_zero_one[1] = 1
    exponent_one_zero = tuple(exponent_one_zero)
    exponent_zero_one = tuple(exponent_zero_one)

    coeffs = {
        exponent_zero: 0.1j,
        exponent_one_zero: 2 + 0j,
        exponent_zero_one: 3j,
    }
    cmtf = ComplexMultivariateTaylorFunction(
        coefficients=coeffs, dimension=max(2, global_dim)
    )  # Dimension at least 2 for the test
    assert (
        pytest.approx(cmtf.get_min_coefficient(tolerance=0.5)) == 2.0
    )  # Min non-negligible magnitude (above 0.5)
    coeffs_negligible = {exponent_zero: 1e-10j}
    cmtf_negligible = ComplexMultivariateTaylorFunction(
        coefficients=coeffs_negligible, dimension=max(2, global_dim)
    )
    assert (
        pytest.approx(cmtf_negligible.get_min_coefficient()) == 0.0
    )  # All negligible, returns 0.0


def test_cmtf_conjugate(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    cmtf = ComplexMultivariateTaylorFunction(
        {exponent_zero: 1 + 1j, exponent_one: 2 - 1j}, dimension=global_dim
    )
    conj_cmtf = cmtf.conjugate()
    assert np.allclose(conj_cmtf.extract_coefficient(exponent_zero), 1 - 1j)
    assert np.allclose(conj_cmtf.extract_coefficient(exponent_one), 2 + 1j)


def test_cmtf_real_part(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    cmtf = ComplexMultivariateTaylorFunction(
        {exponent_zero: 1 + 1j, exponent_one: 2 - 1j}, dimension=global_dim
    )
    real_mtf = cmtf.real_part()
    assert isinstance(real_mtf, mtf)
    assert np.allclose(real_mtf.extract_coefficient(exponent_zero), 1.0)
    assert np.allclose(real_mtf.extract_coefficient(exponent_one), 2.0)


def test_cmtf_imag_part(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    cmtf = ComplexMultivariateTaylorFunction(
        {exponent_zero: 1 + 1j, exponent_one: 2 - 1j}, dimension=global_dim
    )
    imag_mtf = cmtf.imag_part()
    assert isinstance(imag_mtf, mtf)
    assert np.allclose(imag_mtf.extract_coefficient(exponent_zero), 1.0)
    assert np.allclose(imag_mtf.extract_coefficient(exponent_one), -1.0)


def test_cmtf_magnitude_phase_not_implemented(setup_function):
    global_dim, exponent_zero = setup_function
    cmtf = ComplexMultivariateTaylorFunction.from_constant(
        1 + 1j, dimension=MAX_DIMENSION
    )
    with pytest.raises(NotImplementedError):
        cmtf.magnitude()
    with pytest.raises(NotImplementedError):
        cmtf.phase()


# --- CMTF Arithmetic Operations Tests ---
def test_cmtf_addition(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    cmtf1 = ComplexMultivariateTaylorFunction(
        {exponent_zero: 1 + 1j, exponent_one: 2 - 1j}, dimension=global_dim
    )  # (1+j) + (2-j)x
    cmtf2 = ComplexMultivariateTaylorFunction(
        {exponent_zero: 3 - 2j, exponent_one: -1 + 0.5j}, dimension=global_dim
    )  # (3-2j) + (-1+0.5j)x
    cmtf_sum = cmtf1 + cmtf2  # (4-j) + (1-0.5j)x
    assert np.allclose(cmtf_sum.extract_coefficient(exponent_zero), 4 - 1j)
    assert np.allclose(cmtf_sum.extract_coefficient(exponent_one), 1 - 0.5j)
    cmtf_sum_const = cmtf1 + (2 + 0j)  # (3+j) + (2-j)x
    assert np.allclose(cmtf_sum_const.extract_coefficient(exponent_zero), 3 + 1j)
    assert np.allclose(cmtf_sum_const.extract_coefficient(exponent_one), 2 - 1j)
    cmtf_sum_rconst = (2 + 0j) + cmtf1  # commutativity
    assert np.allclose(cmtf_sum_rconst.extract_coefficient(exponent_zero), 3 + 1j)
    assert np.allclose(cmtf_sum_rconst.extract_coefficient(exponent_one), 2 - 1j)


def test_cmtf_subtraction(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    cmtf1 = ComplexMultivariateTaylorFunction(
        {exponent_zero: 5 + 0j, exponent_one: 3 + 1j}, dimension=global_dim
    )  # 5 + (3+j)x
    cmtf2 = ComplexMultivariateTaylorFunction(
        {exponent_zero: 2 + 1j, exponent_one: 1 - 1j}, dimension=global_dim
    )  # (2+j) + (1-j)x
    cmtf_diff = cmtf1 - cmtf2  # (3-j) + (2+2j)x
    assert np.allclose(cmtf_diff.extract_coefficient(exponent_zero), 3 - 1j)
    assert np.allclose(cmtf_diff.extract_coefficient(exponent_one), 2 + 2j)
    cmtf_diff_const = cmtf1 - (2 + 0j)  # (3+0j) + (3+1j)x
    assert np.allclose(cmtf_diff_const.extract_coefficient(exponent_zero), 3 + 0j)
    assert np.allclose(cmtf_diff_const.extract_coefficient(exponent_one), 3 + 1j)
    cmtf_diff_rconst = (4 + 0j) - cmtf1  # (-1+0j) + (-3-1j)x
    assert np.allclose(cmtf_diff_rconst.extract_coefficient(exponent_zero), -1 + 0j)
    assert np.allclose(cmtf_diff_rconst.extract_coefficient(exponent_one), -3 - 1j)


def test_cmtf_multiplication(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    exponent_two = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
        exponent_two[0] = 2
    exponent_one = tuple(exponent_one)
    exponent_two = tuple(exponent_two)

    cmtf1 = ComplexMultivariateTaylorFunction(
        {exponent_zero: 1 + 0j, exponent_one: 1j}, dimension=global_dim
    )  # 1 + jx
    cmtf2 = ComplexMultivariateTaylorFunction(
        {exponent_zero: 1j, exponent_one: -1 + 0j}, dimension=global_dim
    )  # j - x
    cmtf_prod = cmtf1 * cmtf2  # j - x -x + (-j)x^2 = j - 2x -jx^2
    assert np.allclose(cmtf_prod.extract_coefficient(exponent_zero), 1j)
    assert np.allclose(cmtf_prod.extract_coefficient(exponent_one), -2 + 0j)
    assert np.allclose(cmtf_prod.extract_coefficient(exponent_two), -1j)
    cmtf_prod_const = cmtf1 * (2 + 0j)  # (2+0j) + (2j)x
    assert np.allclose(cmtf_prod_const.extract_coefficient(exponent_zero), 2 + 0j)
    assert np.allclose(cmtf_prod_const.extract_coefficient(exponent_one), 2j)
    cmtf_prod_rconst = (2 + 0j) * cmtf1  # commutativity
    assert np.allclose(cmtf_prod_rconst.extract_coefficient(exponent_zero), 2 + 0j)
    assert np.allclose(cmtf_prod_rconst.extract_coefficient(exponent_one), 2j)


def test_cmtf_power(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    exponent_two = list(exponent_zero)
    exponent_three = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
        exponent_two[0] = 2
        exponent_three[0] = 3
    exponent_one = tuple(exponent_one)
    exponent_two = tuple(exponent_two)
    exponent_three = tuple(exponent_three)

    cmtf = ComplexMultivariateTaylorFunction(
        {exponent_zero: 1 + 0j, exponent_one: 1j}, dimension=global_dim
    )  # 1 + jx
    cmtf_sq = cmtf**2  # (1+jx)^2 = 1 + 2jx - x^2
    assert np.allclose(cmtf_sq.extract_coefficient(exponent_zero), 1 + 0j)
    assert np.allclose(cmtf_sq.extract_coefficient(exponent_one), 2j)
    assert np.allclose(cmtf_sq.extract_coefficient(exponent_two), -1 + 0j)
    cmtf_cube = cmtf**3  # (1+jx)^3 = 1 + 3jx -3x^2 -jx^3
    assert np.allclose(cmtf_cube.extract_coefficient(exponent_zero), 1 + 0j)
    assert np.allclose(cmtf_cube.extract_coefficient(exponent_one), 3j)
    assert np.allclose(cmtf_cube.extract_coefficient(exponent_two), -3 + 0j)
    # The MAX_ORDER check might be too specific, consider checking against a
    # reasonable order assert cmtf_cube.extract_coefficient((MAX_ORDER+1,))
    # == pytest.approx(0.0j) # Truncated


def test_cmtf_negation(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    cmtf = ComplexMultivariateTaylorFunction(
        {exponent_zero: 2 + 0j, exponent_one: -1 + 1j}, dimension=global_dim
    )  # 2 + (-1+j)x
    neg_cmtf = -cmtf  # -2 + (1-j)x
    assert np.allclose(neg_cmtf.extract_coefficient(exponent_zero), -2 + 0j)
    assert np.allclose(neg_cmtf.extract_coefficient(exponent_one), 1 - 1j)


def test_cmtf_eval_shape_consistency(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    cmtf = ComplexMultivariateTaylorFunction(
        {exponent_zero: 2 + 0j, exponent_one: -1 + 1j}, dimension=global_dim
    )
    eval_point = [0.5] * global_dim if global_dim > 0 else [0.5]
    if global_dim > 1:
        eval_point = [0.5] + [0] * (
            global_dim - 1
        )  # Evaluate with the first variable as 0.5

    eval_result = cmtf.eval(eval_point[:global_dim])
    assert eval_result.shape == (1,)

    cmtf_sum_result = (cmtf + cmtf).eval(eval_point[:global_dim])
    assert cmtf_sum_result.shape == (1,)

    cmtf_mul_result = (cmtf * (2 + 0j)).eval(eval_point[:global_dim])
    assert cmtf_mul_result.shape == (1,)


# --- convert_to_mtf Tests ---
def test_to_mtf(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    mtf_instance = mtf.to_mtf(5.0)
    assert isinstance(mtf_instance, mtf)
    assert np.allclose(mtf_instance.eval([0] * global_dim), 5.0)
    assert np.allclose(mtf_instance.extract_coefficient(exponent_zero), 5.0)
    assert np.allclose(mtf_instance.extract_coefficient(exponent_one), 0.0)

    # cmtf = mtf.to_mtf(2+1j)
    # assert isinstance(cmtf, ComplexMultivariateTaylorFunction)
    # assert np.allclose(cmtf.eval([0] * global_dim), np.array([2+1j]).reshape(1))
    # assert np.allclose(cmtf.extract_coefficient(exponent_zero),
    # np.array([2+1j]).reshape(1))
    # assert np.allclose(cmtf.extract_coefficient(exponent_one),
    # np.array([0.0j]).reshape(1))

    x_var_mtf = mtf.to_mtf(mtf.var(1))
    assert isinstance(x_var_mtf, mtf)
    assert x_var_mtf.dimension == global_dim
    eval_point = [0] * global_dim
    if global_dim > 0:
        eval_point[0] = 1.0
    assert np.allclose(x_var_mtf.eval(eval_point), 1.0)

    existing_mtf = mtf.from_constant(3.0)
    converted_mtf = mtf.to_mtf(existing_mtf)
    assert converted_mtf is existing_mtf  # Should return same object if already MTF

    with pytest.raises(TypeError):
        mtf.to_mtf("invalid")  # Invalid type

    numpy_scalar = np.float64(7.0)
    mtf_np_scalar = mtf.to_mtf(numpy_scalar)
    assert np.allclose(mtf_np_scalar.eval([0] * global_dim), 7.0)
    assert np.allclose(mtf_np_scalar.extract_coefficient(exponent_zero), 7.0)
    assert np.allclose(mtf_np_scalar.extract_coefficient(exponent_one), 0.0)

    numpy_0d_array = np.array(9.0)
    mtf_np_0d = mtf.to_mtf(numpy_0d_array)
    assert np.allclose(mtf_np_0d.eval([0] * global_dim), 9.0)
    assert np.allclose(mtf_np_0d.extract_coefficient(exponent_zero), 9.0)
    assert np.allclose(mtf_np_0d.extract_coefficient(exponent_one), 0.0)


# --- Elementary Functions Tests ---
elementary_functions_list = [
    (mtf.cos, "cos_taylor"),
    (mtf.sin, "sin_taylor"),
    (mtf.exp, "exp_taylor"),
    (mtf.gaussian, "gaussian_taylor"),
    (mtf.sqrt, "sqrt_taylor"),
    (mtf.log, "log_taylor"),
    (mtf.arctan, "arctan_taylor"),
    (mtf.sinh, "sinh_taylor"),
    (mtf.cosh, "cosh_taylor"),
    (mtf.tanh, "tanh_taylor"),
    (mtf.arcsin, "arcsin_taylor"),
    (mtf.arccos, "arccos_taylor"),
    (mtf.arctanh, "arctanh_taylor"),
]


@pytest.mark.parametrize("func, func_name", elementary_functions_list)
def test_elementary_functions_scalar(setup_function, func, func_name):
    global_dim, exponent_zero = setup_function
    scalar_input = 0.5
    mtf_result = func(scalar_input)
    assert isinstance(mtf_result, mtf)  # Changed assertion here
    zero_point = tuple([0 for _ in range(global_dim)])
    scalar_eval_result = func(scalar_input).eval(
        zero_point
    )  # Evaluate at 0 as Taylor series is around 0

    if func_name == "cos_taylor":
        expected_val = np.cos(scalar_input)
    elif func_name == "sin_taylor":
        expected_val = np.sin(scalar_input)
    elif func_name == "exp_taylor":
        expected_val = np.exp(scalar_input)
    elif func_name == "gaussian_taylor":
        expected_val = np.exp(-(scalar_input**2))
    elif func_name == "sqrt_taylor":
        expected_val = np.sqrt(scalar_input)
    elif func_name == "log_taylor":
        if scalar_input > 0:
            expected_val = np.log(scalar_input)
        else:
            expected_val = np.nan  # Or handle the domain issue appropriately
    elif func_name == "arctan_taylor":
        expected_val = np.arctan(scalar_input)
    elif func_name == "sinh_taylor":
        expected_val = np.sinh(scalar_input)
    elif func_name == "cosh_taylor":
        expected_val = np.cosh(scalar_input)
    elif func_name == "tanh_taylor":
        expected_val = np.tanh(scalar_input)
    elif func_name == "arcsin_taylor":
        if -1 <= scalar_input <= 1:
            expected_val = np.arcsin(scalar_input)
        else:
            expected_val = np.nan
    elif func_name == "arccos_taylor":
        if -1 <= scalar_input <= 1:
            expected_val = np.arccos(scalar_input)
        else:
            expected_val = np.nan
    elif func_name == "arctanh_taylor":
        if -1 < scalar_input < 1:
            expected_val = np.arctanh(scalar_input)
        else:
            expected_val = np.nan

    if func_name in [
        "sqrt_taylor",
        "log_taylor",
        "arcsin_taylor",
        "arccos_taylor",
        "arctanh_taylor",
    ] and np.isnan(expected_val):
        pytest.xfail(f"{func_name} not defined at scalar_input={scalar_input}")
    else:
        assert np.allclose(scalar_eval_result, expected_val)


# --- Additional Tests ---


def test_mtf_equality(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    mtf1 = mtf({exponent_zero: 1.0, exponent_one: 2.0}, dimension=global_dim)
    mtf2 = mtf({exponent_zero: 1.0, exponent_one: 2.0}, dimension=global_dim)
    mtf3 = mtf({exponent_zero: 1.0, exponent_one: 3.0}, dimension=global_dim)
    assert mtf1 == mtf2
    assert mtf1 != mtf3


def test_cmtf_equality(setup_function):
    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)

    cmtf1 = ComplexMultivariateTaylorFunction(
        {exponent_zero: 1 + 1j, exponent_one: 2 - 1j}, dimension=global_dim
    )
    cmtf2 = ComplexMultivariateTaylorFunction(
        {exponent_zero: 1 + 1j, exponent_one: 2 - 1j}, dimension=global_dim
    )
    cmtf3 = ComplexMultivariateTaylorFunction(
        {exponent_zero: 1 + 1j, exponent_one: 3 - 1j}, dimension=global_dim
    )
    assert cmtf1 == cmtf2
    assert cmtf1 != cmtf3


def test_mtf_pickle_unpickle(setup_function):
    import pickle

    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)
    mtf_obj = mtf({exponent_zero: 1.0, exponent_one: 2.0}, dimension=global_dim)
    pickled_mtf = pickle.dumps(mtf_obj)
    unpickled_mtf = pickle.loads(pickled_mtf)
    assert unpickled_mtf == mtf_obj


def test_cmtf_pickle_unpickle(setup_function):
    import pickle

    global_dim, exponent_zero = setup_function
    exponent_one = list(exponent_zero)
    if global_dim > 0:
        exponent_one[0] = 1
    exponent_one = tuple(exponent_one)
    cmtf = ComplexMultivariateTaylorFunction(
        {exponent_zero: 1 + 1j, exponent_one: 2 - 1j}, dimension=global_dim
    )
    pickled_cmtf = pickle.dumps(cmtf)
    unpickled_cmtf = pickle.loads(pickled_cmtf)
    assert unpickled_cmtf == cmtf


# Tests for the MTFExtended module


def test_array_ufunc():
    """
    Test the __array_ufunc__ implementation with a numpy ufunc.
    """
    x = mtf.var(1)
    y = np.sin(x)

    # The result should be a mtf object
    assert isinstance(y, mtf)

    # Check if the result is correct by evaluating at a point
    eval_point = [0.1, 0, 0]
    numerical_result = y.eval(eval_point)
    analytical_result = np.sin(eval_point[0])

    assert np.isclose(numerical_result[0], analytical_result, rtol=1e-5)


def test_compose_method():
    """
    Test the new compose method on the mtf class.
    """
    x = mtf.var(1)
    y = mtf.var(2)

    f = x**2 + y
    g = x + 1

    # Compose f with g, replacing x with g, using the new method
    h = f.compose({1: g})

    # The result should be (x+1)^2 + y = x^2 + 2x + 1 + y

    # Evaluate at a point
    eval_point = [1, 1, 0]
    numerical_result = h.eval(eval_point)
    analytical_result = (eval_point[0] + 1) ** 2 + eval_point[1]

    assert np.isclose(numerical_result[0], analytical_result)


def test_old_compose_is_gone():
    """
    Test that the old standalone compose function is no longer available.
    """
    with pytest.raises(NameError):
        compose(None, None)  # noqa: F821


def test_list2pd():
    """
    Test the list2pd function.
    """
    x = mtf.var(1)
    y = mtf.var(2)

    mtf1 = x + 2 * y
    mtf2 = x**2

    df = mtf.list2pd([mtf1, mtf2], column_names=["f1", "f2"])

    assert isinstance(df, pd.DataFrame)
    assert "Coeff_f1" in df.columns
    assert "Coeff_f2" in df.columns
    assert "Order" in df.columns
    assert "Exponents" in df.columns

    # Check the number of rows.
    # mtf1 has terms (1,0,0) and (0,1,0)
    # mtf2 has term (2,0,0)
    # The constant term (0,0,0) is also present.
    # So there should be 4 rows.
    # Actually, the constant term is not always present if the coefficient is zero.
    # Let's check the number of non-zero coefficients.
    # There might be overlapping coefficients, but in this case there are not.
    # The list2pd function will merge the terms, so the number of rows will be
    # the number of unique exponents.
    # mtf1 has terms (1,0,0) and (0,1,0). mtf2 has term (2,0,0).
    # The exponents are (1,0,0), (0,1,0), (2,0,0). So there should be 3 rows.
    # Let's get the number of unique exponents.
    all_exponents = np.vstack([mtf1.exponents, mtf2.exponents])
    num_unique_exponents = len(np.unique(all_exponents, axis=0))
    assert len(df) == num_unique_exponents


# --- Cleanup Functionality Tests ---


def test_cleanup_default_behavior(setup_function):
    """
    Tests that cleanup of negligible coefficients is enabled by default.
    """
    etol = mtf.get_etol()
    x = mtf.var(1)

    # Create a function with a negligible term
    f = x + etol / 2

    # The negligible constant term should be removed after the addition
    assert len(f.coeffs) == 1
    assert f.extract_coefficient(tuple([0] * mtf.get_max_dimension())).item() == 0.0


def test_disable_cleanup(setup_function):
    """
    Tests that coefficient cleanup can be disabled.
    """
    mtf.set_truncate_after_operation(False)

    etol = mtf.get_etol()
    x = mtf.var(1)

    # Create a function with a negligible term
    f = x + etol / 2

    # The negligible term should NOT be removed
    assert len(f.coeffs) == 2
    assert abs(f.extract_coefficient(tuple([0] * mtf.get_max_dimension())).item()) > 0

    # Reset for other tests
    mtf.set_truncate_after_operation(True)


def test_set_truncate_after_operation_validation(setup_function):
    """
    Tests the input validation for the setter function.
    """
    with pytest.raises(ValueError, match="Input 'enable' must be a boolean value"):
        mtf.set_truncate_after_operation("not a boolean")


def test_array_ufunc_extended():
    """
    Test the __array_ufunc__ implementation with more numpy ufuncs.
    """
    x = mtf.var(1)
    y = mtf.var(2)

    # Test a few more unary ufuncs
    for ufunc, mtf_func in [
        (np.cos, mtf.cos),
        (np.exp, mtf.exp),
        (np.log, mtf.log),
    ]:
        mtf_from_ufunc = ufunc(x + 0.5)  # use an offset to avoid issues at 0 for log
        mtf_from_direct_call = mtf_func(x + 0.5)
        assert mtf_from_ufunc == mtf_from_direct_call

    # Test a binary ufunc
    mtf1 = x + y
    mtf2 = x - y

    ufunc_add_result = np.add(mtf1, mtf2)
    direct_add_result = mtf1 + mtf2
    assert ufunc_add_result == direct_add_result

    # Test with a scalar
    ufunc_add_scalar_result = np.add(mtf1, 5)
    direct_add_scalar_result = mtf1 + 5
    assert ufunc_add_scalar_result == direct_add_scalar_result


def test_from_numpy_array(setup_function):
    """
    Tests the from_numpy_array class method for correct conversion and shape.
    """
    # 1. Prepare a sample NumPy array
    input_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected_shape = input_array.shape

    # 2. Convert the NumPy array to an array of MTF objects
    mtf_array = mtf.from_numpy_array(input_array)

    # 3. Assert the shape of the resulting array
    assert mtf_array.shape == expected_shape

    # 4. Assert that each element is a constant MTF with the correct value
    #    This can be checked by evaluating the MTF at any point, e.g., origin [0, 0, 0]
    global_dim = mtf.get_max_dimension()
    origin = [0] * global_dim
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            expected_value = input_array[i, j]
            # Use `.eval()` to get the constant value from the MTF
            assert np.isclose(mtf_array[i, j].eval(origin), expected_value)

    # 5. Assert that a TypeError is raised for invalid input
    with pytest.raises(TypeError, match="Input must be a NumPy array."):
        mtf.from_numpy_array([1, 2, 3])


def test_to_numpy_array(setup_function):
    """
    Tests the to_numpy_array class method for correct conversion and shape.
    """
    # 1. Create a NumPy array of numbers as a reference
    input_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected_shape = input_array.shape

    # 2. Use a different method to create a NumPy array of MTF objects
    #    to ensure this test is independent.
    mtf_array = np.array([
        [mtf.from_constant(val) for val in row] for row in input_array
    ])

    # 3. Convert the MTF array back to a NumPy array of numbers
    output_array = mtf.to_numpy_array(mtf_array)

    # 4. Assert the shape of the resulting array
    assert output_array.shape == expected_shape

    # 5. Assert that the values are correctly converted
    assert np.allclose(output_array, input_array)

    # 6. Assert that a TypeError is raised for invalid input
    with pytest.raises(TypeError, match="Input must be a NumPy array."):
        mtf.to_numpy_array([1, 2, 3])


def test_get_constant_and_polynomial_part(setup_function):
    """
    Tests the get_constant and get_polynomial_part methods.
    """
    global_dim, exponent_zero = setup_function

    # 1. Create a sample MTF with a constant and a non-constant term
    x = mtf.var(1)
    constant_term = mtf.from_constant(12.34)
    original_mtf = constant_term + x**2

    # 2. Test get_constant()
    constant_value = original_mtf.get_constant()
    assert np.isclose(constant_value, 12.34)

    # 3. Test get_polynomial_part()
    poly_part_mtf = original_mtf.get_polynomial_part()

    # 4. Assert the polynomial part's constant is zero
    poly_constant_value = poly_part_mtf.get_constant()
    assert np.isclose(poly_constant_value, 0.0)

    # 5. Assert the polynomial part contains the non-constant term
    #    The non-constant part should have one term, x^2
    assert len(poly_part_mtf.exponents) == 1
    poly_exp = poly_part_mtf.exponents[0]
    expected_exp = [0] * global_dim
    expected_exp[0] = 2
    assert np.all(poly_exp == expected_exp)

    # 6. Test the fundamental relationship: original_mtf == constant + polynomial_part
    reconstructed_mtf = mtf.from_constant(constant_value) + poly_part_mtf
    assert original_mtf == reconstructed_mtf


def test_numpy_scalar_arithmetic_bug(setup_function):
    """
    Tests for the bug where operations with NumPy scalars were failing.
    Specifically, `numpy.float64 + mtf_object` was not working as expected.
    """
    global_dim, exponent_zero = setup_function
    u = mtf.var(1, dimension=global_dim)

    # Test case 1: numpy.float64 + mtf
    numpy_scalar = np.float64(0.5)
    result = numpy_scalar + u

    # Expected result: 0.5 + u
    # Check constant part
    assert np.isclose(result.get_constant(), 0.5)
    # Check variable part
    exponent_one = [0] * global_dim
    exponent_one[0] = 1
    assert np.isclose(result.extract_coefficient(tuple(exponent_one)).item(), 1.0)

    # Test case 2: mtf + numpy.float64
    result2 = u + numpy_scalar
    assert np.isclose(result2.get_constant(), 0.5)
    assert np.isclose(result2.extract_coefficient(tuple(exponent_one)).item(), 1.0)

    # Test case 3: The original bug report
    num_linspace_points = np.array([0.5, 1.5, 2.5])
    segment_length = np.float64(0.04)
    factor = (num_linspace_points[0] + 0.5 * u) * segment_length

    # Expected: (0.5 + 0.5*u) * 0.04 = 0.02 + 0.02*u
    assert np.isclose(factor.get_constant(), 0.02)
    assert np.isclose(factor.extract_coefficient(tuple(exponent_one)).item(), 0.02)
