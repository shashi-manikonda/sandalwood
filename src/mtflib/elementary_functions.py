"""
Taylor series for elementary functions, integration, and differentiation.

This module provides functions for computing the Taylor series expansions of
common elementary functions (e.g., sin, exp, log). It also provides the
core operators for differentiation and integration, which, together with the
arithmetic operations in `MultivariateTaylorFunction`, form a complete
Differential Algebra (DA).
"""

import math
from typing import Optional

import numpy as np

from . import (
    elementary_coefficients,
)

# Import the new module with loaded coefficients
from .complex_taylor_function import ComplexMultivariateTaylorFunction
from .taylor_function import (
    MultivariateTaylorFunction,
    _generate_exponent,
    _split_constant_polynomial_part,
    _sqrt_taylor,
)


def _apply_constant_factoring(
    input_mtf: MultivariateTaylorFunction,
    constant_factor_function,
    expansion_function_around_zero,
    combine_operation,
) -> MultivariateTaylorFunction:
    """Helper: Applies constant factoring for Taylor expansion."""
    constant_term_C_value, polynomial_part_mtf = _split_constant_polynomial_part(
        input_mtf
    )
    constant_factor_value = constant_factor_function(constant_term_C_value)
    polynomial_expansion_mtf = expansion_function_around_zero(polynomial_part_mtf)

    if combine_operation == "*":
        result_mtf = polynomial_expansion_mtf * constant_factor_value
    elif combine_operation == "+":
        result_mtf = polynomial_expansion_mtf + constant_factor_value
    elif combine_operation == "-":
        result_mtf = polynomial_expansion_mtf - constant_factor_value
    else:
        raise ValueError(
            f"Unsupported combine_operation: {combine_operation}. "
            "Must be '*', '+', or '-'."
        )

    return result_mtf


def _create_composed_taylor_from_coeffs(
    variable,
    coeff_key: str,
    order: Optional[int] = None,
    dynamic_coeff_func=None,
) -> MultivariateTaylorFunction:
    """
    Creates a 1D Taylor series from coefficients and composes it with a variable.

    This helper function streamlines the creation of elementary function
    expansions (like sin, cos, exp) by handling the boilerplate of:
    1. Looking up precomputed coefficients.
    2. Dynamically calculating coefficients beyond the precomputed range if a
       function is provided.
    3. Creating a 1D Taylor series from these coefficients.
    4. Composing the 1D series with the input `variable` (an MTF).
    5. Truncating the result to the desired order.

    Parameters
    ----------
    variable : MultivariateTaylorFunction
        The input MTF to be composed with the 1D Taylor series.
    coeff_key : str
        The key to look up the precomputed coefficients (e.g., "sin", "exp").
    order : int, optional
        The desired truncation order for the final result. If None, the
        global `_MAX_ORDER` is used.
    dynamic_coeff_func : callable, optional
        A function to compute coefficients for orders beyond the precomputed
        range. It should take the order `n` as input and return the
        coefficient value.

    Returns
    -------
    MultivariateTaylorFunction
        The new MTF representing the composed function.

    Raises
    ------
    ValueError
        If precomputed coefficients for the given `coeff_key` are not found.
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()

    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    taylor_1d_coeffs = {}
    taylor_dim_1d = 1
    var_index_1d = 0

    max_precomputed_order = min(order, elementary_coefficients.MAX_PRECOMPUTED_ORDER)
    precomputed_coeffs = elementary_coefficients.precomputed_coefficients.get(coeff_key)
    if precomputed_coeffs is None:
        raise ValueError(
            f"Precomputed coefficients for '{coeff_key}' function not found. "
            "Ensure coefficients are loaded."
        )

    # Use precomputed coefficients up to the available order
    for n_order in range(max_precomputed_order + 1):
        if n_order < len(precomputed_coeffs):
            coeff_val = precomputed_coeffs[n_order]
            if abs(coeff_val) > 1e-16:  # Only store non-zero coefficients
                taylor_1d_coeffs[
                    _generate_exponent(n_order, var_index_1d, taylor_dim_1d)
                ] = coeff_val

    # Dynamically compute coefficients for higher orders if a function is provided
    if order > max_precomputed_order and dynamic_coeff_func:
        print(
            f"Warning: Requested order {order} exceeds precomputed order "
            f"{elementary_coefficients.MAX_PRECOMPUTED_ORDER}. "
            "Calculations may be slower for higher orders."
        )
        for n_order in range(max_precomputed_order + 1, order + 1):
            coeff_val = dynamic_coeff_func(n_order)
            if abs(coeff_val) > 1e-16:  # Only store non-zero coefficients
                taylor_1d_coeffs[
                    _generate_exponent(n_order, var_index_1d, taylor_dim_1d)
                ] = coeff_val

    # Create and compose the 1D Taylor series
    taylor_1d_mtf = type(input_mtf)(
        coefficients=taylor_1d_coeffs, dimension=taylor_dim_1d
    )
    composed_mtf = taylor_1d_mtf.compose({1: input_mtf})
    return composed_mtf.truncate(order)


def _sin_taylor(variable, order: Optional[int] = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `sin(x)`.

    Uses the identity `sin(C + p) = sin(C)cos(p) + cos(C)sin(p)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `sin(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=3, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.sin(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      1    (1,)
    1 -1.666667e-01      3    (3,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    constant_term_C_value, polynomial_part_mtf = _split_constant_polynomial_part(
        input_mtf
    )
    constant_sin_C = math.sin(constant_term_C_value)
    constant_cos_C = math.cos(constant_term_C_value)

    term1_mtf = cos_taylor_around_zero(polynomial_part_mtf) * constant_sin_C
    term2_mtf = sin_taylor_around_zero(polynomial_part_mtf) * constant_cos_C

    result_mtf = term1_mtf + term2_mtf
    return result_mtf.truncate(order)


def _cos_taylor(variable, order: Optional[int] = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `cos(x)`.

    Uses the identity `cos(C + p) = cos(C)cos(p) - sin(C)sin(p)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `cos(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=4, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.cos(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      0    (0,)
    1 -5.000000e-01      2    (2,)
    2  4.166667e-02      4    (4,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    constant_term_C_value, polynomial_part_mtf = _split_constant_polynomial_part(
        input_mtf
    )
    constant_cos_C = math.cos(constant_term_C_value)
    constant_sin_C = math.sin(constant_term_C_value)

    term1_mtf = cos_taylor_around_zero(polynomial_part_mtf) * constant_cos_C
    term2_mtf = sin_taylor_around_zero(polynomial_part_mtf) * constant_sin_C

    result_mtf = term1_mtf - term2_mtf
    return result_mtf.truncate(order)


def sin_taylor_around_zero(
    variable, order: Optional[int] = None
) -> MultivariateTaylorFunction:
    """
    Helper: Taylor expansion of sin(u) around zero, using precomputed coefficients.
    """

    def dynamic_sin(n):
        if n % 2 == 1:
            return (-1) ** ((n - 1) // 2) / math.factorial(n)
        return 0.0

    return _create_composed_taylor_from_coeffs(variable, "sin", order, dynamic_sin)


def cos_taylor_around_zero(
    variable, order: Optional[int] = None
) -> MultivariateTaylorFunction:
    """
    Helper: Taylor expansion of cos(u) around zero, using precomputed coefficients.
    """

    def dynamic_cos(n):
        if n % 2 == 0:
            return (-1) ** (n // 2) / math.factorial(n)
        return 0.0

    return _create_composed_taylor_from_coeffs(variable, "cos", order, dynamic_cos)


def _tan_taylor(variable, order: Optional[int] = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `tan(x)`.

    Calculated as `sin(x) / cos(x)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `tan(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=4, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.tan(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      1    (1,)
    1  3.333333e-01      3    (3,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    sin_mtf = _sin_taylor(variable, order=order)
    cos_mtf = _cos_taylor(variable, order=order)
    tan_mtf = sin_mtf / cos_mtf
    return tan_mtf.truncate(order)


def _exp_taylor(variable, order: Optional[int] = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `exp(x)`.

    Uses the identity `exp(C + p) = exp(C) * exp(p)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `exp(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=3, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.exp(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      0    (0,)
    1  1.000000e+00      1    (1,)
    2  5.000000e-01      2    (2,)
    3  1.666667e-01      3    (3,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    return _apply_constant_factoring(
        input_mtf, math.exp, exp_taylor_around_zero, "*"
    ).truncate(order)


def exp_taylor_around_zero(
    variable, order: Optional[int] = None
) -> MultivariateTaylorFunction:
    """
    Helper: Taylor expansion of exp(u) around zero, using precomputed coefficients.
    """
    return _create_composed_taylor_from_coeffs(
        variable, "exp", order, lambda n: 1.0 / math.factorial(n)
    )


def _gaussian_taylor(
    variable, order: Optional[int] = None
) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of a Gaussian function, `exp(-x^2)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `exp(-x^2)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=4, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.gaussian(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      0    (0,)
    1 -1.000000e+00      2    (2,)
    2  5.000000e-01      4    (4,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    return _exp_taylor(-(input_mtf**2), order=order)


def _log_taylor(variable, order: Optional[int] = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `log(x)`.

    Uses the identity `log(C + p) = log(C) + log(1 + p/C)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`. Must have a positive constant term.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `log(x)`.

    Raises
    ------
    ValueError
        If the constant term of the input is not positive.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=3, max_dimension=1)
    >>> x = mtf.var(1)
    >>> # We compute log(1+x) since log(x) is singular at x=0
    >>> f = mtf.log(1 + x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      1    (1,)
    1 -5.000000e-01      2    (2,)
    2  3.333333e-01      3    (3,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)

    constant_term_C_value, polynomial_part_B_mtf = _split_constant_polynomial_part(
        input_mtf
    )

    if constant_term_C_value <= 1e-9:
        raise ValueError(
            "Constant part of input to log_taylor is too close to zero or "
            "negative. Logarithm is not defined for non-positive values, "
            "and this method requires a positive constant term."
        )
    if constant_term_C_value < 0:  # Explicit check for negative constant part
        raise ValueError(
            "Constant part of input to log_taylor is negative. Logarithm is "
            "not defined for negative values."
        )

    constant_factor_log_C = math.log(constant_term_C_value)
    polynomial_part_x_mtf = polynomial_part_B_mtf / constant_term_C_value
    log_1_plus_x_mtf = log_taylor_1D_expansion(polynomial_part_x_mtf, order=order)
    result_mtf = log_1_plus_x_mtf + constant_factor_log_C
    return result_mtf.truncate(order)


def log_taylor_1D_expansion(
    variable, order: Optional[int] = None
) -> MultivariateTaylorFunction:
    """
    Helper: 1D Taylor expansion of log(1+u) around zero, using precomputed coefficients.
    """

    def dynamic_log(n):
        if n >= 1:
            return ((-1) ** (n - 1)) / n
        return 0.0

    return _create_composed_taylor_from_coeffs(variable, "log", order, dynamic_log)


def _arctan_taylor(variable, order: Optional[int] = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `arctan(x)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `arctan(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=3, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.arctan(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      1    (1,)
    1 -3.333333e-01      3    (3,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)

    constant_term_C_value, polynomial_part_B_mtf = _split_constant_polynomial_part(
        input_mtf
    )

    constant_arctan_C = math.atan(constant_term_C_value)
    denominator_mtf = MultivariateTaylorFunction.from_constant(
        1.0 + constant_term_C_value**2
    ) + (float(constant_term_C_value) * polynomial_part_B_mtf)
    argument_mtf = polynomial_part_B_mtf / denominator_mtf
    arctan_argument_mtf = arctan_taylor_1D_expansion(argument_mtf, order=order)
    result_mtf = constant_arctan_C + arctan_argument_mtf
    return result_mtf.truncate(order)


def arctan_taylor_1D_expansion(
    variable, order: Optional[int] = None
) -> MultivariateTaylorFunction:
    """
    Helper: 1D Taylor expansion of arctan(u) around zero, using
    precomputed coefficients.
    """

    def dynamic_arctan(n):
        if n % 2 == 1:
            term_index = (n - 1) // 2
            return ((-1) ** term_index) / n
        return 0.0

    return _create_composed_taylor_from_coeffs(
        variable, "arctan", order, dynamic_arctan
    )


def _sinh_taylor(variable, order: Optional[int] = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `sinh(x)`.

    Uses the identity `sinh(C + p) = sinh(C)cosh(p) + cosh(C)sinh(p)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `sinh(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=3, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.sinh(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      1    (1,)
    1  1.666667e-01      3    (3,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    constant_term_C_value, polynomial_part_mtf = _split_constant_polynomial_part(
        input_mtf
    )
    constant_sinh_C = math.sinh(constant_term_C_value)
    constant_cosh_C = math.cosh(constant_term_C_value)

    term1_mtf = cosh_taylor_around_zero(polynomial_part_mtf) * constant_sinh_C
    term2_mtf = sinh_taylor_around_zero(polynomial_part_mtf) * constant_cosh_C

    result_mtf = term1_mtf + term2_mtf
    return result_mtf.truncate(order)


def _cosh_taylor(variable, order: Optional[int] = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `cosh(x)`.

    Uses the identity `cosh(C + p) = cosh(C)cosh(p) + sinh(C)sinh(p)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `cosh(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=4, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.cosh(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      0    (0,)
    1  5.000000e-01      2    (2,)
    2  4.166667e-02      4    (4,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)
    constant_term_C_value, polynomial_part_mtf = _split_constant_polynomial_part(
        input_mtf
    )
    constant_cosh_C = math.cosh(constant_term_C_value)
    constant_sinh_C = math.sinh(constant_term_C_value)

    term1_mtf = cosh_taylor_around_zero(polynomial_part_mtf) * constant_cosh_C
    term2_mtf = sinh_taylor_around_zero(polynomial_part_mtf) * constant_sinh_C

    result_mtf = term1_mtf + term2_mtf
    return result_mtf.truncate(order)


def sinh_taylor_around_zero(
    variable, order: Optional[int] = None
) -> MultivariateTaylorFunction:
    """
    Helper: Taylor expansion of sinh(u) around zero, using precomputed coefficients.
    """

    def dynamic_sinh(n):
        if n % 2 == 1:
            return 1.0 / math.factorial(n)
        return 0.0

    return _create_composed_taylor_from_coeffs(variable, "sinh", order, dynamic_sinh)


def cosh_taylor_around_zero(
    variable, order: Optional[int] = None
) -> MultivariateTaylorFunction:
    """
    Helper: Taylor expansion of cosh(u) around zero, using precomputed coefficients.
    """

    def dynamic_cosh(n):
        if n % 2 == 0:
            return 1.0 / math.factorial(n)
        return 0.0

    return _create_composed_taylor_from_coeffs(variable, "cosh", order, dynamic_cosh)


def _tanh_taylor(variable, order: Optional[int] = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `tanh(x)`.

    Calculated as `sinh(x) / cosh(x)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `tanh(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=4, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.tanh(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      1    (1,)
    1 -3.333333e-01      3    (3,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    sinh_mtf = _sinh_taylor(variable, order=order)
    cosh_mtf = _cosh_taylor(variable, order=order)
    tanh_mtf = sinh_mtf / cosh_mtf
    return tanh_mtf.truncate(order)


def _arctanh_taylor(
    variable, order: Optional[int] = None
) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `arctanh(x)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `arctanh(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=3, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.arctanh(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      1    (1,)
    1  3.333333e-01      3    (3,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    input_mtf = MultivariateTaylorFunction.to_mtf(variable)

    constant_term_C_value, polynomial_part_B_mtf = _split_constant_polynomial_part(
        input_mtf
    )  # Corrected variable name here and below
    constant_arctanh_C = math.atanh(
        constant_term_C_value
    )  # Note: math.atanh for scalar input

    denominator_mtf = MultivariateTaylorFunction.from_constant(
        1.0 - constant_term_C_value**2
    ) - (float(constant_term_C_value) * polynomial_part_B_mtf)  # Corrected for arctanh
    argument_mtf = polynomial_part_B_mtf / denominator_mtf
    arctanh_argument_mtf = arctanh_taylor_1D_expansion(argument_mtf, order=order)
    result_mtf = constant_arctanh_C + arctanh_argument_mtf  # Correct scalar addition
    return result_mtf.truncate(order)


def arctanh_taylor_1D_expansion(
    variable, order: Optional[int] = None
) -> MultivariateTaylorFunction:
    """
    Helper: 1D Taylor expansion of arctanh(u) around zero, using
    precomputed coefficients.
    """

    def dynamic_arctanh(n):
        if n % 2 == 1:
            return 1.0 / float(n)
        return 0.0

    return _create_composed_taylor_from_coeffs(
        variable, "arctanh", order, dynamic_arctanh
    )


def _arcsin_taylor(variable, order: Optional[int] = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `arcsin(x)`.

    Uses the identity `arcsin(x) = arctan(x / sqrt(1 - x^2))`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `arcsin(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=3, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.arcsin(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.000000e+00      1    (1,)
    1  1.666667e-01      3    (3,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    x_mtf = MultivariateTaylorFunction.to_mtf(variable)
    x_squared_mtf = x_mtf * x_mtf
    one_minus_x_squared_mtf = 1.0 - x_squared_mtf
    sqrt_of_one_minus_x_squared_mtf = _sqrt_taylor(one_minus_x_squared_mtf)
    argument_mtf = x_mtf / sqrt_of_one_minus_x_squared_mtf
    arcsin_mtf = _arctan_taylor(argument_mtf, order=order)
    return arcsin_mtf.truncate(order)


def _arccos_taylor(variable, order: Optional[int] = None) -> MultivariateTaylorFunction:
    """
    Computes the Taylor expansion of `arccos(x)`.

    Uses the identity `arccos(x) = pi/2 - arcsin(x)`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction or numeric
        The input function `x`.
    order : int, optional
        The truncation order for the result. If None, the global default
        is used.

    Returns
    -------
    MultivariateTaylorFunction
        The Taylor series for `arccos(x)`.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> import numpy as np
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=3, max_dimension=1)
    >>> x = mtf.var(1)
    >>> f = mtf.arccos(x)
    >>> print(f.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  1.570796e+00      0    (0,)
    1 -1.000000e+00      1    (1,)
    2 -1.666667e-01      3    (3,)
    """
    if order is None:
        order = MultivariateTaylorFunction.get_max_order()
    arcsin_mtf = _arcsin_taylor(variable, order=order)  # Get arcsin_taylor expansion
    pi_over_2_constant = np.pi / 2.0
    arccos_mtf = (
        MultivariateTaylorFunction.to_mtf(pi_over_2_constant) - arcsin_mtf
    )  # Perform MTF subtraction
    return arccos_mtf.truncate(order)  # Truncate to the desired order


def _integrate(
    mtf_instance,
    integration_variable_index,
    lower_limit=None,
    upper_limit=None,
):
    r"""
    Performs definite or indefinite integration of an MTF.

    This function corresponds to the inverse derivation operator
    :math:`\partial_{\bigcirc}^{-1}` of the Differential Algebra. It integrates
    the Taylor series with respect to one of its variables.

    Parameters
    ----------
    mtf_instance : MultivariateTaylorFunction
        The function to integrate.
    integration_variable_index : int
        The 1-based index of the variable to integrate with respect to.
    lower_limit : float, optional
        The lower limit for definite integration.
    upper_limit : float, optional
        The upper limit for definite integration.

    Returns
    -------
    MultivariateTaylorFunction
        If an indefinite integral, a new MTF representing the integral.
        If a definite integral, a new MTF representing the result after
        integrating and substituting the bounds.

    Raises
    ------
    ValueError
        If limits are partially provided or if the variable index is invalid.
    TypeError
        If inputs have incorrect types.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=3, max_dimension=2)
    >>> x, y = mtf.var(1), mtf.var(2)
    >>> f = x*y + y**2
    >>>
    >>> # Indefinite integral with respect to x
    >>> F_indef = f.integrate(1)
    >>> print(F_indef.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  5.000000e-01      2    (2, 1)
    1  1.000000e+00      3    (1, 2)
    >>>
    >>> # Definite integral with respect to y from 0 to 2
    >>> F_def = f.integrate(2, lower_limit=0.0, upper_limit=2.0)
    >>> print(F_def.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  2.666667e+00      0    (0, 0)
    1  2.000000e+00      1    (1, 0)
    """
    if not isinstance(
        mtf_instance,
        (MultivariateTaylorFunction, ComplexMultivariateTaylorFunction),
    ):
        raise TypeError(
            "mtf_instance must be a MultivariateTaylorFunction or "
            "ComplexMultivariateTaylorFunction."
        )
    if not isinstance(integration_variable_index, int):
        raise ValueError(
            "integration_variable_index must be an integer dimension index (1-based)."
        )
    if not (1 <= integration_variable_index <= mtf_instance.dimension):
        raise ValueError(
            f"integration_variable_index must be between 1 and "
            f"{mtf_instance.dimension}, inclusive."
        )
    if (lower_limit is not None and upper_limit is None) or (
        upper_limit is not None and lower_limit is None
    ):
        raise ValueError(
            "Both lower_limit and upper_limit must be provided for definite "
            "integration."
        )
    if lower_limit is not None and not isinstance(lower_limit, (int, float)):
        raise TypeError("lower_limit must be a number (int or float).")
    if upper_limit is not None and not isinstance(upper_limit, (int, float)):
        raise TypeError("upper_limit must be a number (int or float).")

    # Handle empty mtf_instance
    if mtf_instance.exponents.size == 0:
        return mtf_instance.copy()

    # Perform the integration. The order of terms increases by 1.
    new_exponents = mtf_instance.exponents.copy()
    new_coeffs = mtf_instance.coeffs.copy()

    p = new_exponents[:, integration_variable_index - 1]
    new_coeffs /= p + 1
    new_exponents[:, integration_variable_index - 1] += 1

    # The indefinite integral may have terms of order > _MAX_ORDER.
    # This is expected, as integration increases polynomial degree.
    # Subsequent operations will truncate it back to the global max order.
    indefinite_integral_mtf = type(mtf_instance)(
        (new_exponents, new_coeffs), dimension=mtf_instance.dimension
    )

    if lower_limit is not None and upper_limit is not None:
        # For definite integration, evaluate at the limits.
        # The intermediate MTFs (upper_limit_mtf, lower_limit_mtf) are not
        # truncated, preserving the full result of the integration.
        upper_limit_mtf = indefinite_integral_mtf.substitute_variable(
            integration_variable_index, upper_limit
        )
        lower_limit_mtf = indefinite_integral_mtf.substitute_variable(
            integration_variable_index, lower_limit
        )

        # The result of the subtraction is also not truncated.
        definite_integral_mtf_full_order = upper_limit_mtf - lower_limit_mtf

        # Finally, truncate the result to the global max order.
        # This is the only place where truncation is needed.
        return definite_integral_mtf_full_order.truncate()
    else:
        # Return the indefinite integral. It may have terms of a higher order
        # than the original function, which is the expected behavior.
        # Subsequent operations will truncate it as needed.
        return indefinite_integral_mtf


def _derivative(mtf_instance, deriv_dim):
    r"""
    Computes the partial derivative of an MTF.

    This function corresponds to the derivation operator
    :math:`\partial_{\bigcirc}` of the Differential Algebra. It differentiates
    the Taylor series with respect to one of its variables.

    Parameters
    ----------
    mtf_instance : MultivariateTaylorFunction
        The function to differentiate.
    deriv_dim : int
        The 1-based index of the variable to differentiate with respect to.

    Returns
    -------
    MultivariateTaylorFunction
        A new MTF representing the partial derivative.

    Raises
    ------
    TypeError
        If `mtf_instance` is not a `MultivariateTaylorFunction`.
    ValueError
        If `deriv_dim` is not a valid dimension index.

    Examples
    --------
    >>> from mtflib import MultivariateTaylorFunction
    >>> mtf = MultivariateTaylorFunction
    >>> mtf.initialize_mtf(max_order=3, max_dimension=2)
    >>> x, y = mtf.var(1), mtf.var(2)
    >>> f = x**3 * y**2
    >>>
    >>> # Differentiate with respect to x
    >>> df_dx = f.derivative(1)
    >>> print(df_dx.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  3.000000e+00      4    (2, 2)
    >>>
    >>> # Differentiate with respect to y
    >>> df_dy = f.derivative(2)
    >>> print(df_dy.get_tabular_dataframe())
       Coefficient  Order Exponents
    0  2.000000e+00      4    (3, 1)
    """
    if not isinstance(mtf_instance, MultivariateTaylorFunction):
        raise TypeError("mtf_instance must be a MultivariateTaylorFunction object.")
    if (
        not isinstance(deriv_dim, int)
        or deriv_dim < 1
        or deriv_dim > mtf_instance.dimension
    ):
        raise ValueError(
            f"deriv_dim must be an integer between 1 and "
            f"{mtf_instance.dimension} inclusive."
        )

    deriv_dim_index = deriv_dim - 1  # Convert 1-based to 0-based index

    # Filter out terms where the exponent in the derivative dimension is 0
    mask = mtf_instance.exponents[:, deriv_dim_index] > 0

    if not np.any(mask):
        return type(mtf_instance)(
            (
                np.empty((0, mtf_instance.dimension), dtype=np.int32),
                np.empty((0,), dtype=mtf_instance.coeffs.dtype),
            ),
            mtf_instance.dimension,
        )

    new_exponents = mtf_instance.exponents[mask].copy()
    new_coeffs = mtf_instance.coeffs[mask].copy()

    p = new_exponents[:, deriv_dim_index]
    new_coeffs *= p
    new_exponents[:, deriv_dim_index] -= 1

    return type(mtf_instance)((new_exponents, new_coeffs), mtf_instance.dimension)
