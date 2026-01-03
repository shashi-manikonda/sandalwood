# mtflib/complex_taylor_function.py
"""
Extends Taylor series to handle complex coefficients.

This module provides the `ComplexMultivariateTaylorFunction` (CMTF) class,
which inherits from `MultivariateTaylorFunction` and is specialized for
Taylor series with complex coefficients. It overloads arithmetic operations
and provides complex-specific functionality like `conjugate`, `real_part`,
and `imag_part`.

The CMTF class is a powerful tool for analyzing the behavior of
complex-valued functions, particularly in fields like physics and
engineering where complex numbers are fundamental.
"""

from collections import defaultdict

import numpy as np

from .taylor_function import MultivariateTaylorFunction


class ComplexMultivariateTaylorFunction(MultivariateTaylorFunction):
    """
    Represents a multivariate Taylor series with complex coefficients.

    This class extends `MultivariateTaylorFunction` to support complex
    arithmetic. All arithmetic operations (`+`, `-`, `*`, `/`, `**`) are
    overloaded to handle complex coefficients correctly.

    Attributes
    ----------
    coeffs : np.ndarray
        A 1D numpy array of dtype `complex128` containing the coefficients.

    Examples
    --------
    >>> from mtflib import ComplexMultivariateTaylorFunction
    >>>
    >>> # Initialize global settings
    >>> cmtf.initialize_mtf(max_order=2, max_dimension=1) # doctest: +ELLIPSIS
    Initializing MTF globals...
    >>>
    >>> # Create a complex constant
    >>> f = ComplexMultivariateTaylorFunction.from_constant(1 + 2j)
    >>>
    >>> # Create a variable
    >>> x = ComplexMultivariateTaylorFunction.from_variable(1, 1)
    >>>
    >>> # Perform complex arithmetic
    >>> g = f + x * (3j)
    >>>
    >>> print(g.get_tabular_dataframe()) # doctest: +ELLIPSIS
    ...Coefficient...
    0...1.0...j...
    1...0.0...j...
    """

    def __init__(self, coefficients, dimension=None, var_name=None, mtf_data=None):
        """
        Initializes a ComplexMultivariateTaylorFunction.

        Ensures that the coefficients are of a complex data type.

        Parameters
        ----------
        coefficients : dict or tuple
            The coefficients of the Taylor series, in the same format as
            `MultivariateTaylorFunction.__init__`.
        dimension : int, optional
            The number of variables. Inferred if not provided.
        var_name : str, optional
            An optional name for the function.
        mtf_data : object, optional
            Internal data for C++ backend. Not for user consumption.
        """
        super().__init__(coefficients, dimension, var_name, mtf_data=mtf_data)
        if self.coeffs.dtype != np.complex128:
            self.coeffs = self.coeffs.astype(np.complex128)

    @classmethod
    def from_constant(cls, constant_value, dimension=None):
        """
        Creates a CMTF representing a constant complex value.

        Parameters
        ----------
        constant_value : complex, float, or int
            The constant value. It will be cast to a complex number.
        dimension : int, optional
            The dimension of the function's domain. If None, the global
            `_MAX_DIMENSION` is used.

        Returns
        -------
        ComplexMultivariateTaylorFunction
            A new CMTF instance representing the constant.

        Examples
        --------
        >>> from mtflib import ComplexMultivariateTaylorFunction
        >>> # Assuming the library is already initialized
        >>> c = ComplexMultivariateTaylorFunction.from_constant(3 + 4j, dimension=1)
        >>> print(c.coeffs[0])
        (3+4j)
        """
        if dimension is None:
            dimension = cls.get_max_dimension()
        constant_value_complex = np.array(
            [complex(constant_value)], dtype=np.complex128
        )
        coeffs = {(0,) * dimension: constant_value_complex}
        return cls(coefficients=coeffs, dimension=dimension)

    @classmethod
    def from_variable(cls, var_index, dimension):
        """
        Creates a CMTF representing a single variable.

        The variable has a coefficient of `1.0 + 0.0j`.

        Parameters
        ----------
        var_index : int
            The 1-based index of the variable to create.
        dimension : int
            The total number of variables.

        Returns
        -------
        ComplexMultivariateTaylorFunction
            A new CMTF instance representing the variable.
        """
        if not (1 <= var_index <= dimension):
            raise ValueError(
                f"Variable index must be in range [1, dimension], "
                f"got {var_index} for dimension {dimension}."
            )

        exponent = [0] * dimension
        exponent[var_index - 1] = 1
        coeffs = {tuple(exponent): np.array([1.0 + 0.0j], dtype=np.complex128)}
        return cls(coefficients=coeffs, dimension=dimension)

    def conjugate(self):
        """
        Computes the complex conjugate of the Taylor function.

        This is done by taking the conjugate of each coefficient.

        Returns
        -------
        ComplexMultivariateTaylorFunction
            A new CMTF representing the complex conjugate.
        """
        return ComplexMultivariateTaylorFunction(
            (self.exponents, np.conjugate(self.coeffs)), self.dimension
        )

    def real_part(self):
        """
        Extracts the real part of the Taylor function.

        Returns
        -------
        MultivariateTaylorFunction
            A new MTF (with real coefficients) representing the real part
            of the function.
        """
        return MultivariateTaylorFunction(
            (self.exponents, np.real(self.coeffs)), self.dimension
        )

    def imag_part(self):
        """
        Extracts the imaginary part of the Taylor function.

        Returns
        -------
        MultivariateTaylorFunction
            A new MTF (with real coefficients) representing the imaginary
            part of the function.
        """
        return MultivariateTaylorFunction(
            (self.exponents, np.imag(self.coeffs)), self.dimension
        )

    def __repr__(self):
        """
        Returns a detailed string representation of the MTF (for debugging).
        """
        df = self.get_tabular_dataframe()
        return f"{df}\n"

    def __str__(self):
        if self.var_name:
            return f"ComplexMultivariateTaylorFunction({self.var_name})"
        df = self.get_tabular_dataframe()
        return f"\n{df}"

    def magnitude(self):
        """
        Raises NotImplementedError as magnitude is not directly representable
        as a CMTF.
        """
        raise NotImplementedError(
            "Magnitude of a ComplexMultivariateTaylorFunction is generally "
            "not a ComplexMultivariateTaylorFunction."
        )

    def phase(self):
        """
        Raises NotImplementedError as phase is not directly representable
        as a CMTF.
        """
        raise NotImplementedError(
            "Phase of a ComplexMultivariateTaylorFunction is generally not a "
            "ComplexMultivariateTaylorFunction."
        )


cmtf = ComplexMultivariateTaylorFunction  # Alias for convenience


def _generate_exponent_combinations(dimension, order):
    """
    Generates all combinations of exponents for a given dimension and order.

    Args:
        dimension (int): The dimension of the multivariate function.
        order (int): The maximum order of terms to generate exponents for.

    Returns:
        list of tuples: A list of exponent tuples.
    """
    if dimension <= 0 or order < 0:
        return []
    if dimension == 1:
        return [(o,) for o in range(order + 1)]
    if order == 0:
        return [(0,) * dimension]

    exponent_combinations = []
    for o in range(order + 1):
        for comb in _generate_exponent_combinations(dimension - 1, o):
            remaining_order = order - o
            for i in range(remaining_order + 1):
                exponent_combinations.append(comb + (i,))
    return exponent_combinations


def _add_coefficient_dicts(dict1, dict2, subtract=False):
    """
    Adds two coefficient dictionaries.

    Args:
        dict1 (defaultdict): First coefficient dictionary.
        dict2 (defaultdict): Second coefficient dictionary.
        subtract (bool, optional): If True, subtract dict2 from dict1.
                                   Defaults to False (add).

    Returns:
        defaultdict: A new coefficient dictionary with the sum (or difference)
                     of coefficients.
    """
    sum_coeffs = defaultdict(
        lambda: (
            np.array([0.0j]).reshape(1)
            if any(isinstance(coeff[0], complex) for coeff in dict1.values())
            or any(isinstance(coeff[0], complex) for coeff in dict2.values())
            else np.array([0.0]).reshape(1)
        )
    )
    for exponents in set(dict1.keys()) | set(dict2.keys()):
        coeff1 = dict1.get(exponents, sum_coeffs.default_factory())
        coeff2 = dict2.get(exponents, sum_coeffs.default_factory())
        if subtract:
            sum_coeffs[exponents] = (
                np.array(coeff1).flatten() - np.array(coeff2).flatten()
            )
        else:
            sum_coeffs[exponents] = (
                np.array(coeff1).flatten() + np.array(coeff2).flatten()
            )
    return sum_coeffs


def convert_to_cmtf(variable):
    """
    Converts a given variable into a ComplexMultivariateTaylorFunction.

    This function handles three types of inputs:
    1. If the input is already a `ComplexMultivariateTaylorFunction`, it is
       returned unchanged.
    2. If the input is a `MultivariateTaylorFunction`, its coefficients are
       cast to complex numbers to create a new
       `ComplexMultivariateTaylorFunction`.
    3. If the input is a scalar (int, float, complex), it is converted into
       a constant `ComplexMultivariateTaylorFunction`.

    Parameters
    ----------
    variable : MultivariateTaylorFunction, ComplexMultivariateTaylorFunction,
               or numeric
        The variable to be converted.

    Returns
    -------
    ComplexMultivariateTaylorFunction
        The resulting CMTF object.

    Raises
    ------
    TypeError
        If the input type is not supported for conversion.
    """
    if isinstance(variable, ComplexMultivariateTaylorFunction):
        return variable
    elif isinstance(variable, MultivariateTaylorFunction):
        exponents = variable.exponents
        coeffs = variable.coeffs.astype(np.complex128)
        return ComplexMultivariateTaylorFunction(
            (exponents, coeffs), variable.dimension
        )
    elif isinstance(variable, (int, float, complex, np.number)):
        dim = 1
        if hasattr(variable, "dimension"):
            dim = variable.dimension
        return ComplexMultivariateTaylorFunction.from_constant(variable, dimension=dim)
    else:
        raise TypeError(
            "Unsupported type for conversion to ComplexMultivariateTaylorFunction."
        )
