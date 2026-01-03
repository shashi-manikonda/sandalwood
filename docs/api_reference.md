# mtflib API Reference

This document provides a detailed API reference for the `mtflib` package.

## `mtflib` Package

The `mtflib` package provides tools for working with multivariate Taylor series expansions.

### `mtf`

An alias for the `MultivariateTaylorFunction` class. It is the main entry point for most users.

## `mtflib.taylor_function` Module

This module contains the core `MultivariateTaylorFunction` class.

### `MultivariateTaylorFunction` Class

The fundamental class for representing a function as a DA vector of its Taylor coefficients.

#### Class Attributes

| Attribute | Type | Description |
|---|---|---|
| `_MAX_ORDER` | `int` | Global setting for the maximum order of Taylor expansions. |
| `_MAX_DIMENSION` | `int` | Global setting for the maximum number of variables. |
| `_ETOL` | `float` | Global tolerance for floating-point comparisons. |
| `_TRUNCATE_AFTER_OPERATION` | `bool` | If `True`, coefficients smaller than `_ETOL` are automatically removed after each operation. |

#### Instance Attributes

| Attribute | Type | Description |
|---|---|---|
| `exponents` | `numpy.ndarray` | A 2D numpy array of shape `(n_terms, dimension)`, where each row represents the multi-index exponent of a term. |
| `coeffs` | `numpy.ndarray` | A 1D numpy array of shape `(n_terms,)` containing the coefficient for each corresponding exponent. |
| `dimension` | `int` | The number of variables in the Taylor series. |
| `var_name` | `str`, optional | An optional name for the function. |

#### Methods

| Method | Description | Arguments | Returns |
|---|---|---|---|
| `initialize_mtf` | Initializes global settings for the mtflib library. | `max_order` (`int`, optional): The default maximum order for Taylor series expansions.<br>`max_dimension` (`int`, optional): The default maximum number of variables for functions.<br>`implementation` (`{'cpp', 'python'}`, optional): The backend implementation to use. | `None` |
| `get_max_coefficient_count` | Calculates max coefficient count for given order/dimension. | `max_order` (`int`, optional): The maximum order.<br>`max_dimension` (`int`, optional): The maximum dimension. | `int`: The maximum number of coefficients. |
| `get_precomputed_coefficients` | Returns the precomputed Taylor coefficients for elementary functions. | `None` | `dict`: The precomputed coefficients. |
| `get_mtf_initialized_status` | Returns initialization status of MTF globals. | `None` | `bool`: `True` if initialized, `False` otherwise. |
| `set_max_order` | Sets the global maximum order for Taylor series. | `order` (`int`): The maximum order. | `None` |
| `get_max_order` | Returns the global maximum order for Taylor series. | `None` | `int`: The maximum order. |
| `get_max_dimension` | Returns the global maximum dimension (number of variables). | `None` | `int`: The maximum dimension. |
| `set_etol` | Sets the global error tolerance (etol) for `mtflib`. | `etol` (`float`): The error tolerance. | `None` |
| `get_etol` | Returns the global error tolerance (etol). | `None` | `float`: The error tolerance. |
| `set_truncate_after_operation` | Sets the global flag to enable or disable automatic coefficient cleanup. | `enable` (`bool`): `True` to enable, `False` to disable. | `None` |
| `from_constant` | Creates a MultivariateTaylorFunction representing a constant value. | `constant_value` (`float` or `int`): The constant value.<br>`dimension` (`int`, optional): The dimension of the function's domain. | `mtf`: A new MTF instance. |
| `var` | Creates a MultivariateTaylorFunction representing a single variable. | `var_index` (`int`): The 1-based index of the variable.<br>`dimension` (`int`, optional): The total number of variables. | `mtf`: An mtf object representing the variable. |
| `list2pd` | Merges a list of MTFs into a single pandas DataFrame for comparison. | `mtfs` (`list` of `mtf`): A list of MTF objects.<br>`column_names` (`list` of `str`, optional): A list of names for the coefficient columns. | `pandas.DataFrame`: A DataFrame for comparison. |
| `to_mtf` | Converts input to MultivariateTaylorFunction or ComplexMultivariateTaylorFunction. | `input_val` (`any`): The input to convert.<br>`dimension` (`int`, optional): The dimension of the function's domain. | `mtf`: The converted MTF object. |
| `to_json` | Serializes the MTF object to a JSON string. | `None` | `str`: A JSON string representation. |
| `from_json` | Creates an MTF object from a JSON string. | `json_str` (`str`): The JSON string. | `mtf` or `ComplexMultivariateTaylorFunction`: The deserialized object. |
| `eval` | Evaluates the Taylor function at a single point. | `evaluation_point` (`array_like`): A 1D array or list representing the point at which to evaluate. | `numpy.ndarray`: A 1-element array containing the result. |
| `neval` | Evaluates the Taylor function at multiple points in a vectorized manner. | `evaluation_points` (`array_like`): A 2D numpy array of shape `(n_points, dimension)`. | `numpy.ndarray` or `torch.Tensor`: An array containing the evaluation result for each input point. |
| `substitute_variable` | Substitutes a variable with a numerical value. | `var_index` (`int`): The 1-based index of the variable to substitute.<br>`value` (`numeric`): The numerical value to substitute. | `mtf`: A new MTF with the variable substituted. |
| `truncate` | Truncates the Taylor series to a specified maximum order. | `order` (`int`, optional): The maximum order to keep. | `mtf`: A new, truncated MTF. |
| `truncate_inplace` | Truncates the MTF **in place** to a specified order. | `order` (`int`, optional): The maximum order to keep. | `mtf`: The same MTF instance, truncated. |
| `substitute_variable_inplace` | Substitutes a variable in the MTF with a numerical value IN-PLACE. | `var_dimension` (`int`): The 1-based index of the variable.<br>`value` (`numeric`): The numerical value. | `None` |
| `is_zero_mtf` | Checks if an MTF is effectively zero. | `mtf` (`mtf`): The MTF to check.<br>`zero_tolerance` (`float`, optional): The tolerance for zero. | `bool`: `True` if the MTF is zero, `False` otherwise. |
| `compose` | Composes this function with other Taylor functions. | `other_function_dict` (`dict[int, mtf]`): A dictionary mapping variable indices (1-based) to the MTF objects to substitute. | `mtf`: A new MTF representing the composed function. |
| `get_tabular_dataframe` | Returns a pandas DataFrame representation of the Taylor function. | `None` | `pandas.DataFrame`: A DataFrame with columns 'Coefficient', 'Order', and 'Exponents'. |
| `extract_coefficient` | Extracts the coefficient for a given multi-index exponent. | `exponents` (`tuple`): A tuple of integers representing the multi-index exponent. | `numpy.ndarray`: A 1-element array containing the coefficient value. |
| `set_coefficient` | Sets the coefficient for a given multi-index exponent. | `exponents` (`tuple`): A tuple of integers for the multi-index exponent.<br>`value` (`numeric`): The new coefficient value. | `None` |
| `get_max_coefficient` | Finds the maximum absolute value among all coefficients. | `None` | `float`: The maximum absolute coefficient value. |
| `get_min_coefficient` | Finds the minimum absolute value among non-negligible coefficients. | `tolerance` (`float`, optional): The tolerance for non-negligible coefficients. | `float`: The minimum absolute non-negligible coefficient value. |
| `symprint` | Converts the MTF to a SymPy expression for pretty printing. | `symbols` (`list` of `str`, optional): A list of symbolic names for the variables.<br>`precision` (`int`, optional): The number of decimal digits for coefficients.<br>`coeff_formatter` (`callable`, optional): A custom function to format coefficients. | `sympy.Expr`: A SymPy expression. |
| `copy` | Returns a copy of the MTF. | `None` | `mtf`: A new MTF object. |
| `get_constant` | Retrieves the constant (zeroth-order) term of the Taylor series. | `None` | `float`: The value of the constant term. |
| `get_polynomial_part` | Returns a new MTF representing the polynomial part of the function. | `None` | `mtf`: A new MTF object with the constant term removed. |
| `from_numpy_array` | Converts a NumPy array of numbers into a NumPy array of mtf objects. | `np_array` (`numpy.ndarray`): The NumPy array to convert.<br>`dimension` (`int`, optional): The dimension for each mtf object. | `numpy.ndarray`: A NumPy array of mtf objects. |
| `to_numpy_array` | Converts a NumPy array of mtf objects into a NumPy array of their constant values. | `mtf_array` (`numpy.ndarray`): The NumPy array of mtf objects to convert. | `numpy.ndarray`: A NumPy array of the constant values. |
| `derivative` | Computes the partial derivative of an MTF. | `deriv_dim` (`int`): The 1-based index of the variable to differentiate with respect to. | `mtf`: A new MTF representing the partial derivative. |
| `integrate` | Performs definite or indefinite integration of an MTF. | `integration_variable_index` (`int`): The 1-based index of the variable to integrate with respect to.<br>`lower_limit` (`float`, optional): The lower limit for definite integration.<br>`upper_limit` (`float`, optional): The upper limit for definite integration. | `mtf`: The integrated MTF. |

## `mtflib.complex_taylor_function` Module

This module provides the `ComplexMultivariateTaylorFunction` for handling complex coefficients.

### `ComplexMultivariateTaylorFunction` Class

A subclass of `MultivariateTaylorFunction` for representing Taylor series with complex coefficients. It inherits all methods from `MultivariateTaylorFunction` and provides the following additional methods.

#### Methods

| Method | Description | Arguments | Returns |
|---|---|---|---|
| `from_constant` | Creates a CMTF representing a constant complex value. | `constant_value` (`complex`, `float`, or `int`): The constant value.<br>`dimension` (`int`, optional): The dimension of the function's domain. | `ComplexMultivariateTaylorFunction`: A new CMTF instance. |
| `from_variable` | Creates a CMTF representing a single variable. | `var_index` (`int`): The 1-based index of the variable.<br>`dimension` (`int`): The total number of variables. | `ComplexMultivariateTaylorFunction`: A new CMTF instance. |
| `conjugate` | Computes the complex conjugate of the Taylor function. | `None` | `ComplexMultivariateTaylorFunction`: A new CMTF representing the complex conjugate. |
| `real_part` | Extracts the real part of the Taylor function. | `None` | `mtf`: A new MTF (with real coefficients) representing the real part. |
| `imag_part` | Extracts the imaginary part of the Taylor function. | `None` | `mtf`: A new MTF (with real coefficients) representing the imaginary part. |

## `mtflib.taylor_map` Module

This module provides the `TaylorMap` class for representing vector-valued functions.

### `TaylorMap` Class

Represents a function from R^n to R^m using Taylor series components.

#### Instance Attributes

| Attribute | Type | Description |
|---|---|---|
| `components` | `numpy.ndarray` | A NumPy array of `MultivariateTaylorFunction` objects that form the components of the map. |
| `map_dim` | `int` | The dimension of the output space (m), which is the number of components. |

#### Methods

| Method | Description | Arguments | Returns |
|---|---|---|---|
| `compose` | Composes this map with another, calculating `self(other(x))`. | `other` (`TaylorMap`): The inner map in the composition. | `TaylorMap`: A new TaylorMap representing the composed function. |
| `get_component` | Retrieves a component function from the map. | `index` (`int`): The 0-based index of the component to retrieve. | `mtf`: The component at the specified index. |
| `get_coefficient` | Gets the coefficient of a specific term in a component function. | `component_index` (`int`): The 0-based index of the component function.<br>`exponent_array` (`numpy.ndarray`): The multi-index exponent of the term. | `float` or `complex`: The value of the coefficient. |
| `set_coefficient` | Sets the coefficient of a specific term in a component function. | `component_index` (`int`): The 0-based index of the component function.<br>`exponent_array` (`numpy.ndarray`): The multi-index exponent of the term.<br>`new_value` (`float` or `complex`): The new value for the coefficient. | `None` |
| `add_component` | Adds a new component function to the map. | `new_component` (`mtf`): The new component to add. | `None` |
| `remove_component` | Removes a component function from the map by its index. | `index` (`int`): The 0-based index of the component to remove. | `None` |
| `truncate` | Truncates all component functions to a specified order. | `order` (`int`): The maximum order for truncation. | `TaylorMap`: A new TaylorMap with the truncated components. |
| `trace` | Calculates the trace of the first-order part of the map. | `None` | `float` or `complex`: The trace value. |
| `map_sensitivity` | Returns a new TaylorMap with coefficients scaled for sensitivity analysis. | `scaling_factors` (`list[float]`): The scaling factors to apply. | `TaylorMap`: A new, scaled TaylorMap. |
| `substitute` | Performs partial or full substitution. | `variable_map` (`dict`): A dict of `{var_index: value}`. | `TaylorMap` or `numpy.ndarray`: A new TaylorMap or a NumPy array of floats. |
| `invert` | Computes the inverse of the TaylorMap using fixed-point iteration. | `None` | `TaylorMap`: A new TaylorMap representing the inverse map. |

## Elementary Functions

These functions are available as static methods on the `MultivariateTaylorFunction` class (e.g., `mtf.sin(...)`).

| Function | Description | Arguments | Returns |
|---|---|---|---|
| `sin` | Computes the Taylor expansion of `sin(x)`. | `variable` (`mtf` or `numeric`): The input function `x`.<br>`order` (`int`, optional): The truncation order. | `mtf`: The Taylor series for `sin(x)`. |
| `cos` | Computes the Taylor expansion of `cos(x)`. | `variable` (`mtf` or `numeric`): The input function `x`.<br>`order` (`int`, optional): The truncation order. | `mtf`: The Taylor series for `cos(x)`. |
| `tan` | Computes the Taylor expansion of `tan(x)`. | `variable` (`mtf` or `numeric`): The input function `x`.<br>`order` (`int`, optional): The truncation order. | `mtf`: The Taylor series for `tan(x)`. |
| `exp` | Computes the Taylor expansion of `exp(x)`. | `variable` (`mtf` or `numeric`): The input function `x`.<br>`order` (`int`, optional): The truncation order. | `mtf`: The Taylor series for `exp(x)`. |
| `gaussian` | Computes the Taylor expansion of a Gaussian function, `exp(-x^2)`. | `variable` (`mtf` or `numeric`): The input function `x`.<br>`order` (`int`, optional): The truncation order. | `mtf`: The Taylor series for `exp(-x^2)`. |
| `log` | Computes the Taylor expansion of `log(x)`. | `variable` (`mtf` or `numeric`): The input function `x`.<br>`order` (`int`, optional): The truncation order. | `mtf`: The Taylor series for `log(x)`. |
| `arctan` | Computes the Taylor expansion of `arctan(x)`. | `variable` (`mtf` or `numeric`): The input function `x`.<br>`order` (`int`, optional): The truncation order. | `mtf`: The Taylor series for `arctan(x)`. |
| `sinh` | Computes the Taylor expansion of `sinh(x)`. | `variable` (`mtf` or `numeric`): The input function `x`.<br>`order` (`int`, optional): The truncation order. | `mtf`: The Taylor series for `sinh(x)`. |
| `cosh` | Computes the Taylor expansion of `cosh(x)`. | `variable` (`mtf` or `numeric`): The input function `x`.<br>`order` (`int`, optional): The truncation order. | `mtf`: The Taylor series for `cosh(x)`. |
| `tanh` | Computes the Taylor expansion of `tanh(x)`. | `variable` (`mtf` or `numeric`): The input function `x`.<br>`order` (`int`, optional): The truncation order. | `mtf`: The Taylor series for `tanh(x)`. |
| `arcsin` | Computes the Taylor expansion of `arcsin(x)`. | `variable` (`mtf` or `numeric`): The input function `x`.<br>`order` (`int`, optional): The truncation order. | `mtf`: The Taylor series for `arcsin(x)`. |
| `arccos` | Computes the Taylor expansion of `arccos(x)`. | `variable` (`mtf` or `numeric`): The input function `x`.<br>`order` (`int`, optional): The truncation order. | `mtf`: The Taylor series for `arccos(x)`. |
| `arctanh` | Computes the Taylor expansion of `arctanh(x)`. | `variable` (`mtf` or `numeric`): The input function `x`.<br>`order` (`int`, optional): The truncation order. | `mtf`: The Taylor series for `arctanh(x)`. |
| `sqrt` | Computes the Taylor expansion of `sqrt(x)`. | `variable` (`mtf` or `numeric`): The input function `x`.<br>`order` (`int`, optional): The truncation order. | `mtf`: The Taylor series for `sqrt(x)`. |
| `isqrt` | Computes the Taylor expansion of `1/sqrt(x)`. | `variable` (`mtf` or `numeric`): The input function `x`.<br>`order` (`int`, optional): The truncation order. | `mtf`: The Taylor series for `1/sqrt(x)`. |

## Utility Functions

| Function | Description | Arguments | Returns |
|---|---|---|---|
| `load_precomputed_coefficients` | Loads or precomputes Taylor coefficients for elementary functions. | `max_order_config` (`int`, optional): The maximum order required for the coefficients. | `dict`: A dictionary of coefficients. |
