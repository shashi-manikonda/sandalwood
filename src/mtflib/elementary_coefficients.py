"""
Manages precomputed Taylor series coefficients.

This module is responsible for computing, caching, and loading the Taylor
series coefficients for various elementary functions. The coefficients are
computed up to a specified order (`MAX_PRECOMPUTED_ORDER`) and stored in
JSON files within the `precomputed_coefficients_data` directory.

This precomputation avoids expensive recalculations during runtime,
significantly speeding up the creation of Taylor series for elementary
functions. The `load_precomputed_coefficients` function handles the logic
of loading from files or recomputing if necessary.
"""

import json  # Standard library for JSON
import math
import os
import sys  # Import sys for getsizeof
from typing import Optional

import numpy as np

MAX_PRECOMPUTED_ORDER = 100  # Global maximum order for precomputation
# Directory to store coefficient files
PRECOMPUTED_COEFFICIENT_DIR = "precomputed_coefficients_data"
precomputed_coefficients = {}  # Dictionary to hold loaded coefficients


def _compute_sin_taylor_coefficients(max_order: int) -> np.ndarray:
    """
    Compute Taylor series coefficients for sin(x) around zero.
    """
    coefficients = np.zeros(max_order + 1)
    for n in range(1, max_order + 1, 2):
        coefficients[n] = ((-1) ** ((n - 1) // 2)) / math.factorial(n)
    return coefficients


def _compute_cos_taylor_coefficients(max_order: int) -> np.ndarray:
    """
    Compute Taylor series coefficients for cos(x) around zero.
    """
    coefficients = np.zeros(max_order + 1)
    for n in range(0, max_order + 1, 2):
        coefficients[n] = ((-1) ** (n // 2)) / math.factorial(n)
    return coefficients


def _compute_exp_taylor_coefficients(max_order: int) -> np.ndarray:
    """
    Compute Taylor series coefficients for exp(x) around zero.
    """
    coefficients = np.zeros(max_order + 1)
    for n in range(0, max_order + 1):
        coefficients[n] = 1.0 / math.factorial(n)
    return coefficients


def _compute_gaussian_taylor_coefficients(max_order: int) -> np.ndarray:
    """
    Compute Taylor series coefficients for gaussian(x) = exp(-x^2) around zero.
    Note: Gaussian series has only even terms.
    """
    coefficients = np.zeros(max_order + 1)
    for n in range(0, max_order + 1, 2):  # Gaussian has only even terms
        k = n // 2
        coefficients[n] = (-1) ** k / math.factorial(k)
    return coefficients


def _compute_sqrt_taylor_coefficients(max_order: int) -> np.ndarray:
    """
    Compute Taylor series coefficients for sqrt(1+x) around zero.
    """
    coefficients = np.zeros(max_order + 1)
    for n in range(0, max_order + 1):
        if n == 0:
            coefficients[n] = 1.0
        elif n == 1:
            coefficients[n] = 0.5
        else:
            coefficients[n] = coefficients[n - 1] * (0.5 - (n - 1)) / n
    return coefficients


def _compute_isqrt_taylor_coefficients(max_order: int) -> np.ndarray:
    """
    Compute Taylor series coefficients for isqrt(1+x) = 1/sqrt(1+x) around zero.
    """
    coefficients = np.zeros(max_order + 1)
    for n in range(0, max_order + 1):
        if n == 0:
            coefficients[n] = 1.0
        elif n == 1:
            coefficients[n] = -0.5
        else:
            coefficients[n] = coefficients[n - 1] * (-0.5 - (n - 1)) / n
    return coefficients


def _compute_log_taylor_coefficients(max_order: int) -> np.ndarray:
    """
    Compute Taylor series coefficients for log(1+x) around zero.
    """
    coefficients = np.zeros(max_order + 1)
    for n in range(0, max_order + 1):
        if n == 0:
            coefficients[n] = 0.0
        elif n >= 1:
            coefficients[n] = ((-1) ** (n - 1)) / n
    return coefficients


def _compute_arctan_taylor_coefficients(max_order: int) -> np.ndarray:
    """
    Compute Taylor series coefficients for arctan(x) around zero.
    Note: Arctan series has only odd terms.
    """
    coefficients = np.zeros(max_order + 1)
    for n in range(0, max_order + 1):
        if n == 0:
            coefficients[n] = 0.0
        elif n % 2 != 0:  # Arctan series has only odd terms
            term_index = (n - 1) // 2
            coefficients[n] = ((-1) ** term_index) / n
        else:
            coefficients[n] = 0.0
    return coefficients


def _compute_sinh_taylor_coefficients(max_order: int) -> np.ndarray:
    """
    Compute Taylor series coefficients for sinh(x) around zero.
    Note: Sinh series has only odd terms.
    """
    coefficients = np.zeros(max_order + 1)
    for n in range(0, max_order + 1):
        if n % 2 != 0:  # Sinh series has only odd terms
            coefficients[n] = 1.0 / math.factorial(n)
        else:
            coefficients[n] = 0.0
    return coefficients


def _compute_cosh_taylor_coefficients(max_order: int) -> np.ndarray:
    """
    Compute Taylor series coefficients for cosh(x) around zero.
    Note: Cosh series has only even terms.
    """
    coefficients = np.zeros(max_order + 1)
    for n in range(0, max_order + 1):
        if n % 2 == 0:  # Cosh series has only even terms
            coefficients[n] = 1.0 / math.factorial(n)
        else:
            coefficients[n] = 0.0
    return coefficients


def _compute_arctanh_taylor_coefficients(max_order: int) -> np.ndarray:
    """
    Compute Taylor series coefficients for arctanh(x) around zero.
    Note: Arctanh series has only odd terms.
    """
    coefficients = np.zeros(max_order + 1)
    for n in range(0, max_order + 1):
        if n == 0:
            coefficients[n] = 0.0
        elif n % 2 != 0:  # Arctanh series has only odd terms
            coefficients[n] = 1.0 / float(n)
        else:
            coefficients[n] = 0.0
    return coefficients


def _compute_arcsin_taylor_coefficients(max_order: int) -> np.ndarray:
    """
    Compute Taylor series coefficients for arcsin(x) around zero.
    Note: Arcsin series has only odd terms.
    """
    coefficients = np.zeros(max_order + 1)
    for n in range(0, max_order + 1):
        if n == 0:
            coefficients[n] = 0.0
        elif n == 1:
            coefficients[n] = 1.0
        elif n % 2 == 0:  # Even terms are zero
            coefficients[n] = 0.0
        elif n % 2 != 0:  # Odd terms
            m = (n - 1) // 2
            numerator = 1.0
            for i in range(m):
                numerator *= 2 * i + 1
            denominator = 1.0
            for i in range(1, m + 1):
                denominator *= 2 * i
            coefficients[n] = (numerator / denominator) * (1 / n)
    return coefficients


def _compute_arccos_taylor_coefficients(max_order: int) -> np.ndarray:
    """
    Compute Taylor series coefficients for arccos(x) around zero.
    Note: Arccos series has only odd terms (except the constant term).
    """
    coefficients = np.zeros(max_order + 1)
    for n in range(0, max_order + 1):
        if n == 0:
            coefficients[n] = math.pi / 2.0  # arccos(0) = pi/2
        elif n == 1:
            coefficients[n] = -1.0
        elif n % 2 == 0:  # Even terms are zero (except n=0)
            coefficients[n] = 0.0
        elif n % 2 != 0:  # Odd terms (for n >= 3)
            m = (n - 1) // 2
            numerator = 1.0
            for i in range(m):
                numerator *= 2 * i + 1
            denominator = 1.0
            for i in range(1, m + 1):
                denominator *= 2 * i
            # Note the negative sign for arccos
            coefficients[n] = -(numerator / denominator) * (1 / n)
    return coefficients


def _compute_inverse_taylor_coefficients(max_order: int) -> np.ndarray:
    """
    Compute Taylor series coefficients for inverse(1+x) = 1/(1+x) around zero.
    """
    coefficients = np.zeros(max_order + 1)
    for n in range(0, max_order + 1):
        coefficients[n] = (-1) ** n
    return coefficients


coefficient_functions = {
    "sin": _compute_sin_taylor_coefficients,
    "cos": _compute_cos_taylor_coefficients,
    "exp": _compute_exp_taylor_coefficients,
    "gaussian": _compute_gaussian_taylor_coefficients,
    "sqrt": _compute_sqrt_taylor_coefficients,
    "isqrt": _compute_isqrt_taylor_coefficients,
    "log": _compute_log_taylor_coefficients,
    "arctan": _compute_arctan_taylor_coefficients,
    "sinh": _compute_sinh_taylor_coefficients,
    "cosh": _compute_cosh_taylor_coefficients,
    "arctanh": _compute_arctanh_taylor_coefficients,
    "arcsin": _compute_arcsin_taylor_coefficients,
    "arccos": _compute_arccos_taylor_coefficients,
    "inverse": _compute_inverse_taylor_coefficients,  # NEW: Added inverse function
}


def load_precomputed_coefficients(max_order_config: Optional[int] = None) -> dict:
    """
    Loads or precomputes Taylor coefficients for elementary functions.

    This function checks for cached coefficients in JSON files. If a file
    exists and its order is sufficient, the coefficients are loaded from it.
    Otherwise, they are recomputed, saved to a JSON file, and then loaded.
    The loaded coefficients are stored in the global `precomputed_coefficients`
    dictionary.

    Parameters
    ----------
    max_order_config : int, optional
        The maximum order required for the coefficients. If not provided,
        the module's `MAX_PRECOMPUTED_ORDER` is used.

    Returns
    -------
    dict
        A dictionary where keys are function names (e.g., 'sin') and
        values are NumPy arrays of their Taylor coefficients up to the
        requested order.

    Examples
    --------
    This function is typically called internally by
    `MultivariateTaylorFunction.initialize_mtf`.

    >>> from mtflib.elementary_coefficients import load_precomputed_coefficients
    >>> # Load coefficients up to order 5
    >>> coeffs = load_precomputed_coefficients(max_order_config=5) # doctest: +ELLIPSIS
    Loading/Precomputing Taylor coefficients up to order 5...
    Global precomputed coefficients loading/generation complete.
    ...
    >>> # Get the precomputed coefficients for sin
    >>> sin_coeffs = coeffs.get('sin')
    >>> print(sin_coeffs)
    [ 0.          1.          0.         -0.16666667  0.          0.00833333]
    """

    if max_order_config is None:
        max_order_to_init = MAX_PRECOMPUTED_ORDER
    else:
        max_order_to_init = max_order_config

    print(f"Loading/Precomputing Taylor coefficients up to order {max_order_to_init}")

    # Construct the path to the directory relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_coefficient_dir = os.path.join(base_dir, PRECOMPUTED_COEFFICIENT_DIR)

    if not os.path.exists(full_coefficient_dir):  # Use full path
        os.makedirs(full_coefficient_dir)

    for func_name, compute_func in coefficient_functions.items():
        filename = os.path.join(
            full_coefficient_dir, f"{func_name}_coefficients.json"
        )  # Use full path
        if os.path.exists(filename):
            try:
                with open(filename, "r") as f:
                    coefficient_list = json.load(f)
                    loaded_coefficients = np.array(coefficient_list)
                if loaded_coefficients.shape[0] < max_order_to_init + 1:
                    print(
                        f"Precomputed {func_name} coefficients "
                        f"(order {loaded_coefficients.shape[0] - 1}) insufficient, "
                        f"recomputing up to order {max_order_to_init}."
                    )
                    coefficients = compute_func(max_order_to_init)
                    try:
                        with open(filename, "w") as f:
                            json.dump(coefficients.tolist(), f, indent=2)
                        print(
                            f"Recomputed and saved {func_name} coefficients "
                            f"up to order {max_order_to_init}."
                        )
                    except Exception as e:
                        print(
                            f"Error saving recomputed coefficients for {func_name} "
                            f"to {filename}: {e}"
                        )
                    precomputed_coefficients[func_name] = coefficients
                else:
                    coefficients = loaded_coefficients[: max_order_to_init + 1]
                    precomputed_coefficients[func_name] = coefficients
                    # print(
                    #     f"Loaded precomputed {func_name} coefficients up to order "
                    #     f"{max_order_to_init} from {filename}."
                    # )

            except Exception as e:
                print(
                    f"Warning: Error loading coefficients for {func_name} "
                    f"from {filename}: {e}. Recomputing..."
                )
                coefficients = compute_func(max_order_to_init)
                try:
                    with open(filename, "w") as f:
                        json.dump(coefficients.tolist(), f, indent=2)
                    print(
                        f"Computed and saved {func_name} coefficients "
                        f"up to order {max_order_to_init}."
                    )
                except Exception as save_e:
                    print(
                        f"Error saving recomputed coefficients for {func_name} "
                        f"to {filename}: {save_e}"
                    )
                precomputed_coefficients[func_name] = coefficients

        else:  # File does not exist
            print(
                f"Precomputing {func_name} Taylor coefficients up to order "
                f"{max_order_to_init} and saving to {filename}"
            )
            coefficients = compute_func(max_order_to_init)
            try:
                with open(filename, "w") as f:
                    json.dump(coefficients.tolist(), f, indent=2)
                print(
                    f"Computed and saved {func_name} coefficients up to order "
                    f"{max_order_to_init}."
                )
            except Exception as e:
                print(f"Error saving coefficients for {func_name} to {filename}: {e}")
            precomputed_coefficients[func_name] = coefficients

    # for func_name in coefficient_functions.keys():  # Verification print
    #     print(f"First 5 {func_name} coefficients: "
    #           f"{precomputed_coefficients[func_name][:5]}")

    print("Global precomputed coefficients loading/generation complete.")
    size_in_bytes = sys.getsizeof(precomputed_coefficients)
    size_in_kb = size_in_bytes / 1024
    size_in_mb = size_in_kb / 1024
    print(
        f"Size of precomputed_coefficients dictionary in memory: {size_in_bytes} "
        f"bytes, {size_in_kb:.2f} KB, {size_in_mb:.2f} MB"
    )

    return precomputed_coefficients


if __name__ == "__main__":
    load_precomputed_coefficients()  # Load with default MAX_PRECOMPUTED_ORDER

    print("\n--- Precomputed Taylor Coefficients (First Few Terms) ---")
    for function_name in sorted(coefficient_functions.keys()):
        coeffs = precomputed_coefficients.get(function_name, np.array([]))
        # Display up to order 5 or less if available
        display_order = min(5, coeffs.shape[0] - 1) if coeffs.shape[0] > 0 else 0
        if display_order >= 0:
            display_coeffs = ", ".join([
                f"{c:.6f}" for c in coeffs[: display_order + 1]
            ])  # Format for display
        else:
            display_coeffs = "Not computed/loaded"

        print(f"{function_name}: Orders 0-{display_order}: [{display_coeffs}...]")

    print("\nPrecomputation and Saving Complete.")
