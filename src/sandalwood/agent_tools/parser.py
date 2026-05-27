import sympy
from sympy import E, oo, pi
from sympy.parsing.sympy_parser import parse_expr

from sandalwood import MultivariateTaylorFunction, mtf


def expression_to_mtf(
    expression: str, dimension: int, max_order: int
) -> MultivariateTaylorFunction:
    """
    Parses a string mathematical expression into a sandalwood MultivariateTaylorFunction.

    Args:
        expression: The mathematical expression as a string (e.g., 'x1**2 + sin(x2)').
        dimension: The maximum number of variables (e.g. 2 for x1, x2).
        max_order: The maximum truncation order for the Taylor series.

    Returns:
        MultivariateTaylorFunction: The resulting MTF object.

    Raises:
        ValueError: If syntax is invalid, refers to variables outside the dimension limits,
                    or uses unsupported functions.
    """
    # 1. Initialize Sandalwood global settings if not initialized, or check compatibility
    if not MultivariateTaylorFunction.get_mtf_initialized_status():
        mtf.initialize_mtf(max_order=max_order, max_dimension=dimension)
    else:
        current_order = mtf.get_max_order()
        current_dim = mtf.get_max_dimension()
        # If requested settings exceed current initialization, trigger re-initialization
        # so the library can check and raise the appropriate RuntimeError.
        if dimension > current_dim or max_order > current_order:
            mtf.initialize_mtf(max_order=max_order, max_dimension=dimension)

    # 2. Syntax check and parse with SymPy
    try:
        expr = parse_expr(expression)
    except Exception as e:
        raise ValueError(f"Invalid mathematical expression syntax: {str(e)}")

    # 3. Check for valid symbols
    allowed_vars = {f"x{i}" for i in range(1, dimension + 1)}
    expr_free_symbols = {str(sym) for sym in expr.free_symbols}
    invalid_symbols = expr_free_symbols - allowed_vars
    if invalid_symbols:
        raise ValueError(
            f"Expression contains invalid variable(s): {sorted(list(invalid_symbols))}. "
            f"For dimension={dimension}, only variables {sorted(list(allowed_vars))} are allowed."
        )

    # 4. Construct variable dict and symbols
    # Explicitly pass dimension to mtf.var so the resulting MTF has the requested dimension
    var_dict = {
        f"x{i}": mtf.var(i, dimension=dimension) for i in range(1, dimension + 1)
    }
    symbols = [sympy.Symbol(f"x{i}") for i in range(1, dimension + 1)]

    # 5. Map SymPy string functions to MTF instance methods
    math_dict = {
        "pi": pi,
        "E": E,
        "e": E,
        "oo": oo,
        "sin": lambda x: x.sin(),
        "cos": lambda x: x.cos(),
        "tan": lambda x: x.tan(),
        "exp": lambda x: x.exp(),
        "log": lambda x: x.log(),
        "sinh": lambda x: x.sinh(),
        "cosh": lambda x: x.cosh(),
        "tanh": lambda x: x.tanh(),
        "sqrt": lambda x: x.sqrt(),
        "arcsin": lambda x: x.arcsin(),
        "arccos": lambda x: x.arccos(),
        "arctan": lambda x: x.arctan(),
        "arcsinh": lambda x: x.arcsinh(),
        "arccosh": lambda x: x.arccosh(),
        "arctanh": lambda x: x.arctanh(),
        "erf": lambda x: x.erf(),
        "cot": lambda x: x.cot(),
        "coth": lambda x: x.coth(),
        "gaussian": lambda x: x.gaussian(),
        "isqrt": lambda x: x.isqrt(),
        "inv_cbrt": lambda x: x.inv_cbrt(),
        "inv_pow_3_2": lambda x: x.inv_pow_3_2(),
        "asin": lambda x: x.arcsin(),
        "acos": lambda x: x.arccos(),
        "atan": lambda x: x.arctan(),
    }

    # 6. Evaluate and convert to MTF using lambdify
    try:
        f = sympy.lambdify(symbols, expr, modules=[math_dict, "math"])
        result = f(*[var_dict[f"x{i}"] for i in range(1, dimension + 1)])
        return MultivariateTaylorFunction.to_mtf(result, dimension)
    except AttributeError as e:
        raise ValueError(
            f"Expression contains an unsupported function or operation: {str(e)}"
        )
    except Exception as e:
        raise ValueError(
            f"Failed to evaluate expression into a MultivariateTaylorFunction: {str(e)}"
        )
