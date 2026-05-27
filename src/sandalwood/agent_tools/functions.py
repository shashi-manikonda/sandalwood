import json
from typing import Optional, Union

import numpy as np

from sandalwood import MultivariateTaylorFunction, TaylorMap, mtf
from sandalwood.agent_tools.parser import expression_to_mtf
from sandalwood.agent_tools.registry import get_object, register_object


def initialize_sandalwood(
    max_order: int, max_dimension: int, implementation: str = "cosy"
) -> str:
    """
    Explicitly initializes or resets the global Sandalwood settings (maximum truncation order and maximum dimension).
    This must be called before using Taylor operations if you want to override the default limits.

    Args:
        max_order: The maximum order (degree) of the Taylor series terms (e.g. 5).
        max_dimension: The maximum number of variables (e.g. 4 for x1, x2, x3, x4).
        implementation: The backend implementation to use. Must be 'cosy' (high-performance C++ backend) or 'python' (pure Python fallback). Default is 'cosy'.

    Returns:
        A success confirmation message or an error message.
    """
    try:
        if implementation not in ("cosy", "python"):
            raise ValueError("implementation must be either 'cosy' or 'python'")
        mtf.initialize_mtf(
            max_order=max_order,
            max_dimension=max_dimension,
            implementation=implementation,
        )
        return f"Sandalwood successfully initialized with max_order={max_order}, max_dimension={max_dimension}, and implementation='{implementation}'."
    except Exception as e:
        return f"Error initializing Sandalwood: {str(e)}"


def parse_expression_to_mtf(
    expression: str, dimension: int, max_order: int, name: Optional[str] = None
) -> str:
    """
    Parses a string mathematical expression into a MultivariateTaylorFunction (MTF) and registers it in memory.
    Supported functions: sin, cos, tan, exp, log, sinh, cosh, tanh, arcsin, arccos, arctan, arcsinh, arccosh, arctanh, sqrt, erf, cot, coth.
    Variables must be named x1, x2, ..., xN up to the specified dimension.

    Args:
        expression: The math formula (e.g., 'x1**2 + sin(x2)').
        dimension: The number of variables (e.g., 2).
        max_order: The maximum truncation order of the Taylor series.
        name: Optional variable name to store this MTF under in the registry (e.g. 'f1'). If omitted, a name like 'mtf_0' is generated.

    Returns:
        A JSON string containing the registry 'ref' name, the string representation of the MTF, and its JSON schema.
    """
    try:
        func = expression_to_mtf(expression, dimension, max_order)
        ref = register_object(func, name)
        df = func.get_tabular_dataframe()
        return json.dumps(
            {
                "ref": ref,
                "message": f"Successfully parsed and registered as '{ref}'",
                "info": df.to_string(index=False),
                "json": func.to_json(),
            },
            indent=2,
        )
    except Exception as e:
        return f"Error parsing expression to MTF: {str(e)}"


def create_taylor_map(
    expressions: list[str], dimension: int, max_order: int, name: Optional[str] = None
) -> str:
    """
    Creates a vector-valued TaylorMap (mapping R^dimension to R^len(expressions)) from a list of mathematical expressions.
    Each expression defines one component of the output map.

    Args:
        expressions: A list of mathematical expressions as strings, one for each component (e.g., ['x1 + x2', 'x1**2 - x2']).
        dimension: The number of input variables (e.g. 2).
        max_order: The maximum truncation order of the Taylor series.
        name: Optional variable name to store this TaylorMap under in the registry (e.g. 'map_A'). If omitted, a name like 'map_0' is generated.

    Returns:
        A JSON string containing the registry 'ref' name, the detailed summary of the map, and its JSON representation.
    """
    try:
        components = [
            expression_to_mtf(expr, dimension, max_order) for expr in expressions
        ]
        tmap = TaylorMap(components)
        ref = register_object(tmap, name)
        return json.dumps(
            {
                "ref": ref,
                "message": f"Successfully created TaylorMap and registered as '{ref}'",
                "info": str(tmap),
                "json": tmap.to_json(),
            },
            indent=2,
        )
    except Exception as e:
        return f"Error creating TaylorMap: {str(e)}"


def evaluate_taylor_map(map_ref: str, point: list[float]) -> Union[list[float], str]:
    """
    Evaluates a TaylorMap at a specific point in phase space.

    Args:
        map_ref: The registry reference name (e.g. 'map_0') or a raw JSON string representing the TaylorMap.
        point: A list of floats representing the coordinates to evaluate at (length must match map input dimension).

    Returns:
        A list of floats representing the output coordinates, or an error string.
    """
    try:
        tmap = get_object(map_ref, TaylorMap)
        return [comp(point) for comp in tmap.components]
    except Exception as e:
        return f"Error evaluating Taylor Map: {str(e)}"


def evaluate_mtf(mtf_ref: str, point: list[float]) -> Union[float, complex, str]:
    """
    Evaluates a single MultivariateTaylorFunction at a specific point.

    Args:
        mtf_ref: The registry reference name (e.g. 'mtf_0') or a raw JSON string representing the MTF.
        point: A list of floats representing the coordinates to evaluate at (length must match MTF dimension).

    Returns:
        A float or complex number representing the evaluated result, or an error string.
    """
    try:
        func = get_object(mtf_ref, MultivariateTaylorFunction)
        val = func(point)
        if hasattr(val, "item"):
            val = val.item()
        return val
    except Exception as e:
        return f"Error evaluating MTF: {str(e)}"


def invert_taylor_map(map_ref: str, name: Optional[str] = None) -> str:
    """
    Inverts a square TaylorMap (which must have no constant terms and an invertible linear Jacobian part).
    Calculates the inverse mapping using fixed-point iteration.

    Args:
        map_ref: The registry reference name (e.g. 'map_0') or a raw JSON string representing the TaylorMap.
        name: Optional variable name to store the inverted map under in the registry. If omitted, a name like 'map_1' is generated.

    Returns:
        A JSON string containing the registry 'ref' name, a summary of the inverted map, and its JSON representation.
    """
    try:
        tmap = get_object(map_ref, TaylorMap)
        inverted = tmap.invert()
        ref = register_object(inverted, name)
        return json.dumps(
            {
                "ref": ref,
                "message": f"Successfully inverted map and registered as '{ref}'",
                "info": str(inverted),
                "json": inverted.to_json(),
            },
            indent=2,
        )
    except Exception as e:
        return f"Error inverting Taylor Map: {str(e)}"


def compute_partial_derivative(
    mtf_ref: str, var_index: int, name: Optional[str] = None
) -> str:
    """
    Computes the partial derivative of a MultivariateTaylorFunction with respect to a specific variable.

    Args:
        mtf_ref: The registry reference name (e.g. 'mtf_0') or a raw JSON string representing the MTF.
        var_index: The 1-based index of the variable to differentiate with respect to (e.g. 1 for x1, 2 for x2).
        name: Optional variable name to store the derivative under in the registry. If omitted, a name like 'mtf_1' is generated.

    Returns:
        A JSON string containing the registry 'ref' name, a summary of the derivative MTF, and its JSON representation.
    """
    try:
        func = get_object(mtf_ref, MultivariateTaylorFunction)
        derivative = func.derivative(var_index)
        ref = register_object(derivative, name)
        df = derivative.get_tabular_dataframe()
        return json.dumps(
            {
                "ref": ref,
                "message": f"Successfully computed derivative and registered as '{ref}'",
                "info": df.to_string(index=False),
                "json": derivative.to_json(),
            },
            indent=2,
        )
    except Exception as e:
        return f"Error computing partial derivative: {str(e)}"


def integrate_mtf(
    mtf_ref: str,
    var_index: int,
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
    name: Optional[str] = None,
) -> str:
    """
    Integrates a MultivariateTaylorFunction with respect to a specific variable (1-based index).
    Supports indefinite integration, or definite integration if limits are provided.

    Args:
        mtf_ref: The registry reference name (e.g. 'mtf_0') or a raw JSON string representing the MTF.
        var_index: The 1-based index of the variable to integrate with respect to (e.g. 1 for x1).
        lower_limit: Optional float for the lower integration limit.
        upper_limit: Optional float for the upper integration limit.
        name: Optional variable name to store the resulting MTF under in the registry. If omitted, a name like 'mtf_1' is generated.

    Returns:
        A JSON string containing the registry 'ref' name, a summary of the integrated MTF, and its JSON representation.
    """
    try:
        func = get_object(mtf_ref, MultivariateTaylorFunction)
        integrated = func.integrate(
            var_index, lower_limit=lower_limit, upper_limit=upper_limit
        )
        ref = register_object(integrated, name)
        df = integrated.get_tabular_dataframe()
        return json.dumps(
            {
                "ref": ref,
                "message": f"Successfully computed integral and registered as '{ref}'",
                "info": df.to_string(index=False),
                "json": integrated.to_json(),
            },
            indent=2,
        )
    except Exception as e:
        return f"Error integrating MTF: {str(e)}"


def compose_taylor_maps(
    map_ref_1: str, map_ref_2: str, name: Optional[str] = None
) -> str:
    """
    Composes two TaylorMaps, calculating map1(map2(x)).
    The output dimension of map2 must match the input dimension of map1.

    Args:
        map_ref_1: The outer TaylorMap (name or JSON string).
        map_ref_2: The inner TaylorMap (name or JSON string).
        name: Optional variable name to store the composed map under in the registry. If omitted, a name like 'map_1' is generated.

    Returns:
        A JSON string containing the registry 'ref' name, a summary of the composed map, and its JSON representation.
    """
    try:
        map1 = get_object(map_ref_1, TaylorMap)
        map2 = get_object(map_ref_2, TaylorMap)
        composed = map1.compose(map2)
        ref = register_object(composed, name)
        return json.dumps(
            {
                "ref": ref,
                "message": f"Successfully composed maps and registered as '{ref}'",
                "info": str(composed),
                "json": composed.to_json(),
            },
            indent=2,
        )
    except Exception as e:
        return f"Error composing Taylor Maps: {str(e)}"


def compose_mtfs(
    mtf_ref: str, inner_mtf_refs: dict[str, str], name: Optional[str] = None
) -> str:
    """
    Composes a MultivariateTaylorFunction with other MultivariateTaylorFunctions.
    Substitutes specified variables of the outer MTF with inner MTF objects.

    Args:
        mtf_ref: The outer MultivariateTaylorFunction (name or JSON string).
        inner_mtf_refs: A dictionary mapping 1-based variable indices (as strings, e.g. "1") to inner MTFs (names or JSON strings).
        name: Optional variable name to store the composed MTF under in the registry. If omitted, a name like 'mtf_1' is generated.

    Returns:
        A JSON string containing the registry 'ref' name, a summary of the composed MTF, and its JSON representation.
    """
    try:
        outer = get_object(mtf_ref, MultivariateTaylorFunction)
        subs_dict = {}
        for var_str, ref in inner_mtf_refs.items():
            try:
                var_idx = int(var_str)
            except ValueError:
                raise ValueError(
                    f"Variable index keys in inner_mtf_refs must be integers represented as strings (e.g. '1'), got '{var_str}'"
                )
            inner_obj = get_object(ref, MultivariateTaylorFunction)
            subs_dict[var_idx] = inner_obj

        composed = outer.compose(subs_dict)
        ref = register_object(composed, name)
        df = composed.get_tabular_dataframe()
        return json.dumps(
            {
                "ref": ref,
                "message": f"Successfully composed MTFs and registered as '{ref}'",
                "info": df.to_string(index=False),
                "json": composed.to_json(),
            },
            indent=2,
        )
    except Exception as e:
        return f"Error composing MTFs: {str(e)}"


def perform_mtf_arithmetic(
    op: str, mtf_ref_1: str, mtf_ref_2: str, name: Optional[str] = None
) -> str:
    """
    Performs basic arithmetic (+, -, *, /, **) between two MultivariateTaylorFunctions or between an MTF and a scalar.

    Args:
        op: The arithmetic operator. Must be one of: '+', '-', '*', '/', '**'.
        mtf_ref_1: The first operand: an MTF reference (e.g. 'mtf_0') or a raw JSON string.
        mtf_ref_2: The second operand: an MTF reference, a raw JSON string, or a scalar string (e.g. '5.0' or '2+3j').
        name: Optional variable name to store the result under in the registry. If omitted, a name like 'mtf_1' is generated.

    Returns:
        A JSON string containing the registry 'ref' name, a summary of the result, and its JSON representation.
    """
    try:
        op = op.strip()
        if op not in ("+", "-", "*", "/", "**"):
            raise ValueError(
                f"Unsupported operator '{op}'. Must be one of '+', '-', '*', '/', '**'"
            )

        obj1 = get_object(mtf_ref_1, MultivariateTaylorFunction)

        # Try to resolve operand 2 as a scalar or MTF
        ref2_stripped = mtf_ref_2.strip()
        try:
            if "j" in ref2_stripped or "+" in ref2_stripped or "-" in ref2_stripped[1:]:
                obj2 = complex(ref2_stripped)
            else:
                obj2 = float(ref2_stripped)
        except ValueError:
            obj2 = get_object(mtf_ref_2, MultivariateTaylorFunction)

        if op == "+":
            res = obj1 + obj2
        elif op == "-":
            res = obj1 - obj2
        elif op == "*":
            res = obj1 * obj2
        elif op == "/":
            res = obj1 / obj2
        elif op == "**":
            res = obj1**obj2

        ref = register_object(res, name)
        df = res.get_tabular_dataframe()
        return json.dumps(
            {
                "ref": ref,
                "message": f"Successfully performed '{op}' and registered as '{ref}'",
                "info": df.to_string(index=False),
                "json": res.to_json(),
            },
            indent=2,
        )
    except Exception as e:
        return f"Error performing arithmetic: {str(e)}"


def perform_complex_operation(op: str, mtf_ref: str, name: Optional[str] = None) -> str:
    """
    Performs complex analysis operations (conjugate, real_part, imag_part) on a ComplexMultivariateTaylorFunction.

    Args:
        op: The operation to perform. Must be one of: 'conjugate', 'real_part', 'imag_part'.
        mtf_ref: The reference name or JSON representation of a ComplexMultivariateTaylorFunction.
        name: Optional variable name to store the result under in the registry. If omitted, a name like 'mtf_1' is generated.

    Returns:
        A JSON string containing the registry 'ref' name, a summary of the result, and its JSON representation.
    """
    try:
        op = op.strip().lower()
        if op not in ("conjugate", "real_part", "imag_part"):
            raise ValueError("op must be one of: 'conjugate', 'real_part', 'imag_part'")

        func = get_object(mtf_ref, MultivariateTaylorFunction)
        from sandalwood.complex_taylor_function import convert_to_cmtf

        c_func = convert_to_cmtf(func)

        if op == "conjugate":
            res = c_func.conjugate()
        elif op == "real_part":
            res = c_func.real_part()
        elif op == "imag_part":
            res = c_func.imag_part()

        ref = register_object(res, name)
        df = res.get_tabular_dataframe()
        return json.dumps(
            {
                "ref": ref,
                "message": f"Successfully computed '{op}' and registered as '{ref}'",
                "info": df.to_string(index=False),
                "json": res.to_json(),
            },
            indent=2,
        )
    except Exception as e:
        return f"Error performing complex operation: {str(e)}"


def mtf_info(mtf_ref: str) -> str:
    """
    Returns a tabular string representation of a MultivariateTaylorFunction,
    allowing you to inspect its terms, orders, and coefficients.

    Args:
        mtf_ref: The registry reference name (e.g. 'mtf_0') or a raw JSON string representing the MTF.

    Returns:
        A string containing the tabular representation of the function.
    """
    try:
        func = get_object(mtf_ref, MultivariateTaylorFunction)
        df = func.get_tabular_dataframe()
        return df.to_string(index=False)
    except Exception as e:
        return f"Error getting MTF info: {str(e)}"


def taylor_map_info(map_ref: str) -> str:
    """
    Returns a detailed string representation of a TaylorMap, showing
    each components' structure, dimensions, and Taylor series coefficients.

    Args:
        map_ref: The registry reference name (e.g. 'map_0') or a raw JSON string representing the TaylorMap.

    Returns:
        A string containing the details of the TaylorMap.
    """
    try:
        tmap = get_object(map_ref, TaylorMap)
        return str(tmap)
    except Exception as e:
        return f"Error getting TaylorMap info: {str(e)}"


def substitute_variable_in_mtf(
    mtf_ref: str, var_index: int, value: float, name: Optional[str] = None
) -> str:
    """
    Substitutes a single variable (1-based index) in a MultivariateTaylorFunction with a numeric value.

    Args:
        mtf_ref: The registry reference name (e.g. 'mtf_0') or a raw JSON string representing the MTF.
        var_index: The 1-based index of the variable to substitute (e.g., 1 for x1).
        value: The numeric value (float) to substitute in place of the variable.
        name: Optional variable name to store the result under in the registry. If omitted, a name like 'mtf_1' is generated.

    Returns:
        A JSON string containing the registry 'ref' name, a summary of the resulting MTF, and its JSON representation.
    """
    try:
        func = get_object(mtf_ref, MultivariateTaylorFunction)
        res = func.substitute_variable(var_index, value)
        ref = register_object(res, name)
        df = res.get_tabular_dataframe()
        return json.dumps(
            {
                "ref": ref,
                "message": f"Successfully substituted x{var_index}={value} and registered as '{ref}'",
                "info": df.to_string(index=False),
                "json": res.to_json(),
            },
            indent=2,
        )
    except Exception as e:
        return f"Error substituting variable in MTF: {str(e)}"


def substitute_in_taylor_map(
    map_ref: str, variable_map: dict[str, float], name: Optional[str] = None
) -> str:
    """
    Performs partial or full variable substitution on a TaylorMap.

    Args:
        map_ref: The registry reference name (e.g. 'map_0') or a raw JSON string representing the TaylorMap.
        variable_map: A dictionary mapping 1-based variable indices (as strings, e.g. "1") to numeric values (e.g. 1.5).
        name: Optional variable name to store the result under in the registry if the substitution is partial. If omitted, a name like 'map_1' is generated.

    Returns:
        A JSON string containing either the evaluated point (for full substitution) or the registry 'ref' name and details of the new TaylorMap (for partial substitution).
    """
    try:
        tmap = get_object(map_ref, TaylorMap)
        converted_map = {}
        for k, v in variable_map.items():
            try:
                converted_map[int(k)] = float(v)
            except ValueError:
                raise ValueError(
                    f"Variable keys in variable_map must be integers represented as strings (e.g. '1'), got '{k}'"
                )

        res = tmap.substitute(converted_map)

        if isinstance(res, TaylorMap):
            ref = register_object(res, name)
            return json.dumps(
                {
                    "ref": ref,
                    "message": f"Successfully performed partial substitution and registered new TaylorMap as '{ref}'",
                    "info": str(res),
                    "json": res.to_json(),
                },
                indent=2,
            )
        else:
            return json.dumps(
                {
                    "message": "Successfully performed full substitution (evaluation)",
                    "result": list(res),
                },
                indent=2,
            )
    except Exception as e:
        return f"Error substituting in Taylor Map: {str(e)}"


def get_mtf_coefficient(mtf_ref: str, exponents: list[int]) -> str:
    """
    Queries the coefficient of a specific term/exponent tuple in a MultivariateTaylorFunction.

    Args:
        mtf_ref: The registry reference name (e.g. 'mtf_0') or a raw JSON string representing the MTF.
        exponents: A list of integers representing the exponent of each variable (e.g., [2, 0] for x1**2).

    Returns:
        A JSON string containing the queried coefficient value.
    """
    try:
        func = get_object(mtf_ref, MultivariateTaylorFunction)
        val = func.extract_coefficient(tuple(exponents))
        if hasattr(val, "item"):
            val = val.item()

        val_str = str(val)

        return json.dumps(
            {
                "exponents": exponents,
                "coefficient": val_str,
                "numeric_value": val.real if isinstance(val, complex) else val,
            },
            indent=2,
        )
    except Exception as e:
        return f"Error extracting coefficient: {str(e)}"


def truncate_object(ref: str, order: int, name: Optional[str] = None) -> str:
    """
    Truncates a MultivariateTaylorFunction or a TaylorMap component terms to the specified maximum order.

    Args:
        ref: The registry reference name (e.g. 'mtf_0' or 'map_0') or a raw JSON string.
        order: The maximum order (degree) to truncate the polynomial terms to.
        name: Optional variable name to store the result under in the registry. If omitted, a name like 'mtf_1' or 'map_1' is generated.

    Returns:
        A JSON string containing the registry 'ref' name, a summary of the truncated object, and its JSON representation.
    """
    try:
        try:
            obj = get_object(ref, MultivariateTaylorFunction)
            is_mtf = True
        except TypeError:
            obj = get_object(ref, TaylorMap)
            is_mtf = False

        truncated = obj.truncate(order)
        new_ref = register_object(truncated, name)

        if is_mtf:
            info = truncated.get_tabular_dataframe().to_string(index=False)
        else:
            info = str(truncated)

        return json.dumps(
            {
                "ref": new_ref,
                "message": f"Successfully truncated object to order={order} and registered as '{new_ref}'",
                "info": info,
                "json": truncated.to_json(),
            },
            indent=2,
        )
    except Exception as e:
        return f"Error truncating object: {str(e)}"


def analyze_taylor_map(map_ref: str) -> str:
    """
    Analyzes the linear part of a TaylorMap (Jacobian), calculating its trace,
    and checks if it can be inverted.

    Args:
        map_ref: The registry reference name (e.g. 'map_0') or a raw JSON string representing the TaylorMap.

    Returns:
        A JSON string containing the Jacobian trace and invertibility status.
    """
    try:
        tmap = get_object(map_ref, TaylorMap)

        try:
            trace_val = tmap.trace()
            if hasattr(trace_val, "item"):
                trace_val = trace_val.item()
            trace_str = str(trace_val)
        except Exception as te:
            trace_str = (
                f"Trace calculation failed (dimensions might not match): {str(te)}"
            )

        invertible = False
        reason = ""
        try:
            if tmap.map_dim == 0:
                invertible = True
                reason = "Empty map is invertible"
            else:
                dim = tmap.components[0].dimension
                if tmap.map_dim != dim:
                    reason = f"Map is not square (input dim {dim} != output dim {tmap.map_dim})"
                else:
                    zero_exp = tuple([0] * dim)
                    has_const = False
                    for i, component in enumerate(tmap.components):
                        const_term = component.extract_coefficient(zero_exp).item()
                        if abs(const_term) > 1e-14:
                            has_const = True
                            break
                    if has_const:
                        reason = "Map contains non-zero constant terms (must be zero for coordinate inversion)"
                    else:
                        jacobian = np.zeros((dim, dim), dtype=np.complex128)
                        for i in range(dim):
                            for j in range(dim):
                                exp = tuple([1 if k == j else 0 for k in range(dim)])
                                coeff = tmap.get_coefficient(i, np.array(exp))
                                jacobian[i, j] = coeff
                        det = np.linalg.det(jacobian)
                        if abs(det) < 1e-14:
                            reason = f"Linear Jacobian part is singular (det={det})"
                        else:
                            invertible = True
                            reason = "Map is invertible"
        except Exception as ie:
            reason = f"Invertibility check failed: {str(ie)}"

        return json.dumps(
            {
                "trace": trace_str,
                "invertible": invertible,
                "reason": reason,
                "dimensions": {
                    "input_dimension": tmap.components[0].dimension
                    if tmap.map_dim > 0
                    else 0,
                    "output_dimension": tmap.map_dim,
                },
            },
            indent=2,
        )
    except Exception as e:
        return f"Error analyzing Taylor Map: {str(e)}"


def evaluate_mtf_batch(
    mtf_ref: str, points: list[list[float]]
) -> Union[list[float], str]:
    """
    Evaluates a single MultivariateTaylorFunction at a batch of points in phase space.
    Uses Sandalwood's high-performance vectorized evaluation (neval).

    Args:
        mtf_ref: The registry reference name (e.g. 'mtf_0') or a raw JSON string representing the MTF.
        points: A list of coordinate lists (e.g. [[1.0, 2.0], [3.0, 4.0]]) to evaluate at.

    Returns:
        A list of floats containing the evaluated function results, or an error string.
    """
    try:
        func = get_object(mtf_ref, MultivariateTaylorFunction)
        pts_arr = np.array(points, dtype=np.float64)
        res = func.neval(pts_arr)
        return [r.item() for r in res]
    except Exception as e:
        return f"Error batch-evaluating MTF: {str(e)}"


def evaluate_taylor_map_batch(
    map_ref: str, points: list[list[float]]
) -> Union[list[list[float]], str]:
    """
    Evaluates a TaylorMap at a batch of points in phase space.
    Evaluates each component on the list of points, returning a list of output coordinate vectors.

    Args:
        map_ref: The registry reference name (e.g. 'map_0') or a raw JSON string representing the TaylorMap.
        points: A list of coordinate lists (e.g. [[1.0, 2.0], [3.0, 4.0]]) to evaluate at.

    Returns:
        A list of evaluated coordinate vectors (e.g., [[out_x1, out_x2], ...]), or an error string.
    """
    try:
        tmap = get_object(map_ref, TaylorMap)
        pts_arr = np.array(points, dtype=np.float64)

        comp_results = []
        for comp in tmap.components:
            comp_res = comp.neval(pts_arr)
            comp_results.append([r.item() for r in comp_res])

        n_points = len(points)
        n_comps = len(tmap.components)
        transposed = [
            [comp_results[c][p] for c in range(n_comps)] for p in range(n_points)
        ]
        return transposed
    except Exception as e:
        return f"Error batch-evaluating Taylor Map: {str(e)}"


def analyze_mtf_diagnostics(
    mtf_ref: str,
    weight: Optional[list[float]] = None,
    stability_var_id: Optional[int] = None,
    stability_order: Optional[int] = None,
) -> str:
    """
    Computes diagnostics for a MultivariateTaylorFunction, including norm,
    weighted norm, and stability/order decay estimates.

    Args:
        mtf_ref: The registry reference name (e.g. 'mtf_0') or a raw JSON string representing the MTF.
        weight: Optional list of floats for computing weighted norm (COSY backend only).
        stability_var_id: Optional 1-based variable index to estimate stability for (COSY backend only).
        stability_order: Optional maximum order to estimate stability up to (COSY backend only).

    Returns:
        A JSON string containing the diagnostics results.
    """
    try:
        func = get_object(mtf_ref, MultivariateTaylorFunction)
        norm_val = func.norm()

        weighted_norm_val = None
        stability_val = None

        if func._IMPLEMENTATION == "cosy":
            if weight is not None:
                weighted_norm_val = func.weighted_norm(weight)
            if stability_var_id is not None:
                stability_val = func.estimate_stability(
                    var_id=stability_var_id, order=stability_order
                )

        return json.dumps(
            {
                "norm": norm_val,
                "weighted_norm": weighted_norm_val,
                "stability_estimate": stability_val,
                "implementation": func._IMPLEMENTATION,
            },
            indent=2,
        )
    except Exception as e:
        return f"Error analyzing MTF diagnostics: {str(e)}"


def compute_poisson_bracket(
    mtf_ref_1: str, mtf_ref_2: str, name: Optional[str] = None
) -> str:
    """
    Computes the Poisson bracket of two MultivariateTaylorFunctions.

    Args:
        mtf_ref_1: The registry reference name or JSON of the first MTF.
        mtf_ref_2: The registry reference name or JSON of the second MTF.
        name: Optional name for the resulting MTF in the registry.

    Returns:
        A JSON string containing the reference name of the new MTF, or an error string.
    """
    try:
        func1 = get_object(mtf_ref_1, MultivariateTaylorFunction)
        func2 = get_object(mtf_ref_2, MultivariateTaylorFunction)
        result = func1.poisson_bracket(func2)
        ref_name = register_object(result, name)
        return json.dumps(
            {
                "ref": ref_name,
                "message": f"Successfully computed Poisson bracket. Saved as '{ref_name}'.",
            },
            indent=2,
        )
    except Exception as e:
        return f"Error computing Poisson bracket: {str(e)}"


def compute_map_sensitivity(
    map_ref: str, scaling_factors: list[float], name: Optional[str] = None
) -> str:
    """
    Computes the sensitivity of a TaylorMap given a list of scaling factors.
    Returns a new TaylorMap with scaled coefficients.

    Args:
        map_ref: The registry reference name or JSON of the TaylorMap.
        scaling_factors: A list of floats representing the scaling factors (must match input dimension).
        name: Optional name for the resulting TaylorMap in the registry.

    Returns:
        A JSON string containing the reference name of the new TaylorMap, or an error string.
    """
    try:
        tmap = get_object(map_ref, TaylorMap)
        new_map = tmap.map_sensitivity(scaling_factors)
        ref_name = register_object(new_map, name)
        return json.dumps(
            {
                "ref": ref_name,
                "message": f"Successfully computed map sensitivity. Saved as '{ref_name}'.",
            },
            indent=2,
        )
    except Exception as e:
        return f"Error computing map sensitivity: {str(e)}"


def extract_map_component(map_ref: str, index: int, name: Optional[str] = None) -> str:
    """
    Extracts a specific MultivariateTaylorFunction component from a TaylorMap by index.

    Args:
        map_ref: The registry reference name or JSON of the TaylorMap.
        index: The 0-based index of the component to extract.
        name: Optional name for the resulting MTF in the registry.

    Returns:
        A JSON string containing the reference name of the extracted MTF, or an error string.
    """
    try:
        tmap = get_object(map_ref, TaylorMap)
        if index < 0 or index >= tmap.map_dim:
            raise ValueError(
                f"Index {index} out of bounds for TaylorMap with dimension {tmap.map_dim}."
            )
        component = tmap.get_component(index)
        ref_name = register_object(component, name)
        return json.dumps(
            {
                "ref": ref_name,
                "message": f"Successfully extracted component {index}. Saved as '{ref_name}'.",
            },
            indent=2,
        )
    except Exception as e:
        return f"Error extracting map component: {str(e)}"
