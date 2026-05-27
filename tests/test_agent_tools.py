import json

import pytest

from sandalwood import MultivariateTaylorFunction, TaylorMap, mtf
from sandalwood.agent_tools import (
    analyze_mtf_diagnostics,
    analyze_taylor_map,
    clear_registry,
    compose_mtfs,
    compose_taylor_maps,
    compute_map_sensitivity,
    compute_partial_derivative,
    compute_poisson_bracket,
    create_taylor_map,
    evaluate_mtf,
    evaluate_mtf_batch,
    evaluate_taylor_map,
    evaluate_taylor_map_batch,
    expression_to_mtf,
    extract_map_component,
    get_mtf_coefficient,
    get_object,
    initialize_sandalwood,
    integrate_mtf,
    invert_taylor_map,
    mtf_info,
    parse_expression_to_mtf,
    perform_complex_operation,
    perform_mtf_arithmetic,
    register_object,
    substitute_in_taylor_map,
    substitute_variable_in_mtf,
    taylor_map_info,
    truncate_object,
)

try:
    from sandalwood.backends.cosy import COSY_AVAILABLE as _COSY_AVAILABLE
except ImportError:
    _COSY_AVAILABLE = False


@pytest.fixture(autouse=True)
def setup_teardown():
    # Automatically clear registry and initialize sandalwood for each test
    clear_registry()
    MultivariateTaylorFunction._INITIALIZED = False
    mtf.initialize_mtf(max_order=3, max_dimension=3)
    yield
    clear_registry()
    MultivariateTaylorFunction._INITIALIZED = False


def test_registry_basics():
    # Test manual registration
    x = mtf.var(1)
    name = register_object(x, "my_var")
    assert name == "my_var"

    retrieved = get_object("my_var", MultivariateTaylorFunction)
    assert retrieved is x

    # Test auto registration
    auto_name = register_object(x)
    assert auto_name.startswith("mtf_")

    tmap = TaylorMap([x, x])
    map_name = register_object(tmap)
    assert map_name.startswith("map_")

    retrieved_map = get_object(map_name, TaylorMap)
    assert retrieved_map is tmap


def test_registry_json_fallback():
    # Verify that get_object parses raw JSON if passed
    x = mtf.var(1)
    json_str = x.to_json()

    retrieved = get_object(json_str, MultivariateTaylorFunction)
    assert retrieved.dimension == x.dimension
    assert abs(retrieved([1.0, 0.0, 0.0]) - 1.0) < 1e-14


def test_registry_failures():
    # Not found
    with pytest.raises(ValueError, match="not found in registry"):
        get_object("nonexistent", MultivariateTaylorFunction)

    # Invalid JSON syntax
    with pytest.raises(ValueError, match="Failed to parse object from JSON"):
        get_object("{invalid_json}", MultivariateTaylorFunction)

    # Type mismatch
    x = mtf.var(1)
    register_object(x, "my_mtf")
    with pytest.raises(TypeError, match="was expected"):
        get_object("my_mtf", TaylorMap)


def test_parser():
    # Standard expression
    func = expression_to_mtf("x1**2 + sin(x2)", dimension=2, max_order=3)
    assert isinstance(func, MultivariateTaylorFunction)
    # Evaluate at x1=2, x2=0 -> 2**2 + sin(0) = 4
    assert abs(func([2.0, 0.0]) - 4.0) < 1e-7

    # Constant expression
    const_func = expression_to_mtf("5.0", dimension=2, max_order=2)
    assert isinstance(const_func, MultivariateTaylorFunction)
    assert abs(const_func([1.0, 2.0]) - 5.0) < 1e-14

    # Complex constant
    complex_func = expression_to_mtf("3 + 4*I", dimension=2, max_order=2)
    assert abs(complex_func([0.0, 0.0]) - (3 + 4j)) < 1e-14

    # Invalid syntax
    with pytest.raises(ValueError, match="Invalid mathematical expression syntax"):
        expression_to_mtf("x1 + * x2", dimension=2, max_order=2)

    # Invalid variable
    with pytest.raises(ValueError, match="invalid variable"):
        expression_to_mtf("x3 + 1", dimension=2, max_order=2)

    # Unsupported function
    with pytest.raises(
        ValueError, match="Failed to evaluate expression|unsupported function"
    ):
        expression_to_mtf("nonexistent_func(x1)", dimension=2, max_order=2)


def test_initialize_sandalwood_tool():
    MultivariateTaylorFunction._INITIALIZED = False
    res = initialize_sandalwood.invoke({
        "max_order": 4,
        "max_dimension": 2,
        "implementation": "python",
    })
    assert "Sandalwood successfully initialized" in res["data"]
    assert mtf.get_max_order() == 4
    assert mtf.get_max_dimension() == 2

    # Error handling
    err = initialize_sandalwood.invoke({
        "max_order": 4,
        "max_dimension": 2,
        "implementation": "invalid",
    })
    assert err["status"] == "error"


def test_parse_and_create_map_tools():
    # Parse MTF tool
    res = parse_expression_to_mtf.invoke({
        "expression": "x1**2 + x2",
        "dimension": 2,
        "max_order": 2,
        "name": "f1",
    })
    data = res["data"]
    assert data["ref"] == "f1"
    assert "Successfully parsed" in data["message"]

    # Create TaylorMap tool
    map_res = create_taylor_map.invoke({
        "expressions": ["x1 + x2", "x1 - x2"],
        "dimension": 2,
        "max_order": 2,
        "name": "mapA",
    })
    map_data = map_res["data"]
    assert map_data["ref"] == "mapA"
    assert "TaylorMap with 2 components" in map_data["info"]


def test_evaluation_tools():
    # Evaluate TaylorMap
    create_taylor_map.invoke({
        "expressions": ["x1 + x2", "x1 - x2"],
        "dimension": 2,
        "max_order": 2,
        "name": "mapA",
    })
    eval_res = evaluate_taylor_map.invoke({"map_ref": "mapA", "point": [1.5, 0.5]})
    assert eval_res["data"] == [2.0, 1.0]

    # Evaluate MTF
    parse_expression_to_mtf.invoke({
        "expression": "x1*x2",
        "dimension": 2,
        "max_order": 2,
        "name": "f1",
    })
    eval_f1 = evaluate_mtf.invoke({"mtf_ref": "f1", "point": [2.0, 3.0]})
    assert abs(eval_f1["data"] - 6.0) < 1e-14


def test_inversion_and_calculus_tools():
    # Derivatives
    parse_expression_to_mtf.invoke({
        "expression": "x1**3 + x2",
        "dimension": 2,
        "max_order": 3,
        "name": "f1",
    })
    deriv_res = compute_partial_derivative.invoke({
        "mtf_ref": "f1",
        "var_index": 1,
        "name": "df1",
    })
    deriv_data = deriv_res["data"]
    assert deriv_data["ref"] == "df1"
    # df1/dx1 = 3*x1**2. Evaluate at [2.0, 0.0] -> 12
    deriv_val = evaluate_mtf.invoke({"mtf_ref": "df1", "point": [2.0, 0.0]})
    assert abs(deriv_val["data"] - 12.0) < 1e-7

    # Integration
    # Integrate df1 = 3*x1**2 wrt 1 -> x1**3
    int_res = integrate_mtf.invoke({"mtf_ref": "df1", "var_index": 1, "name": "F1"})
    int_data = int_res["data"]
    assert int_data["ref"] == "F1"
    int_val = evaluate_mtf.invoke({"mtf_ref": "F1", "point": [2.0, 0.0]})
    assert abs(int_val["data"] - 8.0) < 1e-7

    # Definite integration (integrate 3*x1**2 from 1 to 2 -> 2**3 - 1**3 = 7)
    def_int_res = integrate_mtf.invoke({
        "mtf_ref": "df1",
        "var_index": 1,
        "lower_limit": 1.0,
        "upper_limit": 2.0,
        "name": "F1_def",
    })
    def_int_data = def_int_res["data"]
    # The result of a definite integral is a constant function
    def_int_val = evaluate_mtf.invoke({"mtf_ref": "F1_def", "point": [0.0, 0.0]})
    assert abs(def_int_val["data"] - 7.0) < 1e-7


def test_inversion_tool():
    # TaylorMap inversion
    # F = [x1 + x2**2, x2]
    # F_inv = [x1 - x2**2, x2]
    create_taylor_map.invoke({
        "expressions": ["x1 + x2**2", "x2"],
        "dimension": 2,
        "max_order": 2,
        "name": "mapF",
    })
    inv_res = invert_taylor_map.invoke({"map_ref": "mapF", "name": "mapF_inv"})
    inv_data = inv_res["data"]
    assert inv_data["ref"] == "mapF_inv"

    eval_point = evaluate_taylor_map.invoke({
        "map_ref": "mapF_inv",
        "point": [2.0, 1.0],
    })
    # F_inv([2, 1]) = [2 - 1, 1] = [1, 1]
    assert abs(eval_point["data"][0] - 1.0) < 1e-7
    assert abs(eval_point["data"][1] - 1.0) < 1e-7


def test_composition_tools():
    # Map Composition
    # map1 = [x1**2, x2]
    # map2 = [x1 + 1, x2] (Wait, compose of maps is self(other(x))
    # map1(map2) = [(x1+1)**2, x2] = [x1**2 + 2*x1 + 1, x2]
    create_taylor_map.invoke({
        "expressions": ["x1**2", "x2"],
        "dimension": 2,
        "max_order": 2,
        "name": "map1",
    })
    create_taylor_map.invoke({
        "expressions": ["x1 + 1", "x2"],
        "dimension": 2,
        "max_order": 2,
        "name": "map2",
    })
    comp_res = compose_taylor_maps.invoke({
        "map_ref_1": "map1",
        "map_ref_2": "map2",
        "name": "map_comp",
    })
    comp_data = comp_res["data"]
    assert comp_data["ref"] == "map_comp"

    eval_val = evaluate_taylor_map.invoke({"map_ref": "map_comp", "point": [1.0, 2.0]})
    assert abs(eval_val["data"][0] - 4.0) < 1e-7  # (1+1)**2
    assert abs(eval_val["data"][1] - 2.0) < 1e-7

    # MTF Composition
    # f = x1**2 + x2
    # g1 = x1 + 1
    # f(g1, x2) = (x1+1)**2 + x2
    parse_expression_to_mtf.invoke({
        "expression": "x1**2 + x2",
        "dimension": 2,
        "max_order": 2,
        "name": "f",
    })
    parse_expression_to_mtf.invoke({
        "expression": "x1 + 1",
        "dimension": 2,
        "max_order": 2,
        "name": "g1",
    })
    comp_mtf_res = compose_mtfs.invoke({
        "mtf_ref": "f",
        "inner_mtf_refs": {"1": "g1"},
        "name": "f_composed",
    })
    comp_mtf_data = comp_mtf_res["data"]
    assert comp_mtf_data["ref"] == "f_composed"

    val = evaluate_mtf.invoke({"mtf_ref": "f_composed", "point": [1.0, 2.0]})
    assert abs(val["data"] - 6.0) < 1e-7  # (1+1)**2 + 2 = 6


def test_arithmetic_tools():
    parse_expression_to_mtf.invoke({
        "expression": "x1",
        "dimension": 2,
        "max_order": 2,
        "name": "f1",
    })
    parse_expression_to_mtf.invoke({
        "expression": "x2",
        "dimension": 2,
        "max_order": 2,
        "name": "f2",
    })

    # f1 + f2
    res = perform_mtf_arithmetic.invoke({
        "op": "+",
        "mtf_ref_1": "f1",
        "mtf_ref_2": "f2",
        "name": "f3",
    })
    data = res["data"]
    assert data["ref"] == "f3"
    val = evaluate_mtf.invoke({"mtf_ref": "f3", "point": [1.5, 2.5]})
    assert abs(val["data"] - 4.0) < 1e-14

    # f1 * 3.0 (scalar)
    res_mul = perform_mtf_arithmetic.invoke({
        "op": "*",
        "mtf_ref_1": "f1",
        "mtf_ref_2": "3.0",
        "name": "f_mul",
    })
    val_mul = evaluate_mtf.invoke({"mtf_ref": "f_mul", "point": [1.5, 2.5]})
    assert abs(val_mul["data"] - 4.5) < 1e-14


def test_complex_operations():
    # CMTF tests
    parse_expression_to_mtf.invoke({
        "expression": "(1 + 2j)*x1",
        "dimension": 2,
        "max_order": 2,
        "name": "z",
    })

    # Conjugate
    conj_res = perform_complex_operation.invoke({
        "op": "conjugate",
        "mtf_ref": "z",
        "name": "z_conj",
    })
    conj_data = conj_res["data"]
    assert conj_data["ref"] == "z_conj"
    val_conj = evaluate_mtf.invoke({"mtf_ref": "z_conj", "point": [1.0, 0.0]})
    assert abs(val_conj["data"] - (1 - 2j)) < 1e-14

    # Real part
    real_res = perform_complex_operation.invoke({
        "op": "real_part",
        "mtf_ref": "z",
        "name": "z_real",
    })
    real_data = real_res["data"]
    assert real_data["ref"] == "z_real"
    val_real = evaluate_mtf.invoke({"mtf_ref": "z_real", "point": [2.0, 0.0]})
    assert abs(val_real["data"] - 2.0) < 1e-14


def test_info_tools():
    parse_expression_to_mtf.invoke({
        "expression": "x1**2",
        "dimension": 2,
        "max_order": 2,
        "name": "f",
    })
    info_str = mtf_info.invoke({"mtf_ref": "f"})
    assert "Exponents" in info_str["data"]
    assert "(2, 0)" in info_str["data"]

    create_taylor_map.invoke({
        "expressions": ["x1", "x2"],
        "dimension": 2,
        "max_order": 2,
        "name": "map",
    })
    map_str = taylor_map_info.invoke({"map_ref": "map"})
    assert "TaylorMap with 2 components" in map_str["data"]


def test_advanced_agent_tools():
    # 1. MTF substitution
    parse_expression_to_mtf.invoke({
        "expression": "x1**2 + x2",
        "dimension": 2,
        "max_order": 2,
        "name": "f",
    })
    sub_res = substitute_variable_in_mtf.invoke({
        "mtf_ref": "f",
        "var_index": 1,
        "value": 3.0,
        "name": "f_sub",
    })
    sub_data = sub_res["data"]
    assert sub_data["ref"] == "f_sub"
    # Result is constant 9 + x2. Evaluate at x2=2 -> 11
    val = evaluate_mtf.invoke({"mtf_ref": "f_sub", "point": [0.0, 2.0]})
    assert abs(val["data"] - 11.0) < 1e-14

    # 2. TaylorMap substitution (partial)
    create_taylor_map.invoke({
        "expressions": ["x1 + x2**2", "x2"],
        "dimension": 2,
        "max_order": 2,
        "name": "map_A",
    })
    sub_map_res = substitute_in_taylor_map.invoke({
        "map_ref": "map_A",
        "variable_map": {"2": 2.0},
        "name": "map_A_sub",
    })
    sub_map_data = sub_map_res["data"]
    assert sub_map_data["ref"] == "map_A_sub"
    # Result map is [x1 + 4, 2]. Evaluate at x1=1 -> [5, 2]
    eval_val = evaluate_taylor_map.invoke({"map_ref": "map_A_sub", "point": [1.0, 0.0]})
    assert abs(eval_val["data"][0] - 5.0) < 1e-14
    assert abs(eval_val["data"][1] - 2.0) < 1e-14

    # 3. TaylorMap substitution (full)
    sub_full_res = substitute_in_taylor_map.invoke({
        "map_ref": "map_A",
        "variable_map": {"1": 1.0, "2": 2.0},
    })
    sub_full_data = sub_full_res["data"]
    assert sub_full_data["result"] == [5.0, 2.0]

    # 4. Get coefficient
    coeff_res = get_mtf_coefficient.invoke({"mtf_ref": "f", "exponents": [2, 0]})
    coeff_data = coeff_res["data"]
    assert coeff_data["exponents"] == [2, 0]
    assert float(coeff_data["coefficient"]) == 1.0

    # 5. Truncate object
    parse_expression_to_mtf.invoke({
        "expression": "x1**3 + x2**2",
        "dimension": 2,
        "max_order": 3,
        "name": "f_high",
    })
    trunc_res = truncate_object.invoke({"ref": "f_high", "order": 2, "name": "f_low"})
    trunc_data = trunc_res["data"]
    assert trunc_data["ref"] == "f_low"
    assert "x1**3" not in trunc_data["info"]  # Should be truncated away

    # 6. Analyze map (trace & invertibility)
    create_taylor_map.invoke({
        "expressions": ["2*x1 + x2**2", "x2"],
        "dimension": 2,
        "max_order": 2,
        "name": "map_non_inv",
    })
    analysis_res = analyze_taylor_map.invoke({"map_ref": "map_non_inv"})
    analysis_data = analysis_res["data"]
    # trace = 2 + 1 = 3
    assert complex(analysis_data["trace"]) == 3.0
    assert analysis_data["invertible"] is True


def test_mcp_resources():
    # Simulate MCP resource read logic directly on registry
    parse_expression_to_mtf.invoke({
        "expression": "x1",
        "dimension": 2,
        "max_order": 2,
        "name": "var1",
    })

    from sandalwood.agent_tools.mcp_server import get_variable, list_variables

    vars_str = list_variables()
    import json
    vars_data = json.loads(vars_str)
    assert "var1" in vars_data
    assert vars_data["var1"]["type"] == "MultivariateTaylorFunction"

    var_detail = get_variable("var1")
    assert "Exponents" in var_detail
    assert "(1, 0)" in var_detail


def test_math_special_functions():
    # Test special math functions added to the parser
    f_gaussian = expression_to_mtf("gaussian(x1)", dimension=1, max_order=3)
    assert abs(f_gaussian([0.0]) - 1.0) < 1e-14

    f_isqrt = expression_to_mtf("isqrt(1.0 + x1)", dimension=1, max_order=2)
    assert abs(f_isqrt([0.0]) - 1.0) < 1e-14

    f_inv_cbrt = expression_to_mtf("inv_cbrt(1.0 + x1)", dimension=1, max_order=2)
    assert abs(f_inv_cbrt([0.0]) - 1.0) < 1e-14

    if _COSY_AVAILABLE:
        f_inv_pow_3_2 = expression_to_mtf(
            "inv_pow_3_2(1.0 + x1)", dimension=1, max_order=2
        )
        assert abs(f_inv_pow_3_2([0.0]) - 1.0) < 1e-14

    # Aliases
    f_asin = expression_to_mtf("asin(x1)", dimension=1, max_order=3)
    assert abs(f_asin([0.0]) - 0.0) < 1e-14

    f_acos = expression_to_mtf("acos(x1)", dimension=1, max_order=3)
    assert abs(f_acos([0.0]) - 1.5707963267948966) < 1e-14

    f_atan = expression_to_mtf("atan(x1)", dimension=1, max_order=3)
    assert abs(f_atan([0.0]) - 0.0) < 1e-14


def test_batch_evaluation_tools():
    # Parse MTF and register it
    parse_expression_to_mtf.invoke({
        "expression": "x1**2 + x2",
        "dimension": 2,
        "max_order": 2,
        "name": "f_batch",
    })

    # Batch evaluate MTF
    pts = [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]
    res = evaluate_mtf_batch.invoke({"mtf_ref": "f_batch", "points": pts})
    assert res["data"] == [3.0, 13.0, 0.0]

    # Create TaylorMap and register it
    create_taylor_map.invoke({
        "expressions": ["x1 + x2", "x1 - x2"],
        "dimension": 2,
        "max_order": 2,
        "name": "map_batch",
    })

    # Batch evaluate TaylorMap
    res_map = evaluate_taylor_map_batch.invoke({"map_ref": "map_batch", "points": pts})
    assert res_map["data"] == [[3.0, -1.0], [7.0, -1.0], [0.0, 0.0]]


def test_analyze_mtf_diagnostics():
    parse_expression_to_mtf.invoke({
        "expression": "3.0*x1 + x2**2",
        "dimension": 2,
        "max_order": 2,
        "name": "f_diag",
    })

    # Analyze diagnostics
    res_diag = analyze_mtf_diagnostics.invoke({"mtf_ref": "f_diag"})
    data = res_diag["data"]
    assert "norm" in data
    assert data["norm"] > 0
    assert "implementation" in data


def test_mcp_prompt():
    from sandalwood.agent_tools.mcp_server import sandalwood_da_expert

    prompt_str = sandalwood_da_expert()
    assert "Differential Algebra" in prompt_str
    assert "MultivariateTaylorFunction" in prompt_str
    assert "evaluate_mtf_batch" in prompt_str


def test_advanced_da_capabilities():
    # 1. Poisson bracket
    # [x1**2, x2] = (dx1**2/dx1)*(dx2/dx2) - (dx1**2/dx2)*(dx2/dx1) = 2*x1 * 1 - 0 = 2*x1
    parse_expression_to_mtf.invoke({
        "expression": "x1**2",
        "dimension": 2,
        "max_order": 2,
        "name": "f_p1",
    })
    parse_expression_to_mtf.invoke({
        "expression": "x2",
        "dimension": 2,
        "max_order": 2,
        "name": "f_p2",
    })
    pb_res = compute_poisson_bracket.invoke({
        "mtf_ref_1": "f_p1",
        "mtf_ref_2": "f_p2",
        "name": "f_pb",
    })

    if _COSY_AVAILABLE:
        pb_data = pb_res["data"]
        assert pb_data["ref"] == "f_pb"
        pb_val = evaluate_mtf.invoke({"mtf_ref": "f_pb", "point": [3.0, 0.0]})
        assert abs(pb_val["data"] - 6.0) < 1e-14
    else:
        assert pb_res["status"] == "error"

    # 2. Map Sensitivity
    create_taylor_map.invoke({
        "expressions": ["2*x1 + x2**2", "3*x2"],
        "dimension": 2,
        "max_order": 2,
        "name": "map_base",
    })
    # Scale by [2.0, 1.0]
    sens_res = compute_map_sensitivity.invoke({
        "map_ref": "map_base",
        "scaling_factors": [2.0, 1.0],
        "name": "map_sens",
    })
    sens_data = sens_res["data"]
    assert sens_data["ref"] == "map_sens"
    sens_val = evaluate_taylor_map.invoke({"map_ref": "map_sens", "point": [1.0, 2.0]})
    # new_map_x1: 2*(2.0*x1) + (1.0*x2)**2 = 4*x1 + x2**2 -> 4(1) + 4 = 8
    # new_map_x2: 3*(1.0*x2) = 3*x2 -> 3(2) = 6
    assert abs(sens_val["data"][0] - 8.0) < 1e-14
    assert abs(sens_val["data"][1] - 6.0) < 1e-14

    # 3. Extract Map Component
    ext_res = extract_map_component.invoke({
        "map_ref": "map_base",
        "index": 0,
        "name": "comp_0",
    })
    ext_data = ext_res["data"]
    assert ext_data["ref"] == "comp_0"
    comp_val = evaluate_mtf.invoke({"mtf_ref": "comp_0", "point": [1.0, 2.0]})
    # 2*(1) + 2**2 = 6
    assert abs(comp_val["data"] - 6.0) < 1e-14
