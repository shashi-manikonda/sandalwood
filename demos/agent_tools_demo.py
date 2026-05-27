import json
from sandalwood.agent_tools import (
    initialize_sandalwood,
    parse_expression_to_mtf,
    perform_mtf_arithmetic,
    compute_partial_derivative,
    integrate_mtf,
    create_taylor_map,
    invert_taylor_map,
    evaluate_taylor_map,
    evaluate_mtf,
    perform_complex_operation,
    mtf_info,
    taylor_map_info,
)

print("=================================================================")
print("===              Sandalwood Agent Tools Demo                  ===")
print("=================================================================\n")

# 1. Initialize Sandalwood Settings
print("1. Initializing Sandalwood Environment")
print("--------------------------------------")
init_res = initialize_sandalwood.invoke({"max_order": 4, "max_dimension": 2, "implementation": "python"})
print(init_res)
print()

# 2. Parse mathematical expressions to create stateful MTFs
print("2. Parsing and Registering MTFs")
print("-------------------------------")
# f1 = x1**2
f1_res = parse_expression_to_mtf.invoke({
    "expression": "x1**2",
    "dimension": 2,
    "max_order": 4,
    "name": "f1"
})
f1_data = json.loads(f1_res)
print(f"Registered ref: '{f1_data['ref']}'")
print(f"MTF Terms:\n{f1_data['info']}\n")

# f2 = sin(x2)
f2_res = parse_expression_to_mtf.invoke({
    "expression": "sin(x2)",
    "dimension": 2,
    "max_order": 4,
    "name": "f2"
})
f2_data = json.loads(f2_res)
print(f"Registered ref: '{f2_data['ref']}'")
print(f"MTF Terms:\n{f2_data['info']}\n")

# 3. Perform MTF Arithmetic
print("3. MTF Arithmetic (f3 = f1 + f2)")
print("--------------------------------")
f3_res = perform_mtf_arithmetic.invoke({
    "op": "+",
    "mtf_ref_1": "f1",
    "mtf_ref_2": "f2",
    "name": "f3"
})
f3_data = json.loads(f3_res)
print(f"Registered f3 ref: '{f3_data['ref']}'")
print(f"MTF Terms (f(x1, x2) = x1**2 + sin(x2)):\n{f3_data['info']}\n")

# 4. Calculus: Differentiation
print("4. Calculus: Partial Derivative (df/dx1)")
print("----------------------------------------")
deriv_res = compute_partial_derivative.invoke({
    "mtf_ref": "f3",
    "var_index": 1,
    "name": "df3_dx1"
})
deriv_data = json.loads(deriv_res)
print(f"Registered derivative ref: '{deriv_data['ref']}'")
print(f"MTF Terms (df/dx1 = 2*x1):\n{deriv_data['info']}\n")

# 5. Calculus: Definite Integration
print("5. Calculus: Definite Integration (integrate 2*x1 wrt x1 from 1.0 to 3.0)")
print("--------------------------------------------------------------------------")
# definite integral of 2*x1 from 1 to 3: x1**2 | 1 to 3 = 9 - 1 = 8
int_res = integrate_mtf.invoke({
    "mtf_ref": "df3_dx1",
    "var_index": 1,
    "lower_limit": 1.0,
    "upper_limit": 3.0,
    "name": "def_integral"
})
int_data = json.loads(int_res)
print(f"Registered integral ref: '{int_data['ref']}'")
print(f"MTF Terms (Constant 8.0):\n{int_data['info']}")
val = evaluate_mtf.invoke({"mtf_ref": "def_integral", "point": [0.0, 0.0]})
print(f"Evaluated integral value: {val}\n")

# 6. Complex Analysis (CMTF)
print("6. Complex Analysis with CMTF")
print("-----------------------------")
z_res = parse_expression_to_mtf.invoke({
    "expression": "(1 + 3j) * x1",
    "dimension": 2,
    "max_order": 2,
    "name": "z"
})
z_data = json.loads(z_res)
print(f"Registered complex function ref: '{z_data['ref']}'")

# Take the complex conjugate of z
conj_res = perform_complex_operation.invoke({
    "op": "conjugate",
    "mtf_ref": "z",
    "name": "z_conj"
})
conj_data = json.loads(conj_res)
print(f"Registered conjugate ref: '{conj_data['ref']}'")
eval_conj = evaluate_mtf.invoke({"mtf_ref": "z_conj", "point": [1.0, 0.0]})
print(f"Conjugate evaluated at x1=1: {eval_conj}\n")

# 7. Taylor Maps (Vector-Valued Functions) and Inversion
print("7. Coordinate Transformation & Inversion (TaylorMap)")
print("----------------------------------------------------")
# map_A = [x1 + x2**2, x2]
map_res = create_taylor_map.invoke({
    "expressions": ["x1 + x2**2", "x2"],
    "dimension": 2,
    "max_order": 3,
    "name": "map_A"
})
map_data = json.loads(map_res)
print(f"Registered TaylorMap ref: '{map_data['ref']}'")
print(f"Map structure:\n{map_data['info']}")

# Invert coordinate transform (F_inv = [x1 - x2**2, x2])
inv_res = invert_taylor_map.invoke({
    "map_ref": "map_A",
    "name": "map_A_inv"
})
inv_data = json.loads(inv_res)
print(f"Registered Inverse map ref: '{inv_data['ref']}'")
print(f"Inverse map structure:\n{inv_data['info']}")

# 8. Multi-step Map Evaluation Validation
print("8. Multi-Step Validation")
print("------------------------")
start_point = [2.0, 1.0]
# Forward map F([2, 1]) = [2 + 1, 1] = [3, 1]
fwd_val = evaluate_taylor_map.invoke({"map_ref": "map_A", "point": start_point})
print(f"Forward Map evaluated at {start_point} -> {fwd_val}")

# Inverse map F_inv([3, 1]) = [3 - 1, 1] = [2, 1]
bwd_val = evaluate_taylor_map.invoke({"map_ref": "map_A_inv", "point": fwd_val})
print(f"Inverse Map evaluated at {fwd_val} -> {bwd_val}")
print(f"Does F_inv(F(x)) == x? {bwd_val == start_point}")
print("\nDemo Completed Successfully!")
