import subprocess
import os
import numpy as np
import pandas as pd
import time
import json
import argparse
from sandalwood import mtf

# Mapping of MTF elementary function names to COSY function names
# Updated to match Sandalwood method names or expected COSY names
MTF_TO_COSY_FUNCTIONS = {
    "mtf.sin": "SIN",
    "mtf.cos": "COS",
    "mtf.exp": "EXP",
    "mtf.sqrt": "SQRT",
    "mtf.log": "LOG",
    "mtf.arctan": "ATAN",
    "mtf.tan": "TAN",
    "mtf.arcsin": "ASIN",
    "mtf.arccos": "ACOS",
    "mtf.sinh": "SINH",
    "mtf.cosh": "COSH",
    "mtf.tanh": "TANH",
}

# Local COSY configuration
# We use the COSY executable built in the sandalwood repo
COSY_EXECUTABLE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "sandalwood", "backends", "cosy", "cosy_src", "cosy"))
COSY_WORKING_DIR = os.path.abspath(os.path.dirname(__file__))

def convert_mtf_expression_to_cosy(expression_string):
    """Converts MTF expression string to COSY expression string."""
    expression_string = expression_string.strip()
    if expression_string.startswith('(') and expression_string.endswith(')'):
        balance = 0
        top_level_parentheses = True
        for i, char in enumerate(expression_string):
            if char == '(': balance += 1
            elif char == ')': balance -= 1
            if balance == 0 and i < len(expression_string) - 1: top_level_parentheses = False; break
        if balance == 0 and top_level_parentheses:
            inner_expression = expression_string[1:-1]
            cosy_inner = convert_mtf_expression_to_cosy(inner_expression)
            return f"({cosy_inner})"

    if expression_string in ["x", "y", "z", "u", "v", "w"]: 
        return f"DA({['x', 'y', 'z', 'u', 'v', 'w'].index(expression_string) + 1})"
    
    try: return str(float(expression_string))
    except ValueError: pass

    if '**' in expression_string:
        split_expr = split_expression_by_operator(expression_string, '**')
        if split_expr:
            base_str, exponent_str = split_expr
            return f"POW({convert_mtf_expression_to_cosy(base_str)}, {convert_mtf_expression_to_cosy(exponent_str)})"

    for operator in ['*', '/']:
        if split_expression_by_operator(expression_string, operator):
            left_arg_str, right_arg_str = split_expression_by_operator(expression_string, operator)
            cosy_left = convert_mtf_expression_to_cosy(left_arg_str)
            cosy_right = convert_mtf_expression_to_cosy(right_arg_str)
            return f"({cosy_left} {operator} {cosy_right})" if operator == '*' else f"({cosy_left} / ({cosy_right}))"

    for operator in ['+', '-']:
        if split_expression_by_operator(expression_string, operator):
            left_arg_str, right_arg_str = split_expression_by_operator(expression_string, operator)
            cosy_left = convert_mtf_expression_to_cosy(left_arg_str)
            cosy_right = convert_mtf_expression_to_cosy(right_arg_str)
            return f"({cosy_left} {operator} {cosy_right})"

    if '(' in expression_string and ')' in expression_string:
        func_name = expression_string[:expression_string.find('(')].strip()
        if func_name in MTF_TO_COSY_FUNCTIONS:
            cosy_func_name = MTF_TO_COSY_FUNCTIONS[func_name]
            args_str = expression_string[expression_string.find('(')+1:expression_string.rfind(')')]
            cosy_args = ", ".join([convert_mtf_expression_to_cosy(arg.strip()) for arg in split_arguments(args_str)])
            return f"{cosy_func_name}({cosy_args})"
        else: raise ValueError(f"COSY conversion not implemented for function: {func_name}")

    try: float(expression_string); return expression_string
    except ValueError: pass
    if expression_string in ["x", "y", "z", "u", "v", "w"]: return f"DA({['x', 'y', 'z', 'u', 'v', 'w'].index(expression_string) + 1})"
    raise TypeError(f"Unsupported expression type: '{expression_string}'")

def split_expression_by_operator(expression_string, operator):
    """Splits expression by operator respecting parentheses."""
    balance = 0; split_index = -1
    if operator in ['+', '-']:
        for i, char in enumerate(expression_string):
            if char == '(': balance += 1
            elif char == ')': balance -= 1
            elif char == operator and balance == 0: split_index = i; break
    elif operator in ['**', '*', '/']:
        for i in range(len(expression_string) - 1, -1, -1):
            char = expression_string[i]
            if char == ')': balance += 1
            elif char == '(': balance -= 1
            elif operator == '**' and i + 1 < len(expression_string) and expression_string[i:i+2] == '**' and balance == 0: split_index = i; break
            elif operator != '**' and char == operator and balance == 0: split_index = i; break

    if split_index != -1:
        left_arg_str = expression_string[:split_index].strip()
        right_arg_str = expression_string[split_index+ len(operator):].strip()
        return left_arg_str, right_arg_str
    return None

def split_arguments(arguments_string):
    """Splits function arguments by commas, respecting parentheses."""
    args = []; arg_level = 0; current_arg = ''
    if not arguments_string: return args
    for char in arguments_string:
        if char == ',' and arg_level == 0: args.append(current_arg.strip()); current_arg = ''
        else: current_arg += char;
        if char == '(': arg_level += 1
        elif char == ')': arg_level -= 1
    args.append(current_arg.strip())
    return args

def extract_cosy_coefficients(cosy_output, dimension):
    """Parses COSY output and extracts Taylor expansion coefficients."""
    coefficients = {}
    lines = cosy_output.strip().split('\n')
    start_extracting = False
    for line in lines:
        line = line.strip()
        if line.startswith("I  COEFFICIENT"):
            start_extracting = True
            continue
        if line.startswith("----") and start_extracting:
            break
        if start_extracting and line:
            parts = line.split()
            try:
                coefficient = float(parts[1])
                # order = int(parts[2]) 
                exponents = tuple(int(exp) for exp in parts[3:])
                coefficients[exponents] = coefficient
            except (IndexError, ValueError):
                continue
    return coefficients

def get_mtf_coefficients(mtf_function):
    """Evaluates MTF expression and returns coefficients as a dict."""
    exponents = mtf_function.exponents
    coeffs = mtf_function.coeffs
    return {tuple(exp): c for exp, c in zip(exponents, coeffs)}

def create_raw_cosy_script(function_name, cosy_expression_str, order, dimension, num_iterations=1000):
    """Generates and runs COSY script, returns output and time."""
    script_filename = f"benchmark_raw_{function_name}.fox"
    cosy_script = f"""
INCLUDE 'COSY';

PROCEDURE RUN;
VARIABLE ORDER 1;
VARIABLE DIM 1;
VARIABLE NM1 1;
PROCEDURE TESTEXP NM1;
    VARIABLE TEMP NM1;
    VARIABLE I 1;

    FUNCTION POW R N; VARIABLE I 2; VARIABLE TEMP NM1; TEMP:=1;
        IF N#0; LOOP I 1 N 1; TEMP:=TEMP*R;ENDLOOP; ENDIF;
        IF N=0; TEMP:=1; ENDIF; POW:=TEMP; ENDFUNCTION;

    WRITE 6 '{function_name}';
    LOOP I 1 {num_iterations};
        TEMP:={cosy_expression_str};
    ENDLOOP;
    WRITE 6 TEMP;
ENDPROCEDURE;

ORDER := {order};
DIM := {dimension};

DAINI ORDER DIM 0 NM1;
TESTEXP NM1;

ENDPROCEDURE;

RUN;
END;
"""
    fox_filepath = os.path.join(COSY_WORKING_DIR, script_filename)
    dat_filepath = os.path.join(COSY_WORKING_DIR, "foxyinp.dat")
    
    with open(fox_filepath, "w") as f: f.write(cosy_script)
    with open(dat_filepath, "w") as f_dat: f_dat.write(os.path.splitext(script_filename)[0])

    start_time = time.perf_counter()
    try:
        with open(dat_filepath, 'r') as dat_file:
            process = subprocess.run([COSY_EXECUTABLE_PATH], stdin=dat_file, capture_output=True, text=True, check=True, cwd=COSY_WORKING_DIR)
        execution_time = time.perf_counter() - start_time
        return process.stdout, execution_time
    except Exception as e:
        print(f"Raw COSY execution failed: {e}")
        return None, None

def compare_coefficients(mtf_coefficients, cosy_coefficients):
    """Compares coefficients and calculates RMSE."""
    all_exponents = set(cosy_coefficients.keys()) | set(mtf_coefficients.keys())
    squared_errors = []
    for exponents in all_exponents:
        cosy_coeff = cosy_coefficients.get(exponents, 0.0)
        mtf_coeff = mtf_coefficients.get(exponents, 0.0)
        squared_errors.append((cosy_coeff - mtf_coeff)**2)
    return np.sqrt(np.mean(squared_errors)) if squared_errors else 0.0

def run_benchmarks(expressions, order, dimensions, num_iterations=1000):
    results = []
    variable_names = ["x", "y", "z", "u", "v", "w"][:dimensions]
    
    for name, mtf_expr, cosy_expr in expressions:
        print(f"\n--- Benchmarking {name} ---")
        
        # 1. Sandalwood Python Backend
        mtf.initialize_mtf(max_order=order, max_dimension=dimensions, implementation="python")
        globals_dict = {"mtf": mtf}
        for i in range(dimensions):
            globals_dict[variable_names[i]] = mtf.var(i+1)
        
        start_python = time.perf_counter()
        for _ in range(num_iterations):
            res_python = eval(mtf_expr, globals_dict)
        time_python = time.perf_counter() - start_python
        coeffs_python = get_mtf_coefficients(res_python)
        
        # 2. Sandalwood COSY Backend
        mtf.initialize_mtf(max_order=order, max_dimension=dimensions, implementation="cosy")
        # Re-setup variables for the new backend
        globals_dict = {"mtf": mtf}
        for i in range(dimensions):
            globals_dict[variable_names[i]] = mtf.var(i+1)
        
        start_sand_cosy = time.perf_counter()
        for _ in range(num_iterations):
            res_sand_cosy = eval(mtf_expr, globals_dict)
        time_sand_cosy = time.perf_counter() - start_sand_cosy
        coeffs_sand_cosy = get_mtf_coefficients(res_sand_cosy)
        
        # 3. Raw COSY
        cosy_output, time_raw_cosy = create_raw_cosy_script(name, cosy_expr, order, dimensions, num_iterations)
        coeffs_raw_cosy = extract_cosy_coefficients(cosy_output, dimensions) if cosy_output else {}
        
        # Accuracy Checks
        rmse_sand_python_vs_raw = compare_coefficients(coeffs_python, coeffs_raw_cosy) if coeffs_raw_cosy else np.nan
        rmse_sand_cosy_vs_raw = compare_coefficients(coeffs_sand_cosy, coeffs_raw_cosy) if coeffs_raw_cosy else np.nan
        
        results.append({
            "Function": name,
            "Python (s)": time_python,
            "Sandalwood COSY (s)": time_sand_cosy,
            "Raw COSY (s)": time_raw_cosy,
            "RMSE (Py vs Raw)": rmse_sand_python_vs_raw,
            "RMSE (SCosy vs Raw)": rmse_sand_cosy_vs_raw,
            "Ratio (Py/SCosy)": time_python / time_sand_cosy if time_sand_cosy > 0 else np.nan,
            "Ratio (SCosy/Raw)": time_sand_cosy / time_raw_cosy if time_raw_cosy and time_raw_cosy > 0 else np.nan
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--dims", type=int, default=4)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    benchmark_expressions = [
        ("mul_intensive", "(x + y + z + u)**2", "POW((DA(1)+DA(2)+DA(3)+DA(4)),2)"),
        ("sin_complex", "mtf.sin(0.5 + x + y)", "SIN(0.5 + DA(1) + DA(2))"),
        ("exp_test", "mtf.exp(x - 0.5)", "EXP(DA(1) - 0.5)"),
        ("log_test", "mtf.log(1.0 + x)", "LOG(1.0 + DA(1))"),
        ("sqrt_test", "mtf.sqrt(1.5 + x)", "SQRT(1.5 + DA(1))"),
    ]

    df = run_benchmarks(benchmark_expressions, args.order, args.dims, args.iters)
    print("\n--- Benchmark Summary ---")
    print(df.to_string(index=False))
