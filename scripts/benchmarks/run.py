import argparse
import sys
import os
import time
import pandas as pd
import numpy as np
from core import BenchmarkEngine, ARTIFACTS_DIR

# Ensure we can import sandalwood from the parent src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from sandalwood import TaylorMap, mtf

FULL_OPS = [
    # Arithmetic
    ("Add", "x + y", "DA(1)+DA(2)"),
    ("Sub", "x - y", "DA(1)-DA(2)"),
    ("Mul", "x * y", "DA(1)*DA(2)"),
    ("Div", "(1+x)/(1+y)", "(1+DA(1))/(1+DA(2))"),
    ("Pow", "(1+x)**3", "(1+DA(1))**3"),
    # Elementary Functions
    ("Sin", "mtf.sin(x)", "SIN(DA(1))"),
    ("Cos", "mtf.cos(x)", "COS(DA(1))"),
    ("Tan", "mtf.tan(x)", "TAN(DA(1))"),
    ("Exp", "mtf.exp(x)", "EXP(DA(1))"),
    ("Log", "mtf.log(1+x)", "LOG(1+DA(1))"),
    ("Sqrt", "mtf.sqrt(1+x)", "SQRT(1+DA(1))"),
    ("Asin", "mtf.arcsin(0.5*x)", "ASIN(0.5*DA(1))"),
    ("Acos", "mtf.arccos(0.5*x)", "ACOS(0.5*DA(1))"),
    ("Atan", "mtf.arctan(x)", "ATAN(DA(1))"),
    ("Sinh", "mtf.sinh(x)", "SINH(DA(1))"),
    ("Cosh", "mtf.cosh(x)", "COSH(DA(1))"),
    ("Tanh", "mtf.tanh(x)", "TANH(DA(1))"),
]

def run_ops_benchmark(engine, args):
    """Benchmarks individual operations (Python vs COSY Backend)."""
    ops = [
        ("Add", "x + y", "DA(1)+DA(2)"),
        ("Mul", "x * y", "DA(1)*DA(2)"),
        ("Pow", "(x+y)**3", "(DA(1)+DA(2))*(DA(1)+DA(2))*(DA(1)+DA(2))"),
        ("Sin", "mtf.sin(x)", "SIN(DA(1))"),
        ("Exp", "mtf.exp(x)", "EXP(DA(1))"),
        ("Log", "mtf.log(1+x)", "LOG(1+DA(1))"),
    ]
    
    results = []
    for name, mtf_expr, cosy_expr in ops:
        print(f"Benchmarking {name}...")
        c_py, t_py = engine.run_sandalwood(mtf_expr, "python", args.iters)
        c_sc, t_sc = engine.run_sandalwood(mtf_expr, "cosy", args.iters)
        
        results.append({
            "Operation": name,
            "Python Time": engine.format_time(t_py),
            "Sandalwood COSY Time": engine.format_time(t_sc),
            "Speedup": engine.format_speedup(t_py / t_sc if t_sc > 0 else 0)
        })
        
    df = pd.DataFrame(results)
    if args.json:
        print(df.to_json(orient='records'))
    else:
        print("\n" + df.to_string(index=False))
    engine.save_markdown_results(df, "Operation Benchmarks")

def run_raw_comparison(engine, args):
    """Compares Sandalwood (Python/COSY) with Raw COSY execution."""
    cases = [
        ("mul_intensive", "(x + y + z + u)**2", "(DA(1)+DA(2)+DA(3)+DA(4))*(DA(1)+DA(2)+DA(3)+DA(4))"),
        ("sin_complex", "mtf.sin(0.5 + x + y)", "SIN(0.5 + DA(1) + DA(2))"),
        ("exp_test", "mtf.exp(x - 0.5)", "EXP(DA(1) - 0.5)"),
    ]
    
    results = []
    for name, mtf_expr, cosy_expr in cases:
        print(f"Comparing {name} with Raw COSY...")
        c_py, t_py = engine.run_sandalwood(mtf_expr, "python", args.iters)
        c_sc, t_sc = engine.run_sandalwood(mtf_expr, "cosy", args.iters)
        c_raw, t_raw = engine.run_raw_cosy(name, cosy_expr, args.iters)
        
        rmse_py = engine.calculate_rmse(c_py, c_raw)
        rmse_sc = engine.calculate_rmse(c_sc, c_raw)
        
        results.append({
            "Case": name,
            "Python Time": engine.format_time(t_py),
            "SCosy Time": engine.format_time(t_sc),
            "Raw COSY Time": engine.format_time(t_raw),
            "RMSE (Py vs Raw)": f"{rmse_py:.2e}",
            "RMSE (SCosy vs Raw)": f"{rmse_sc:.2e}",
            "Speedup (Py/SCosy)": engine.format_speedup(t_py / t_sc if t_sc > 0 else 0)
        })
        
    df = pd.DataFrame(results)
    if args.json:
        print(df.to_json(orient='records'))
    else:
        print("\n" + df.to_string(index=False))
    engine.save_markdown_results(df, "Raw COSY Comparison")

def run_batch_eval(engine, args):
    """Benchmarks batch evaluation performance."""
    print(f"Benchmarking batch evaluation with {args.npoints} points...")
    globals_dict = engine.setup_mtf("cosy")
    import numpy as np
    
    # Create a reasonably complex polynomial
    x, y, z = globals_dict['x'], globals_dict['y'], globals_dict['z']
    poly = (x + y + z + 1.0)**3
    
    points = np.random.rand(args.npoints, engine.dimension)
    
    # Batch
    start = time.perf_counter()
    res_batch = poly.neval(points)
    t_batch = time.perf_counter() - start
    
    # Serial (subset)
    n_sample = min(args.npoints, 100)
    start = time.perf_counter()
    for i in range(n_sample):
        poly.eval(points[i])
    t_serial_avg = (time.perf_counter() - start) / n_sample
    
    t_serial_est = t_serial_avg * args.npoints
    
    results = [{
        "Points": args.npoints,
        "Batch Total": engine.format_time(t_batch),
        "Serial Est": engine.format_time(t_serial_est),
        "Speedup": engine.format_speedup(t_serial_est / t_batch if t_batch > 0 else 0)
    }]
    
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    engine.save_markdown_results(df, "Batch Evaluation Benchmarks")

def run_profile(engine, args):
    """Runs a cProfile on core operations."""
    import cProfile, pstats, io
    print("Running cProfile on complex arithmetic and mapping...")
    
    globals_dict = engine.setup_mtf("cosy")
    from sandalwood import TaylorMap
    x, y, z, u = globals_dict['x'], globals_dict['y'], globals_dict['z'], globals_dict['u']
    
    pr = cProfile.Profile()
    pr.enable()
    
    # Operations
    _ = (x + y + z + u) ** 3 * mtf.exp(x)
    
    # Composition
    m1 = TaylorMap([x + 0.1 * y * y, y + 0.1 * x, z, u])
    m2 = TaylorMap([x, y + 0.1 * z * z, z + 0.1 * y, u])
    _ = m1.compose(m2)
    
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())
    
    # Save text result as MD code block
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(ARTIFACTS_DIR, f"profile_{timestamp}.md")
    with open(filepath, "w") as f:
        f.write(f"# Profiling Results - {timestamp}\n\n```\n{s.getvalue()}\n```\n")
    print(f"Profile saved to {filepath}")

import json
import subprocess

def run_full_benchmark(args):
    """Performs a comprehensive parametric sweep across orders and variables."""
    orders = [2, 4, 6, 8, 10]
    vars_list = [4, 6]
    iters = args.iters if args.iters != 100 else 10
    
    full_results = []
    
    python_bin = sys.executable
    script_path = __file__
    
    for v in vars_list:
        for o in orders:
            print(f"\n>>> Sweep: Variables={v}, Order={o} <<<")
            
            # Use subprocess to run the ops and raw comparisons for this (v, o)
            # Ops
            cmd_ops = [python_bin, script_path, "--mode", "ops", "--order", str(o), "--dims", str(v), "--iters", str(iters), "--json"]
            try:
                res_ops = subprocess.run(cmd_ops, capture_output=True, text=True, check=True)
                # JSON might be mixed with other output (initialization prints)
                # We search for the JSON part (starting with [ and ending with ])
                out = res_ops.stdout
                json_part = out[out.find('['):out.rfind(']')+1]
                ops_data = json.loads(json_part)
                
                # Raw
                cmd_raw = [python_bin, script_path, "--mode", "raw", "--order", str(o), "--dims", str(v), "--iters", str(iters), "--json"]
                res_raw = subprocess.run(cmd_raw, capture_output=True, text=True, check=True)
                out = res_raw.stdout
                json_part = out[out.find('['):out.rfind(']')+1]
                raw_data = json.loads(json_part)
                
                # Combine data for this (v, o)
                # Map operation name to timings
                raw_map = {item['Case']: item for item in raw_data}
                
                # Merge into full results
                # We have 17 operations total. 
                # (Some are in ops, some in raw. Actually core.py's run_ops_benchmark and run_raw_comparison have hardcoded sublists.)
                # I should probably unify these or handle them both.
                
                # Let's just collect everything from the subprocesses
                # Each item will have 'Operation' or 'Case' key.
                
                # We need clean data for the HTML report.
                for item in ops_data:
                    full_results.append({
                        "Operation": item['Operation'],
                        "Variables": v,
                        "Order": o,
                        "Python Time (s)": engine_format_to_float(item['Python Time']),
                        "SCosy Time (s)": engine_format_to_float(item['Sandalwood COSY Time']),
                        "Raw-Cosy Time (s)": np.nan,
                        "Speedup (S-Cosy)": item['Speedup']
                    })
                
                for item in raw_data:
                    full_results.append({
                        "Operation": item['Case'],
                        "Variables": v,
                        "Order": o,
                        "Python Time (s)": engine_format_to_float(item['Python Time']),
                        "SCosy Time (s)": engine_format_to_float(item['SCosy Time']),
                        "Raw-Cosy Time (s)": engine_format_to_float(item['Raw COSY Time']),
                        "Speedup (S-Cosy)": item['Speedup (Py/SCosy)'],
                        "Efficiency (vs Raw)": engine_format_to_float(item['Raw COSY Time']) / engine_format_to_float(item['SCosy Time']) if engine_format_to_float(item['SCosy Time']) > 0 else np.nan
                    })
                    
            except Exception as e:
                print(f"Error in sweep (v={v}, o={o}): {e}")
                if 'res_ops' in locals(): print(res_ops.stderr)
    
    # Generate report
    engine = BenchmarkEngine(10, 6)
    plots = engine.generate_plots(full_results)
    report_path = engine.generate_html_report(full_results, plots)
    
    print(f"\nFull benchmark complete!")
    print(f"HTML Report: {report_path}")

def engine_format_to_float(s):
    """Converts formatted timing string (e.g. '1.5 ms') back to float in seconds."""
    if not isinstance(s, str) or s == "N/A" or "nan" in s.lower(): return np.nan
    try:
        val, unit = s.split()
        val = float(val)
        if unit == "ms": return val * 1e-3
        if unit == "µs": return val * 1e-6
        if unit == "ns": return val * 1e-9
        return val
    except: return np.nan

def main():
    parser = argparse.ArgumentParser(description="Unified Sandalwood Benchmark Suite")
    parser.add_argument("--mode", choices=["ops", "raw", "batch", "profile", "full"], default="ops")
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--dims", type=int, default=4)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--npoints", type=int, default=10000)
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    
    args = parser.parse_args()
    
    if args.mode == "full":
        run_full_benchmark(args)
        return

    engine = BenchmarkEngine(args.order, args.dims)
    
    if args.mode == "ops":
        run_ops_benchmark(engine, args)
    elif args.mode == "raw":
        run_raw_comparison(engine, args)
    elif args.mode == "batch":
        run_batch_eval(engine, args)
    elif args.mode == "profile":
        run_profile(engine, args)

if __name__ == "__main__":
    main()
