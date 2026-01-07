import argparse
import sys
import os
import time
import tracemalloc
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

def format_memory(bytes_val):
    """Formats bytes to human-readable string."""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024**2:
        return f"{bytes_val/1024:.2f} KB"
    else:
        return f"{bytes_val/1024**2:.2f} MB"

def measure_memory(func, *args, **kwargs):
    """Runs a function and captures peak memory usage."""
    tracemalloc.start()
    try:
        result = func(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


COMPLEX_CASES = [
    ("mul_intensive", "(x + y + z + u)**2", "(DA(1)+DA(2)+DA(3)+DA(4))*(DA(1)+DA(2)+DA(3)+DA(4))"),
    ("sin_complex", "mtf.sin(0.5 + x + y)", "SIN(0.5 + DA(1) + DA(2))"),
    ("exp_test", "mtf.exp(x - 0.5)", "EXP(DA(1) - 0.5)"),
]

ALL_BENCHMARKS = FULL_OPS + COMPLEX_CASES

def run_ops_benchmark(engine, args):
    """Benchmarks individual operations (Python vs COSY Backend)."""
    ops = ALL_BENCHMARKS
    
    if args.filter:
        ops = [op for op in ops if args.filter.lower() in op[0].lower()]

    results = []
    for name, mtf_expr, cosy_expr in ops:
        print(f"Benchmarking {name}...")
        
        if args.memory:
            (c_py, t_py, s_py), mem_py = measure_memory(engine.run_sandalwood, mtf_expr, "python", args.iters)
            (c_sc, t_sc, s_sc), mem_sc = measure_memory(engine.run_sandalwood, mtf_expr, "cosy", args.iters)
        else:
            c_py, t_py, s_py = engine.run_sandalwood(mtf_expr, "python", args.iters)
            c_sc, t_sc, s_sc = engine.run_sandalwood(mtf_expr, "cosy", args.iters)
            mem_py, mem_sc = 0, 0

        row = {
            "Operation": name,
            "Python Time": engine.format_time(t_py, s_py),
            "S-COSY Time": engine.format_time(t_sc, s_sc),
            "Speedup": engine.format_speedup(t_py / t_sc if t_sc > 0 else 0)
        }

        if args.memory:
            row["Py Mem"] = format_memory(mem_py)
            row["S-COSY Mem"] = format_memory(mem_sc)

        results.append(row)
        
    df = pd.DataFrame(results)
    if args.json:
        print(df.to_json(orient='records'))
    else:
        try:
             print("\n" + df.to_markdown(index=False, tablefmt="grid"))
        except ImportError:
             print("\n" + df.to_string(index=False))
    engine.save_markdown_results(df, "Operation Benchmarks")

def run_raw_comparison(engine, args):
    """Compares Sandalwood (Python/COSY) with Raw COSY execution."""
    cases = ALL_BENCHMARKS
    
    if args.filter:
        cases = [c for c in cases if args.filter.lower() in c[0].lower()]

    results = []
    for name, mtf_expr, cosy_expr in cases:
        print(f"Comparing {name} with Raw COSY...")

        if args.memory:
             (c_py, t_py, s_py), mem_py = measure_memory(engine.run_sandalwood, mtf_expr, "python", args.iters)
             (c_sc, t_sc, s_sc), mem_sc = measure_memory(engine.run_sandalwood, mtf_expr, "cosy", args.iters)
        else:
             c_py, t_py, s_py = engine.run_sandalwood(mtf_expr, "python", args.iters)
             c_sc, t_sc, s_sc = engine.run_sandalwood(mtf_expr, "cosy", args.iters)
             mem_py, mem_sc = 0, 0

        c_raw, t_raw, s_raw = engine.run_raw_cosy(name, cosy_expr, args.iters)
        
        rmse_py = engine.calculate_rmse(c_py, c_raw)
        rmse_sc = engine.calculate_rmse(c_sc, c_raw)
        
        row = {
            "Operation": name,
            "Python Time": engine.format_time(t_py, s_py),
            "S-COSY Time": engine.format_time(t_sc, s_sc),
            "Raw COSY Time": engine.format_time(t_raw, s_raw),
            "RMSE (Py vs Raw)": f"{rmse_py:.2e}",
            "RMSE (SCosy vs Raw)": f"{rmse_sc:.2e}",
            "Speedup (Py/SCosy)": engine.format_speedup(t_py / t_sc if t_sc > 0 else 0)
        }

        if args.memory:
            row["Py Mem"] = format_memory(mem_py)
            row["S-COSY Mem"] = format_memory(mem_sc)

        results.append(row)
        
    df = pd.DataFrame(results)
    if args.json:
        print(df.to_json(orient='records'))
    else:
        try:
             print("\n" + df.to_markdown(index=False, tablefmt="grid"))
        except ImportError:
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
    try:
         print("\n" + df.to_markdown(index=False, tablefmt="grid"))
    except ImportError:
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
            
            # Use data from raw comparison which now covers EVERYTHING
            cmd_raw = [python_bin, script_path, "--mode", "raw", "--order", str(o), "--dims", str(v), "--iters", str(iters), "--json"]
            try:
                res_raw = subprocess.run(cmd_raw, capture_output=True, text=True, check=True)
                out = res_raw.stdout
                json_part = out[out.find('['):out.rfind(']')+1]
                raw_data = json.loads(json_part)
                
                for item in raw_data:
                    full_results.append({
                        "Operation": item['Operation'],
                        "Variables": v,
                        "Order": o,
                        "Python Time (s)": engine_format_to_float(item['Python Time']),
                        "SCosy Time (s)": engine_format_to_float(item['S-COSY Time']),
                        "Raw-Cosy Time (s)": engine_format_to_float(item['Raw COSY Time']),
                        "Speedup (S-Cosy)": item['Speedup (Py/SCosy)'],
                        "Efficiency (vs Raw)": engine_format_to_float(item['Raw COSY Time']) / engine_format_to_float(item['S-COSY Time']) if engine_format_to_float(item['S-COSY Time']) > 0 else np.nan
                    })
                    
            except Exception as e:
                print(f"Error in sweep (v={v}, o={o}): {e}")
                
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
    parser.add_argument("--memory", action="store_true", help="Enable memory profiling")
    parser.add_argument("--filter", type=str, help="Filter benchmarks by name pattern")
    
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
