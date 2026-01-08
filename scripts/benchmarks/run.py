import argparse
import os
import signal
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd


class TimeLimit:
    """Context manager for limiting execution time."""
    def __init__(self, seconds):
        self.seconds = seconds
    
    def __enter__(self):
        if self.seconds:
            signal.signal(signal.SIGALRM, self._handle_timeout)
            signal.alarm(int(self.seconds))
        return self
    
    def __exit__(self, type, value, traceback):
        if self.seconds:
            signal.alarm(0)
            
    def _handle_timeout(self, signum, frame):
        raise TimeoutError(f"Execution exceeded {self.seconds}s")

from core import ARTIFACTS_DIR, BenchmarkEngine

# Ensure we can import sandalwood from the parent src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from sandalwood import TaylorMap, mtf


def generate_benchmark_cases(dims):
    """Generates benchmark cases string that use all available variables."""
    var_names = ["x", "y", "z", "u", "v", "w", "a", "b"][:dims]
    
    py_sum = " + ".join(var_names)
    cosy_sum = " + ".join([f"DA({i+1})" for i in range(dims)])

    cases = [
        # Arithmetic
        # Note: Using ALL variables in expressions as requested
        ("Add", f"({py_sum}) + ({py_sum})", f"({cosy_sum})+({cosy_sum})"),
        ("Sub", f"({py_sum}) - ({py_sum})", f"({cosy_sum})-({cosy_sum})"),
        ("Mul", f"({py_sum}) * ({py_sum})", f"({cosy_sum})*({cosy_sum})"),
        ("Div", f"({py_sum}) / ({py_sum} + 0.01)", f"({cosy_sum}) / ({cosy_sum} + 0.01)"),
        # Pow: (1+sum)*(1+sum)*(1+sum) - Explicit multiplication to avoid COSY ^ operator issues on DA
        ("Cube (Mul)", f"({py_sum})**3", f"({cosy_sum})*(({cosy_sum})*({cosy_sum}))"),
        
        # Elementary Functions
        ("Sin", f"mtf.sin({py_sum})", f"SIN({cosy_sum})"),
        ("Cos", f"mtf.cos({py_sum})", f"COS({cosy_sum})"),
        ("Tan", f"mtf.tan({py_sum})", f"TAN({cosy_sum})"),
        ("Exp", f"mtf.exp({py_sum})", f"EXP({cosy_sum})"),
        ("Log", f"mtf.log(1+{py_sum})", f"LOG(1+{cosy_sum})"),
        ("Sqrt", f"mtf.sqrt(1+{py_sum})", f"SQRT(1+{cosy_sum})"),
        ("Asin", f"mtf.arcsin(0.5*({py_sum}))", f"ASIN(0.5*({cosy_sum}))"),
        ("Acos", f"mtf.arccos(0.5*({py_sum}))", f"ACOS(0.5*({cosy_sum}))"),
        ("Atan", f"mtf.arctan({py_sum})", f"ATAN({cosy_sum})"),
        ("Sinh", f"mtf.sinh({py_sum})", f"SINH({cosy_sum})"),
        ("Cosh", f"mtf.cosh({py_sum})", f"COSH({cosy_sum})"),
        ("Tanh", f"mtf.tanh({py_sum})", f"TANH({cosy_sum})"),
        
        # Complex Cases
        ("mul_intensive", f"({py_sum})**2", f"({cosy_sum})*({cosy_sum})"),
        ("sin_complex", f"mtf.sin(0.5 + {py_sum})", f"SIN(0.5 + {cosy_sum})"),
        ("exp_test", f"mtf.exp({py_sum} - 0.5)", f"EXP({cosy_sum} - 0.5)"),
    ]
    return cases

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

def run_ops_benchmark(engine, args):
    """Benchmarks individual operations (Python vs COSY Backend)."""
    ops = generate_benchmark_cases(args.dims)
    
    if args.filter:
        ops = [op for op in ops if args.filter.lower() in op[0].lower()]

    results = []
    for name, mtf_expr, cosy_expr in ops:
        print(f"Benchmarking {name}...")
        
        try:
            with TimeLimit(args.timeout):
                # Always run timing first (without memory overhead)
                c_py, t_py, s_py = engine.run_sandalwood(mtf_expr, "python", args.iters)
                c_sc, t_sc, s_sc = engine.run_sandalwood(mtf_expr, "cosy", args.iters)
                
                mem_py, mem_sc = 0, 0
                if args.memory:
                     # Run separately for memory profile if requested
                     _, mem_py = measure_memory(engine.run_sandalwood, mtf_expr, "python", args.iters)
                     _, mem_sc = measure_memory(engine.run_sandalwood, mtf_expr, "cosy", args.iters)
        except TimeoutError:
            print(f"Skipping {name} - Timed out (>{args.timeout}s)", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Skipping {name} - Failed: {e}", file=sys.stderr)
            continue

        row = {
            "Operation": name,
            "Python Time": engine.format_time(t_py, s_py),
            "S-COSY Time": engine.format_time(t_sc, s_sc),
            "Speedup (vs Python)": engine.format_speedup(t_py / t_sc if t_sc > 0 else 0)
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
    cases = generate_benchmark_cases(args.dims)
    
    if args.filter:
        cases = [c for c in cases if args.filter.lower() in c[0].lower()]

    results = []
    display_results = []
    
    for name, mtf_expr, cosy_expr in cases:
        print(f"Comparing {name} with Raw COSY...", file=sys.stderr)

        c_py, t_py, s_py = {}, np.nan, np.nan
        c_sc, t_sc, s_sc = {}, np.nan, np.nan
        mem_py, mem_sc = 0, 0
        
        if args.mode == "cosy_raw":
             # Skip Python benchmark in dedicated COSY mode
             c_py, t_py, s_py = {}, np.nan, np.nan
        else:
            try:
                with TimeLimit(args.timeout):
                    # Timing run first
                    c_py, t_py, s_py = engine.run_sandalwood(mtf_expr, "python", args.iters)
                    if args.memory:
                        # Memory run second
                         _, mem_py = measure_memory(engine.run_sandalwood, mtf_expr, "python", args.iters)
            except TimeoutError:
                print(f"  [Python] Timed out (>{args.timeout}s)", file=sys.stderr)
            except Exception as e:
                print(f"  [Python] Failed: {e}", file=sys.stderr)

        # 2. Run S-COSY Benchmark
        try:
            with TimeLimit(args.timeout):
                # Timing run first
                c_sc, t_sc, s_sc = engine.run_sandalwood(mtf_expr, "cosy", args.iters)
                if args.memory:
                    # Memory run second
                     _, mem_sc = measure_memory(engine.run_sandalwood, mtf_expr, "cosy", args.iters)
        except TimeoutError:
            print(f"  [S-COSY] Timed out (>{args.timeout}s)", file=sys.stderr)
        except Exception as e:
            print(f"  [S-COSY] Failed: {e}", file=sys.stderr)

        # 3. Run Raw COSY Benchmark (Handles its own timeout)
        c_raw, t_raw, s_raw = engine.run_raw_cosy(name, cosy_expr, args.iters, timeout=args.timeout)
        
        # If S-COSY AND Raw COSY failed, there is no point in reporting (we need at least one COSY metric)
        if pd.isna(t_sc) and pd.isna(t_raw):
            print(f"Skipping {name} - Both S-COSY and Raw COSY failed/timed out", file=sys.stderr)
            continue

        # Calculate RMSE/Speedup relative to whatever is available
        rmse_py = engine.calculate_rmse(c_py, c_raw) if not pd.isna(t_py) and not pd.isna(t_raw) else np.nan
        rmse_sc = engine.calculate_rmse(c_sc, c_raw) if not pd.isna(t_sc) and not pd.isna(t_raw) else np.nan
        
        speedup_py = t_py / t_sc if (not pd.isna(t_py) and not pd.isna(t_sc) and t_sc > 0) else np.nan
        
        def safe_fmt(val): return f"{val:.2e}" if not pd.isna(val) else "N/A"

        # Raw data for JSON/Analysis
        row = {
            "Operation": name,
            "Python Time": t_py if not pd.isna(t_py) else None,
            "S-COSY Time": t_sc if not pd.isna(t_sc) else None,
            "Raw COSY Time": t_raw if not pd.isna(t_raw) else None,
            "RMSE (Py vs Raw)": rmse_py if not pd.isna(rmse_py) else None,
            "RMSE (SCosy vs Raw)": rmse_sc if not pd.isna(rmse_sc) else None,
            "Speedup (vs Python)": speedup_py if not pd.isna(speedup_py) else None,
            "Python Expr": mtf_expr,
            "COSY Expr": cosy_expr
        }
        
        # Formatted data for Console Display
        disp_row = {
            "Operation": name,
            "Python Time": engine.format_time(t_py, s_py) if not pd.isna(t_py) else "N/A",
            "S-COSY Time": engine.format_time(t_sc, s_sc) if not pd.isna(t_sc) else "N/A",
            "Raw COSY Time": engine.format_time(t_raw, s_raw) if not pd.isna(t_raw) else "N/A",
            "RMSE (Py)": safe_fmt(rmse_py),
            "RMSE (SC)": safe_fmt(rmse_sc),
            "Speedup": engine.format_speedup(speedup_py) if not pd.isna(speedup_py) else "N/A",
        }

        if args.memory:
            row["Py Mem"] = mem_py
            row["S-COSY Mem"] = mem_sc
            disp_row["Py Mem"] = format_memory(mem_py) if mem_py > 0 else "N/A"
            disp_row["S-COSY Mem"] = format_memory(mem_sc) if mem_sc > 0 else "N/A"

        results.append(row)
        display_results.append(disp_row)
        
    df = pd.DataFrame(results)
    df_disp = pd.DataFrame(display_results)
    
    if args.json:
        # JSON output must be pure raw data
        print(df.to_json(orient='records'))
    else:
        try:
             print("\n" + df_disp.to_markdown(index=False, tablefmt="grid"))
        except ImportError:
             print("\n" + df_disp.to_string(index=False))
             
    engine.save_markdown_results(df_disp, "Raw COSY Comparison")

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

def run_batch_math(engine, args):
    """Benchmarks vectorized arithmetic (Batch Operations)."""
    print(f"Benchmarking batch arithmetic (Vector[MTF] + Vector[MTF]) with N={args.iters}...")
    globals_dict = engine.setup_mtf("cosy")
    import numpy as np
    
    x = globals_dict['x']
    y = globals_dict['y']
    
    # Create N distinct MTFs
    print("Creating test vectors...")
    # Using small constants to prevent optimization/caching if any
    vec_a = np.array([x + i * 1e-5 for i in range(args.iters)], dtype=object)
    vec_b = np.array([y + i * 1e-5 for i in range(args.iters)], dtype=object)
    
    # 1. Vectorized (Batch) Add
    print("Running Vectorized Add...")
    start = time.process_time()
    res_vec_add = vec_a + vec_b
    t_vec_add = (time.process_time() - start)
    
    # 2. Loop Add
    print("Running Loop Add...")
    start = time.process_time()
    res_loop_add = [a + b for a, b in zip(vec_a, vec_b)]
    t_loop_add = (time.process_time() - start)
    
    # 3. Vectorized (Batch) Mul
    print("Running Vectorized Mul...")
    start = time.process_time()
    res_vec_mul = vec_a * vec_b
    t_vec_mul = (time.process_time() - start)
    
    # 4. Loop Mul
    print("Running Loop Mul...")
    start = time.process_time()
    res_loop_mul = [a * b for a, b in zip(vec_a, vec_b)]
    t_loop_mul = (time.process_time() - start)
    
    results = [
        {
            "Type": "Add",
            "Operations": args.iters,
            "Vectorized Time": engine.format_time(t_vec_add),
            "Loop Time": engine.format_time(t_loop_add),
            "Speedup": f"{t_loop_add / t_vec_add:.1f}x" if t_vec_add > 0 else "N/A"
        },
        {
            "Type": "Mul",
            "Operations": args.iters,
            "Vectorized Time": engine.format_time(t_vec_mul),
            "Loop Time": engine.format_time(t_loop_mul),
            "Speedup": f"{t_loop_mul / t_vec_mul:.1f}x" if t_vec_mul > 0 else "N/A"
        }
    ]
    
    df = pd.DataFrame(results)
    try:
         print("\n" + df.to_markdown(index=False, tablefmt="grid"))
    except ImportError:
         print("\n" + df.to_string(index=False))
    engine.save_markdown_results(df, "Batch Math Benchmarks")

def run_profile(engine, args):
    """Runs a cProfile on core operations."""
    import cProfile
    import io
    import pstats
    print("Running cProfile on complex arithmetic and mapping...")
    
    globals_dict = engine.setup_mtf("cosy")
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
    iters = args.iters
    
    full_results = []
    
    python_bin = sys.executable
    script_path = __file__
    
    for v in vars_list:
        for o in orders:
            print(f"\n>>> Sweep: Variables={v}, Order={o} <<<")
            
            # Use data from raw comparison which now covers EVERYTHING
            mode_flag = "raw"
            if args.mode == "full_cosy": mode_flag = "cosy_raw"

            cmd_raw = [python_bin, script_path, "--mode", mode_flag, "--order", str(o), "--dims", str(v), "--iters", str(iters), "--timeout", str(args.timeout), "--json"]
            try:
                res_raw = subprocess.run(cmd_raw, capture_output=True, text=True, check=True)
                out = res_raw.stdout
                
                start_idx = out.find('[')
                end_idx = out.rfind(']')
                
                if start_idx == -1 or end_idx == -1:
                    raise ValueError(f"No JSON found in output. Output was:\n{out}")
                    
                json_part = out[start_idx:end_idx+1]
                raw_data = json.loads(json_part)
                
                for item in raw_data:
                    t_py = item.get('Python Time')
                    t_sc = item.get('S-COSY Time')
                    t_raw = item.get('Raw COSY Time')
                    
                    # Handle NaNs from JSON (which might come as None)
                    if t_py is None: t_py = np.nan
                    if t_sc is None: t_sc = np.nan
                    if t_raw is None: t_raw = np.nan

                    speedup_py = item.get('Speedup (vs Python)')
                    if speedup_py is None: speedup_py = np.nan
                    
                    # Recompute raw speedup if needed, or trust child
                    speedup_raw = t_raw / t_sc if (t_sc > 0 and not np.isnan(t_sc) and not np.isnan(t_raw)) else np.nan

                    full_results.append({
                        "Operation": item['Operation'],
                        "Variables": v,
                        "Order": o,
                        "Python Time (s)": t_py,
                        "SCosy Time (s)": t_sc,
                        "Raw-Cosy Time (s)": t_raw,
                        "Speedup (vs Python)": speedup_py,
                        "Speedup (vs Raw COSY)": speedup_raw,
                        "Python Expr": item.get('Python Expr', ''),
                        "COSY Expr": item.get('COSY Expr', '')
                    })
            
            except Exception as e:
                print(f"Error in sweep (v={v}, o={o}): {e}")

    # Generate report title based on mode
    report_title = "Benchmark Report"
    if args.mode == "full_cosy": report_title = "COSY Backend Performance Report"

    engine = BenchmarkEngine(10, 6)
    plots = engine.generate_plots(full_results)
    report_path = engine.generate_html_report(full_results, plots, method_info={"iterations": iters, "title": report_title})
    
    print("\nFull benchmark complete!")
    print(f"HTML Report: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Unified Sandalwood Benchmark Suite")
    parser.add_argument("--mode", choices=["ops", "raw", "batch", "batch_math", "profile", "full", "full_cosy", "cosy_raw"], default="ops")
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--dims", type=int, default=4)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--npoints", type=int, default=10000)
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--memory", action="store_true", help="Enable memory profiling")
    parser.add_argument("--filter", type=str, help="Filter benchmarks by name pattern")
    parser.add_argument("--timeout", type=int, default=30, help="Benchmark timeout in seconds")
    
    args = parser.parse_args()
    
    if args.mode in ["full", "full_cosy"]:
        run_full_benchmark(args)
        return

    engine = BenchmarkEngine(args.order, args.dims)
    
    if args.mode == "ops":
        run_ops_benchmark(engine, args)
    elif args.mode in ["raw", "cosy_raw"]:
        run_raw_comparison(engine, args)
    elif args.mode == "batch":
        run_batch_eval(engine, args)
    elif args.mode == "batch_math":
        run_batch_math(engine, args)
    elif args.mode == "profile":
        run_profile(engine, args)

if __name__ == "__main__":
    main()
