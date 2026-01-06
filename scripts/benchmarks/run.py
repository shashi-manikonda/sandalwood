import argparse
import sys
import os
import time
import pandas as pd
from core import BenchmarkEngine, ARTIFACTS_DIR

# Ensure we can import sandalwood from the parent src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from sandalwood import TaylorMap, mtf

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

def main():
    parser = argparse.ArgumentParser(description="Unified Sandalwood Benchmark Suite")
    parser.add_argument("--mode", choices=["ops", "raw", "batch", "profile"], default="ops")
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--dims", type=int, default=4)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--npoints", type=int, default=10000)
    
    args = parser.parse_args()
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
