import json
import subprocess
import sys

from tabulate import tabulate


def run_test(op, backend, dims=4, order=5, iters=1000, use_complex=False):
    cmd = [
        sys.executable, "scripts/benchmark_mtf.py",
        "--op", op,
        "--backend", backend,
        "--dims", str(dims),
        "--order", str(order),
        "--iters", str(iters)
    ]
    if use_complex:
        cmd.append("--complex")
    try:
        # Enforce 5s timeout per benchmark
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        # Output format: "RESULT_JSON: {...}"
        for line in result.stdout.splitlines():
            if line.startswith("RESULT_JSON:"):
                json_str = line.split("RESULT_JSON:")[1].strip()
                return json.loads(json_str)
        print(f"No result found in output: {result.stdout}")
        return None
    except subprocess.TimeoutExpired:
        print(f"Timeout (>5s): {cmd}")
        return {"error": "Timeout"}
    except subprocess.CalledProcessError:
        # If the backend failed (e.g. COSY missing), we treat it as failure but continue
        # Don't print stack trace for expected missing library errors unless debugging
        # print(f"Failed: {cmd}")
        # print(e.stderr)
        return {"error": "Failed"}

def format_time(stats):
    if not stats or "error" in stats:
        return stats.get("error", "N/A") if stats else "N/A"

    mean = stats["mean"]
    std = stats["std"]

    # Choose unit
    if mean < 1e-6:
        val = mean * 1e9
        err = std * 1e9
        unit = "ns"
    elif mean < 1e-3:
        val = mean * 1e6
        err = std * 1e6
        unit = "µs"
    elif mean < 1:
        val = mean * 1e3
        err = std * 1e3
        unit = "ms"
    else:
        val = mean
        err = std
        unit = "s"

    return f"{val:.2f} ± {err:.2f} {unit}"

def get_speedup(py_stats, cosy_stats):
    if not py_stats or "mean" not in py_stats: return "N/A"
    if not cosy_stats or "mean" not in cosy_stats: return "N/A"

    t_py = py_stats["mean"]
    t_cosy = cosy_stats["mean"]

    if t_cosy == 0: return "Inf"
    speedup = t_py / t_cosy
    return f"{speedup:.2f}x"

def main():
    real_ops = [
        "add", "sub", "mul", "div", "pow", "exp", "log", "sqrt",
        "sin", "cos", "tan", "cot", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "coth", "erf", "derivative", "integrate", "eval"
    ]
    complex_ops = [
        "add", "sub", "mul", "div", "exp", "sin"  # Foundation set for complex
    ]
    
    dims = 4  # Reduced from 6 for speed
    orders = [4, 8] # Reduced set for checking
    # Reduce iterations to avoid timeout (5s limit)
    iters = 5

    headers = ["Order", "Operation", "Python", "COSY", "Speedup"]
    
    # --- Real DA Benchmarks ---
    print(f"\n=== Real DA Benchmarks (Dimensions={dims}) ===")
    real_table_data = []

    for order in orders:
        current_iters = iters
        if order >= 10:
            current_iters = max(1, iters // 5)

        print(f"Running Order={order} (Iters={current_iters})...")

        for op in real_ops:
            sys.stderr.write(f"\r  Op={op} ...")
            sys.stderr.flush()
            
            py_res = run_test(op, "python", dims, order, current_iters, use_complex=False)
            cosy_res = run_test(op, "cosy", dims, order, current_iters, use_complex=False)

            row = [
                order,
                op,
                format_time(py_res),
                format_time(cosy_res),
                get_speedup(py_res, cosy_res)
            ]
            real_table_data.append(row)
        sys.stderr.write("\n")

    print(tabulate(real_table_data, headers=headers, tablefmt="github"))
    
    # --- Complex DA Benchmarks ---
    print(f"\n=== Complex DA Benchmarks (Dimensions={dims}) ===")
    complex_table_data = []

    for order in orders:
        current_iters = iters
        if order >= 10:
            current_iters = max(1, iters // 5)

        print(f"Running Order={order} (Iters={current_iters})...")

        for op in complex_ops:
            sys.stderr.write(f"\r  Op={op} ...")
            sys.stderr.flush()
            
            py_res = run_test(op, "python", dims, order, current_iters, use_complex=True)
            cosy_res = run_test(op, "cosy", dims, order, current_iters, use_complex=True)

            row = [
                order,
                op,
                format_time(py_res),
                format_time(cosy_res),
                get_speedup(py_res, cosy_res)
            ]
            complex_table_data.append(row)
        sys.stderr.write("\n")
    
    print(tabulate(complex_table_data, headers=headers, tablefmt="github"))
    sys.stderr.write("\nDone.\n")

if __name__ == "__main__":
    main()
