import subprocess
import sys
import os
import json
from tabulate import tabulate

def run_test(op, backend, dims=4, order=5, iters=1000):
    cmd = [
        sys.executable, "scripts/benchmark_mtf.py",
        "--op", op,
        "--backend", backend,
        "--dims", str(dims),
        "--order", str(order),
        "--iters", str(iters)
    ]
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
    except subprocess.CalledProcessError as e:
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
    ops = [
        "add", "sub", "mul", "div", "pow", "exp", "log", "sqrt",
        "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh", "derivative", "integrate", "eval"
    ]
    dims = 6
    orders = [2, 8, 6, 10]
    # Reduce iterations to avoid timeout (5s limit)
    iters = 25

    # print(f"Running benchmarks (Dims={dims}, Iters={iters})...")
    
    headers = ["Order", "Operation", "Python", "COSY", "Speedup"]
    table_data = []

    # We can perform a dry run or just print incrementally.
    # Since tabulate needs all data, we collect it.
    
    for order in orders:
        # Adjust iterations based on order to prevent timeout
        current_iters = iters
        if order >= 16:
            current_iters = max(1, iters // 10)
        elif order >= 12:
            current_iters = max(1, iters // 5)

        print(f"Running Order={order} (Iters={current_iters})...")

        for op in ops:
            # Print progress to stderr so it doesn't mess up if we were piping stdout,
            # though here we are just printing a final table.
            sys.stderr.write(f"\r  Op={op} ...")
            sys.stderr.flush()
            
            py_res = run_test(op, "python", dims, order, current_iters)
            cosy_res = run_test(op, "cosy", dims, order, current_iters)

            row = [
                order,
                op,
                format_time(py_res),
                format_time(cosy_res),
                get_speedup(py_res, cosy_res)
            ]
            table_data.append(row)
        sys.stderr.write("\n")

    sys.stderr.write("\nDone.\n")

    print(tabulate(table_data, headers=headers, tablefmt="github"))

if __name__ == "__main__":
    main()
