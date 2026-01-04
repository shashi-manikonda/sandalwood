import subprocess
import sys
import os

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
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Output format: "RESULT: backend: X.XXXXXX s/iter"
        for line in result.stdout.splitlines():
            if line.startswith("RESULT:"):
                time_s = float(line.split(":")[2].strip().split()[0])
                return time_s
        print(f"No result found in output: {result.stdout}")
        return float('inf')
    except subprocess.CalledProcessError as e:
        print(f"Failed: {cmd}")
        print(e.stdout)
        print(e.stderr)
        return float('inf')

def main():
    ops = ["add", "mul", "sin", "cos", "exp"]
    dims = 6 
    orders = [6, 12, 16]
    iters = 100
    
    print(f"Running benchmarks (Dims={dims}, Iters={iters})...")
    print("| Order | Operation | Python (s) | COSY (s) | Speedup |")
    print("|---|---|---|---|---|")
    
    for order in orders:
        for op in ops:
            t_py = run_test(op, "python", dims, order, iters)
            t_cosy = run_test(op, "cosy", dims, order, iters)
            
            speedup = t_py / t_cosy if t_cosy > 0 else 0
            print(f"| {order} | {op} | {t_py:.6f} | {t_cosy:.6f} | {speedup:.2f}x |")

if __name__ == "__main__":
    main()
