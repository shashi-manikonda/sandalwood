import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime
from sandalwood import mtf

# Local COSY configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
COSY_BIN = os.path.join(BASE_DIR, "cosy_bin")
COSY_FOX_SRC = os.path.join(BASE_DIR, "COSY.fox")

# Ensure artifacts directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MTF_TO_COSY = {
    "mtf.sin": "SIN", "mtf.cos": "COS", "mtf.exp": "EXP",
    "mtf.sqrt": "SQRT", "mtf.log": "LOG", "mtf.arctan": "ATAN",
    "mtf.tan": "TAN", "mtf.arcsin": "ASIN", "mtf.arccos": "ACOS",
    "mtf.sinh": "SINH", "mtf.cosh": "COSH", "mtf.tanh": "TANH",
    "mtf.gaussian": "EXP(-(DA(1)**2))", # Custom handling
}

class BenchmarkEngine:
    def __init__(self, order, dimension):
        self.order = order
        self.dimension = dimension
        self.variable_names = ["x", "y", "z", "u", "v", "w"][:dimension]

    def get_system_info(self):
        """Returns a dictionary containing system information."""
        import platform
        import psutil

        info = {
            "OS": platform.system(),
            "OS Release": platform.release(),
            "Architecture": platform.machine(),
            "Processor": platform.processor(),
            "Python Version": platform.python_version(),
            "CPU Count (Physical)": psutil.cpu_count(logical=False),
            "CPU Count (Logical)": psutil.cpu_count(logical=True),
            "Total RAM": f"{psutil.virtual_memory().total / (1024**3):.2f} GB"
        }
        return info
        
    def setup_mtf(self, implementation):
        """Initializes Sandalwood MTF with the specified backend."""
        mtf.initialize_mtf(max_order=self.order, max_dimension=self.dimension, implementation=implementation)
        globals_dict = {"mtf": mtf}
        for i, vn in enumerate(self.variable_names):
            globals_dict[vn] = mtf.var(i+1)
        return globals_dict

    def run_sandalwood(self, expression, implementation, iterations, repeats=5, warmup=1):
        """Runs a benchmark on a Sandalwood expression with statistics."""
        globals_dict = self.setup_mtf(implementation)
        
        # Compile expression first to avoid parsing overhead during timing
        code = compile(expression, "<string>", "eval")

        # Warmup
        for _ in range(warmup):
            for _ in range(iterations):
                _ = eval(code, globals_dict)

        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(iterations):
                res = eval(code, globals_dict)
            elapsed = time.perf_counter() - start
            timings.append(elapsed)

        avg_time = np.mean(timings)
        std_time = np.std(timings)

        # Extract coefficients from last result
        exponents = res.exponents
        coeffs = res.coeffs
        return {tuple(exp): c for exp, c in zip(exponents, coeffs)}, avg_time, std_time

    def run_raw_cosy(self, name, cosy_expr, iterations, repeats=5, warmup=1):
        """Runs a benchmark using Raw COSY script execution with statistics."""
        script_name = f"tmp_{name}.fox"
        fox_path = os.path.join(ARTIFACTS_DIR, script_name)
        dat_path = os.path.join(ARTIFACTS_DIR, "foxyinp.dat")
        
        # Prepare system files in artifacts dir if they don't exist
        # COSY needs COSY.fox to be present to run properly
        src_fox = os.path.join(BASE_DIR, "COSY.fox")
        dst_fox = os.path.join(ARTIFACTS_DIR, "COSY.fox")
        if os.path.exists(src_fox) and not os.path.exists(dst_fox):
             import shutil
             shutil.copy(src_fox, dst_fox)

        cosy_script = f"""
INCLUDE 'COSY';
PROCEDURE RUN;
VARIABLE ORDER 1; VARIABLE DIM 1; VARIABLE NM1 1;
PROCEDURE TESTEXP NM1;
    VARIABLE TEMP NM1; VARIABLE I 1;
    WRITE 6 '{name}';
    LOOP I 1 {iterations}; TEMP:={cosy_expr}; ENDLOOP;
    WRITE 6 TEMP;
ENDPROCEDURE;
ORDER := {self.order}; DIM := {self.dimension};
DAINI ORDER DIM 0 NM1;
TESTEXP NM1;
ENDPROCEDURE;
RUN;
END;
"""
        with open(fox_path, "w") as f: f.write(cosy_script)
        with open(dat_path, "w") as f_dat: f_dat.write(os.path.splitext(script_name)[0])

        # Warmup (not fully applicable since it's a process, but good for disk cache)
        for _ in range(warmup):
             with open(dat_path, 'r') as dat_file:
                subprocess.run([COSY_BIN], stdin=dat_file, capture_output=True, text=True, cwd=ARTIFACTS_DIR)

        timings = []
        last_process = None

        try:
            for _ in range(repeats):
                start = time.perf_counter()
                with open(dat_path, 'r') as dat_file:
                    last_process = subprocess.run([COSY_BIN], stdin=dat_file, capture_output=True, text=True, check=True, cwd=ARTIFACTS_DIR)
                elapsed = time.perf_counter() - start
                timings.append(elapsed)

            avg_time = np.mean(timings)
            std_time = np.std(timings)
            return self.parse_cosy_output(last_process.stdout), avg_time, std_time
        except Exception as e:
            print(f"Raw COSY failed: {e}")
            return None, None, None

    @staticmethod
    def format_time(seconds, std_dev=None):
        """Formats time in seconds to a string with appropriate units (s, ms, µs)."""
        if pd.isna(seconds): return "N/A"

        unit = "s"
        factor = 1.0

        if seconds < 1e-3:
            unit = "µs"
            factor = 1e6
        elif seconds < 1:
            unit = "ms"
            factor = 1e3

        val = seconds * factor
        if std_dev is not None and not pd.isna(std_dev):
            std = std_dev * factor
            return f"{val:.2f} ± {std:.2f} {unit}"
        return f"{val:.2f} {unit}"

    @staticmethod
    def format_speedup(ratio):
        """Formats a speedup ratio to a meaningful string."""
        if pd.isna(ratio) or ratio == 0: return "N/A"
        if ratio == float('inf'): return "Inf"
        if ratio >= 10:
             return f"{int(ratio)}x"
        return f"{ratio:.2f}x"

    def parse_cosy_output(self, output):
        """Parses COSY output to extract coefficients dictionary."""
        coefficients = {}
        lines = output.strip().split('\n')
        start = False
        for line in lines:
            line = line.strip()
            if line.startswith("I  COEFFICIENT"): start = True; continue
            if line.startswith("----") and start: break
            if start and line:
                parts = line.split()
                try:
                    coeff = float(parts[1])
                    exp = tuple(int(e) for e in parts[3:])
                    coefficients[exp] = coeff
                except: continue
        return coefficients

    @staticmethod
    def calculate_rmse(coeffs_a, coeffs_b):
        """Calculates RMSE between two coefficient dictionaries."""
        if not coeffs_a or not coeffs_b: return np.nan
        all_keys = set(coeffs_a.keys()) | set(coeffs_b.keys())
        if not all_keys: return 0.0
        sq_errors = [(coeffs_a.get(k, 0.0) - coeffs_b.get(k, 0.0))**2 for k in all_keys]
        return np.sqrt(np.mean(sq_errors))

    @staticmethod
    def save_markdown_results(df, title):
        """Saves results dataframe as a timestamped Markdown table."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{title.lower().replace(' ', '_')}_{timestamp}.md"
        filepath = os.path.join(ARTIFACTS_DIR, filename)
        
        with open(filepath, "w") as f:
            f.write(f"# {title} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n")
            
        print(f"Results saved to {filepath}")
        return filepath

    def generate_plots(self, full_results):
        """Generates plots for timing vs order and returns as base64 strings."""
        plots = {}
        df = pd.DataFrame(full_results)
        
        # Plot 1: Timing vs Order for different variants (Python vs S-Cosy)
        # We'll pick a few representative ops or just average all
        for op in df['Operation'].unique():
            plt.figure(figsize=(10, 6))
            op_df = df[df['Operation'] == op]
            
            for vars_count in op_df['Variables'].unique():
                v_df = op_df[op_df['Variables'] == vars_count]
                plt.plot(v_df['Order'], v_df['Python Time (s)'], marker='o', label=f'Python (v={vars_count})')
                plt.plot(v_df['Order'], v_df['SCosy Time (s)'], marker='s', label=f'S-Cosy (v={vars_count})')
            
            plt.title(f'Timing vs Order: {op}')
            plt.xlabel('Order')
            plt.ylabel('Time (s)')
            plt.yscale('log')
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.legend()
            
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            plots[op] = base64.b64encode(buf.getvalue()).decode('utf-8')
            
        return plots

    def generate_html_report(self, full_results, plots):
        """Generates a styled HTML report with tables and embedded plots."""
        df = pd.DataFrame(full_results)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sys_info = self.get_system_info()

        sys_html = "".join([f"<li><strong>{k}:</strong> {v}</li>" for k, v in sys_info.items()])
        
        # Export CSV
        csv_path = os.path.join(ARTIFACTS_DIR, f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Raw CSV exported to: {csv_path}")

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sandalwood Comprehensive Benchmark Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f8f9fa; color: #333; }}
                h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: #fff; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                th, td {{ padding: 12px; text-align: left; border: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #e9ecef; }}
                .plot-container {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-top: 30px; }}
                .plot-item {{ background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }}
                .plot-item img {{ max-width: 100%; height: auto; }}
                .summary {{ background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            </style>
        </head>
        <body>
            <h1>Sandalwood Benchmark Report</h1>
            <div class="summary">
                <p><strong>Generated on:</strong> {timestamp}</p>
                <p><strong>Summary:</strong> This report compares the performance of Sandalwood's Python backend, Sandalwood's COSY backend (S-Cosy), and direct COSY script execution (Raw-Cosy) across various expansion orders and variables.</p>
            </div>

            <h2>Performance Comparison Table</h2>
            {df.to_html(index=False, classes='table')}

            <h2>Performance Visualizations (Log Scale)</h2>
            <div class="plot-container">
        """
        
        for op, img_data in plots.items():
            html += f"""
                <div class="plot-item">
                    <h3>{op}</h3>
                    <img src="data:image/png;base64,{img_data}" alt="{op} plot">
                </div>
            """
            
        html += """
            </div>
        </body>
        </html>
        """
        
        filepath = os.path.join(ARTIFACTS_DIR, "benchmark_report.html")
        with open(filepath, "w") as f:
            f.write(html)
        
        print(f"HTML Report generated at {filepath}")
        return filepath
