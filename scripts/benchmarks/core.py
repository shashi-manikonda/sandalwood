"""
core.py
=======
Purpose:
    Core benchmarking engine and reporting utilities for the Sandalwood library.
    Provides the infrastructure for executing mathematical expressions on different backends.

Logic:
    - BenchmarkEngine (Class):
        - setup_mtf: Initializes Sandalwood with specific backends and variables.
        - run_sandalwood: Compiles expressions into lambdas and measures execution time.
        - run_raw_cosy: Dynamically generates, compiles, and runs standalone Fortran 
          binaries to measure base COSY performance.
        - run_ops_benchmark: Orchestrates a comparison between Python and COSY backends.
        - generate_report: Produces rich HTML dashboards with system info and comparison plots.
    - Path Utilities: Manages locations for COSY binaries, source files, and benchmark artifacts.

Input/Arguments:
    - Relies on internal configuration and parameters passed to BenchmarkEngine methods.

Output:
    - Standardized timing data (avg, std dev) for various operations.
    - Visualizations (Matplotlib) and formatted reports (Markdown/HTML).
"""
import base64
import os
import subprocess
import time
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sandalwood import mtf

# Local COSY configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
COSY_BIN = os.path.join(BASE_DIR, "cosy_bin")
COSY_FOX_SRC = os.path.join(BASE_DIR, "COSY.fox")

# Ensure artifacts directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MTF_TO_COSY = {
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
    "mtf.gaussian": "EXP(-(DA(1)**2))",  # Custom handling
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
            "Total RAM": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        }
        return info

    def setup_mtf(self, implementation):
        """Initializes Sandalwood MTF with the specified backend."""
        mtf.initialize_mtf(
            max_order=self.order,
            max_dimension=self.dimension,
            implementation=implementation,
        )
        globals_dict = {"mtf": mtf}
        for i, vn in enumerate(self.variable_names):
            globals_dict[vn] = mtf.var(i + 1)
        return globals_dict

    def run_sandalwood(
        self, expression, implementation, iterations, repeats=5, warmup=1
    ):
        """Runs a benchmark on a Sandalwood expression with statistics."""
        globals_dict = self.setup_mtf(implementation)

        # Wrap expression in a lambda and compile to avoid eval() overhead in loop
        # This converts "x + y" into a callable function object
        code = compile(f"lambda: {expression}", "<string>", "eval")
        func = eval(code, globals_dict)

        # Warmup
        for _ in range(warmup):
            for _ in range(iterations):
                _ = func()

        timings = []
        for _ in range(repeats):
            start = time.process_time()
            for _ in range(iterations):
                res = func()
            elapsed = time.process_time() - start
            timings.append(elapsed / iterations)

        avg_time = np.mean(timings)
        std_time = np.std(timings)

        # Extract coefficients from last result
        exponents = res.exponents
        coeffs = res.coeffs
        return {tuple(exp): c for exp, c in zip(exponents, coeffs)}, avg_time, std_time

    def run_raw_cosy(
        self, name, cosy_expr, iterations, repeats=5, warmup=1, timeout=None
    ):
        """Runs a benchmark using Raw COSY script execution with statistics."""
        script_name = f"tmp_{name}.fox"
        fox_path = os.path.join(ARTIFACTS_DIR, script_name)
        dat_path = os.path.join(ARTIFACTS_DIR, "foxyinp.dat")

        # Prepare system files in artifacts dir if they don't exist
        # COSY needs COSY.fox to be present to run properly
        src_fox = os.path.join(BASE_DIR, "COSY.fox")
        dst_fox = os.path.join(ARTIFACTS_DIR, "COSY.fox")
        
        # If local COSY.fox is missing, try to find it from external source
        if not os.path.exists(src_fox):
            cosy_src_env = os.environ.get("SANDALWOOD_COSY_SRC")
            if cosy_src_env:
                candidate_paths = [
                    os.path.join(cosy_src_env, "cosy.fox"),
                    os.path.join(cosy_src_env, "../apps/cosy.fox"),
                    os.path.join(cosy_src_env, "apps/cosy.fox"),
                ]
                for path in candidate_paths:
                    if os.path.exists(path):
                        src_fox = path
                        break
        
        if os.path.exists(src_fox) and not os.path.exists(dst_fox):
            import shutil
            shutil.copy(src_fox, dst_fox)
        elif not os.path.exists(src_fox) and not os.path.exists(dst_fox):
            raise FileNotFoundError(
                "COSY.fox not found. Since COSY Infinity is proprietary, its files are not distributed "
                "with Sandalwood. Please place a copy of your licensed cosy.fox in scripts/benchmarks/COSY.fox "
                "or set the SANDALWOOD_COSY_SRC environment variable."
            )

        # Also copy COSY.bin and DAINI.DAT if available
        for fname in ["COSY.bin", "DAINI.DAT"]:
            src = os.path.join(BASE_DIR, fname)
            dst = os.path.join(ARTIFACTS_DIR, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                import shutil
                shutil.copy(src, dst)

        timings = []
        last_process = None

        # Use internal COSY CPUSEC timing to measure computation directly
        # This avoids process startup overhead completely

        try:
            for _ in range(repeats):
                cosy_script = f"""
INCLUDE 'COSY';
PROCEDURE RUN;
VARIABLE ORDER 1; VARIABLE DIM 1; VARIABLE NM1 1;
PROCEDURE TESTEXP NM1;
    VARIABLE TEMP NM1; VARIABLE I 1;
    VARIABLE T1 1; VARIABLE T2 1;
    WRITE 6 '{name}';
    CPUSEC T1;
    LOOP I 1 {iterations}; TEMP:={cosy_expr} + DA(1)*0; ENDLOOP;
    CPUSEC T2;
    WRITE 6 'TIME_SEC ' T2-T1;
    WRITE 6 TEMP;
ENDPROCEDURE;
ORDER := {self.order}; DIM := {self.dimension};
DAINI ORDER DIM 0 NM1;
TESTEXP NM1;
ENDPROCEDURE;
RUN;
END;
"""
                with open(fox_path, "w") as f:
                    f.write(cosy_script)
                with open(dat_path, "w") as f_dat:
                    f_dat.write(os.path.splitext(script_name)[0])

                with open(dat_path, "r") as dat_file:
                    try:
                        last_process = subprocess.run(
                            [COSY_BIN],
                            stdin=dat_file,
                            capture_output=True,
                            text=True,
                            check=True,
                            cwd=ARTIFACTS_DIR,
                            timeout=timeout,
                        )
                    except subprocess.TimeoutExpired:
                        print(f"  [Raw COSY] Timed out after {timeout}s")
                        return {}, np.nan, np.nan

                # Parse TIME_SEC from output
                output_lines = last_process.stdout.strip().split("\n")
                run_time = None
                for i, line in enumerate(output_lines):
                    if "TIME_SEC" in line:
                        parts = line.strip().split()
                        # Case 1: TIME_SEC 1.23E-02 (Same line)
                        if len(parts) >= 2:
                            try:
                                run_time = float(parts[1])
                                break
                            except:
                                pass
                        # Case 2: TIME_SEC \n 1.23E-02 (Next line)
                        if i + 1 < len(output_lines):
                            try:
                                next_line = output_lines[i + 1].strip()
                                run_time = float(next_line)
                                break
                            except:
                                pass

                if run_time is not None:
                    # Normalize by iterations to get time per operation
                    timings.append(run_time / iterations)

            if not timings:
                return None, None, None

            avg_time = np.mean(timings)
            std_time = np.std(timings)
            return self.parse_cosy_output(last_process.stdout), avg_time, std_time
        except Exception as e:
            import sys
            print(f"Raw COSY failed: {e}", file=sys.stderr)
            return {}, np.nan, np.nan

    @staticmethod
    def format_time(seconds, std_dev=None):
        """Formats time in seconds to a string with appropriate units (s, ms, µs)."""
        if pd.isna(seconds):
            return "N/A"

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
        if pd.isna(ratio) or ratio == 0:
            return "N/A"
        if ratio == float("inf"):
            return "Inf"
        if ratio >= 10:
            return f"{int(ratio)}x"
        return f"{ratio:.2f}x"

    def parse_cosy_output(self, output):
        """Parses COSY output to extract coefficients dictionary."""
        coefficients = {}
        lines = output.strip().split("\n")
        start = False
        for line in lines:
            line = line.strip()
            if line.startswith("I  COEFFICIENT"):
                start = True
                continue
            if line.startswith("----") and start:
                break
            if start and line:
                parts = line.split()
                try:
                    coeff = float(parts[1])
                    exp = tuple(int(e) for e in parts[3:])
                    coefficients[exp] = coeff
                except:
                    continue
        return coefficients

    @staticmethod
    def calculate_rmse(coeffs_a, coeffs_b):
        """Calculates RMSE between two coefficient dictionaries."""
        if not coeffs_a or not coeffs_b:
            return np.nan
        all_keys = set(coeffs_a.keys()) | set(coeffs_b.keys())
        if not all_keys:
            return 0.0
        sq_errors = [
            (coeffs_a.get(k, 0.0) - coeffs_b.get(k, 0.0)) ** 2 for k in all_keys
        ]
        return np.sqrt(np.mean(sq_errors))

    @staticmethod
    def save_markdown_results(df, title):
        """Saves results dataframe as a timestamped Markdown table."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{title.lower().replace(' ', '_')}_{timestamp}.md"
        filepath = os.path.join(ARTIFACTS_DIR, filename)

        with open(filepath, "w") as f:
            f.write(f"# {title} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            try:
                f.write(df.to_markdown(index=False))
            except ImportError:
                f.write(df.to_string(index=False))
            f.write("\n")

        print(f"Results saved to {filepath}")
        return filepath

    def generate_plots(self, full_results):
        """Generates plots for timing vs order and returns as base64 strings."""
        plots = {}
        df = pd.DataFrame(full_results)

        # Plot 1: Timing vs Order for different variants (Python vs S-Cosy)
        # We'll pick a few representative ops or just average all
        for op in df["Operation"].unique():
            plt.figure(figsize=(10, 6))
            op_df = df[df["Operation"] == op]

            for vars_count in op_df["Variables"].unique():
                v_df = op_df[op_df["Variables"] == vars_count]
                # Filter out NaNs for plotting
                v_df_py = v_df.dropna(subset=["Python Time (s)"])
                v_df_sc = v_df.dropna(subset=["SCosy Time (s)"])
                v_df_raw = v_df.dropna(subset=["Raw-Cosy Time (s)"])

                plt.plot(
                    v_df_py["Order"],
                    v_df_py["Python Time (s)"],
                    marker="o",
                    label=f"Python (v={vars_count})",
                )
                plt.plot(
                    v_df_sc["Order"],
                    v_df_sc["SCosy Time (s)"],
                    marker="s",
                    label=f"S-Cosy (v={vars_count})",
                )
                if not v_df_raw.empty:
                    plt.plot(
                        v_df_raw["Order"],
                        v_df_raw["Raw-Cosy Time (s)"],
                        marker="^",
                        linestyle="--",
                        label=f"Raw-Cosy (v={vars_count})",
                    )

            plt.title(f"Timing vs Order: {op}")
            plt.xlabel("Order")
            plt.ylabel("Time (s)")
            plt.yscale("log")
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.legend()

            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            plots[op] = base64.b64encode(buf.getvalue()).decode("utf-8")

        return plots

    def get_system_info(self):
        """Returns a dictionary containing system information."""
        import platform
        import subprocess
        from datetime import datetime

        import psutil

        # Get GCC/GFortran version
        try:
            gcc_v = (
                subprocess
                .check_output(["gfortran", "--version"])
                .decode()
                .split("\n")[0]
            )
        except:
            gcc_v = "gfortran not found"

        info = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Host": platform.node(),
            "OS": f"{platform.system()} {platform.release()}",
            "CPU": platform.processor(),
            "CPU Cores": f"{psutil.cpu_count(logical=False)}C / {psutil.cpu_count(logical=True)}T",
            "RAM": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            "Python": platform.python_version(),
            "Compiler": gcc_v,
            "Compiler Flags": "-O3 -march=native -ffixed-form -flto -funroll-loops -std=legacy",
        }
        return info

    def format_dataframe_for_display(self, df):
        """Formats the dataframe values (times to ms/us, speedups to 2f) for HTML display."""
        df_disp = df.copy()
        time_cols = [c for c in df.columns if "Time" in c]

        for col in time_cols:
            # Convert seconds float to readable string
            df_disp[col] = df_disp[col].apply(lambda x: self.format_time(x))
            # Rename column to remove (s)
            new_name = col.replace(" (s)", "")
            df_disp.rename(columns={col: new_name}, inplace=True)

        # Format Speedup columns
        speed_cols = [c for c in df_disp.columns if "Speedup" in c]
        for col in speed_cols:
            df_disp[col] = df_disp[col].apply(lambda x: self.format_speedup(x))

        return df_disp

    def generate_html_report(self, full_results, plots, method_info=None):
        """Generates a styled HTML report with tables and embedded plots."""
        if method_info is None:
            method_info = {}
        df = pd.DataFrame(full_results)
        sys_info = self.get_system_info()

        # Format DF for display
        df_display = self.format_dataframe_for_display(df)

        # System Info Table
        sys_rows = "".join([
            f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>"
            for k, v in sys_info.items()
        ])

        # Methodology Text

        # Export CSV (raw numbers)
        csv_path = os.path.join(
            ARTIFACTS_DIR,
            f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        df.to_csv(csv_path, index=False)
        print(f"Raw CSV exported to: {csv_path}")
        iters = method_info.get("iterations", 100)
        report_title = method_info.get("title", "Sandalwood Benchmark Report")

        # HTML Template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_title}</title>
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
                .summary, .sysinfo, .methodology {{ background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .sysinfo table {{ width: auto; min-width: 50%; border: none; box-shadow: none; margin: 0; }}
            /* Sortable Table Styles */
            th {{ cursor: pointer; position: relative; }}
            th:hover {{ background-color: #2980b9; }}
            th::after {{ content: '↕'; position: absolute; right: 5px; opacity: 0.5; font-size: 0.8em; }}
            </style>
            <script>
            function sortTable(n) {{
              var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
              table = document.getElementById("benchmarkTable");
              switching = true;
              dir = "asc"; 
              while (switching) {{
                switching = false;
                rows = table.rows;
                for (i = 1; i < (rows.length - 1); i++) {{
                  shouldSwitch = false;
                  x = rows[i].getElementsByTagName("TD")[n];
                  y = rows[i + 1].getElementsByTagName("TD")[n];
                  
                  var xVal = x.getAttribute("data-val");
                  var yVal = y.getAttribute("data-val");
                  if (xVal === null) xVal = x.innerHTML.toLowerCase();
                  if (yVal === null) yVal = y.innerHTML.toLowerCase();
                  
                  var xNum = parseFloat(xVal);
                  var yNum = parseFloat(yVal);
                  if (!isNaN(xNum) && !isNaN(yNum)) {{ xVal = xNum; yVal = yNum; }}

                  if (dir == "asc") {{
                    if (xVal > yVal) {{ shouldSwitch = true; break; }}
                  }} else if (dir == "desc") {{
                    if (xVal < yVal) {{ shouldSwitch = true; break; }}
                  }}
                }}
                if (shouldSwitch) {{
                  rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                  switching = true;
                  switchcount ++;      
                }} else {{
                  if (switchcount == 0 && dir == "asc") {{
                    dir = "desc";
                    switching = true;
                  }}
                }}
              }}
            }}
            </script>
        </head>
        <body>
            <h1>{report_title}</h1>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="sysinfo">
                <h2>System Configuration</h2>
                <table>
                    {sys_rows}
                </table>
            </div>

            <div class="summary">
                <p><strong>Summary:</strong> This report compares the performance of Sandalwood's Python backend, Sandalwood's COSY backend (S-Cosy), and direct COSY script execution (Raw-Cosy) across various expansion orders and variables.</p>
            </div>
            
            <div class="methodology">
                <h2>Benchmark Methodology</h2>
                <ul>
                    <li><strong>Iterations:</strong> Each operation is executed <b>{iters}</b> times inside the benchmark loop to average out jitter.</li>
                    <li><strong>Raw COSY Overhead Compensation:</strong> We use the COSY internal procedure <code>CPUSEC</code> to measure the CPU time directly around the benchmark loop inside the compiled script. This completely excludes process startup/shutdown time and compilation overhead, providing a highly accurate measurement of the computation itself without need for external compensation.</li>
                    <li><strong>Metrics:</strong> 
                        <ul>
                            <li><b>Speedup (vs Python):</b> <code>Time(Python) / Time(S-Cosy)</code>. Higher is better.</li>
                            <li><b>Speedup (vs Raw COSY):</b> <code>Time(Raw COSY) / Time(S-Cosy)</code>. Higher is better. Values near 1.0 indicate S-Cosy matches native performance.</li>
                        </ul>
                    </li>
                    <li><strong>Dynamic Expressions:</strong> Expressions now utilize all active variables (e.g., for 4 variables and <code>Sin</code>, we compute <code>sin(x+y+z+u)</code>). This ensures complexity scales with dimensionality.</li>
                    <li><strong>Execution Timeout:</strong> Benchmarks exceeding the configured timeout (default 30s) are skipped to prevent hanging on excessively complex cases.</li>
                    <li><strong>Note on Power Operation:</strong> For the <code>Pow</code> benchmark, the standard power operator <code>^</code> is not supported for DA objects in the available COSY binary. Therefore, <code>(1+x)**3</code> is implemented using explicit multiplication: <code>(1+x)*(1+x)*(1+x)</code>.</li>
                </ul>
            </div>

            <h2>Performance Comparison Table</h2>
            <p><em>Click column headers to sort.</em></p>
            """

        # Build Custom Sortable Table
        cols = [
            ("Operation", "Operation"),
            ("Variables", "Variables"),
            ("Order", "Order"),
            ("Python Time (s)", "Python Time"),
            ("SCosy Time (s)", "S-COSY Time"),
            ("Raw-Cosy Time (s)", "Raw COSY Time"),
            ("Speedup (vs Python)", "Speedup (vs Py)"),
            ("Speedup (vs Raw COSY)", "Speedup (vs Raw)"),
            ("Python Expr", "Py Expr"),
            ("COSY Expr", "COSY Expr"),
        ]

        html += "<table id='benchmarkTable'><thead><tr>"
        for i, (key, label) in enumerate(cols):
            html += f"<th onclick='sortTable({i})'>{label}</th>"
        html += "</tr></thead><tbody>"

        for row in full_results:
            html += "<tr>"
            for key, label in cols:
                val = row.get(key, "")
                display_val = val
                sort_val = val

                if "Time" in key:
                    display_val = self.format_time(val)
                    if pd.isna(val):
                        sort_val = 999999
                elif "Speedup" in key:
                    display_val = self.format_speedup(val)
                    if pd.isna(val):
                        sort_val = -1

                html += f"<td data-val='{sort_val}'>{display_val}</td>"
            html += "</tr>"
        html += "</tbody></table>"

        html += """
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
