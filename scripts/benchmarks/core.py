import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from sandalwood import mtf

# Local COSY configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
COSY_BIN = os.path.join(BASE_DIR, "cosy_bin")
COSY_FOX_SRC = os.path.join(BASE_DIR, "COSY.fox")

# Ensure artifacts directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# MTF to COSY function name mapping
MTF_TO_COSY = {
    "mtf.sin": "SIN", "mtf.cos": "COS", "mtf.exp": "EXP",
    "mtf.sqrt": "SQRT", "mtf.log": "LOG", "mtf.arctan": "ATAN",
    "mtf.tan": "TAN", "mtf.arcsin": "ASIN", "mtf.arccos": "ACOS",
    "mtf.sinh": "SINH", "mtf.cosh": "COSH", "mtf.tanh": "TANH",
}

class BenchmarkEngine:
    def __init__(self, order, dimension):
        self.order = order
        self.dimension = dimension
        self.variable_names = ["x", "y", "z", "u", "v", "w"][:dimension]
        
    def setup_mtf(self, implementation):
        """Initializes Sandalwood MTF with the specified backend."""
        mtf.initialize_mtf(max_order=self.order, max_dimension=self.dimension, implementation=implementation)
        globals_dict = {"mtf": mtf}
        for i, vn in enumerate(self.variable_names):
            globals_dict[vn] = mtf.var(i+1)
        return globals_dict

    def run_sandalwood(self, expression, implementation, iterations):
        """Runs a benchmark on a Sandalwood expression."""
        globals_dict = self.setup_mtf(implementation)
        start = time.perf_counter()
        for _ in range(iterations):
            res = eval(expression, globals_dict)
        elapsed = time.perf_counter() - start
        
        # Extract coefficients
        exponents = res.exponents
        coeffs = res.coeffs
        return {tuple(exp): c for exp, c in zip(exponents, coeffs)}, elapsed

    def run_raw_cosy(self, name, cosy_expr, iterations):
        """Runs a benchmark using Raw COSY script execution."""
        script_name = f"tmp_{name}.fox"
        fox_path = os.path.join(ARTIFACTS_DIR, script_name)
        dat_path = os.path.join(ARTIFACTS_DIR, "foxyinp.dat")
        
        # Prepare system files in artifacts dir if they don't exist
        for f in ["COSY.bin", "DAINI.DAT"]:
             src = os.path.join(BASE_DIR, f)
             dst = os.path.join(ARTIFACTS_DIR, f)
             if os.path.exists(src) and not os.path.exists(dst):
                 import shutil
                 shutil.copy(src, dst)

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

        start = time.perf_counter()
        try:
            with open(dat_path, 'r') as dat_file:
                process = subprocess.run([COSY_BIN], stdin=dat_file, capture_output=True, text=True, check=True, cwd=ARTIFACTS_DIR)
            elapsed = time.perf_counter() - start
            return self.parse_cosy_output(process.stdout), elapsed
        except Exception as e:
            print(f"Raw COSY failed: {e}")
            return None, None

    @staticmethod
    def format_time(seconds):
        """Formats time in seconds to a string with appropriate units (s, ms, µs)."""
        if pd.isna(seconds): return "N/A"
        if seconds < 1e-6:
            return f"{seconds * 1e9:.2f} ns"
        elif seconds < 1e-3:
            return f"{seconds * 1e6:.2f} µs"
        elif seconds < 1:
            return f"{seconds * 1e3:.2f} ms"
        else:
            return f"{seconds:.4f} s"

    @staticmethod
    def format_speedup(ratio):
        """Formats a speedup ratio to a meaningful string."""
        if pd.isna(ratio) or ratio == 0: return "N/A"
        if ratio == float('inf'): return "Inf"
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
