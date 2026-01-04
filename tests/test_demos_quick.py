
import pytest
import os
import sys
import subprocess
import json
import tempfile

def find_demos():
    """Recursively finds all .ipynb and .py files in the demos directory."""
    demo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demos"))
    demos = []
    for root, _, files in os.walk(demo_root):
        if ".ipynb_checkpoints" in root:
            continue
        for f in files:
            if f.endswith((".ipynb", ".py")):
                demos.append(os.path.join(root, f))
    return demos

@pytest.mark.parametrize("demo_path", find_demos())
def test_demo_quick(demo_path):
    """
    Fast verification of a demo file.
    Runs the code in a subprocess to ensure isolation and speed.
    """
    fname = os.path.basename(demo_path)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        exec_path = os.path.join(temp_dir, "run_demo.py")
        
        if demo_path.endswith(".ipynb"):
            # Extract code from notebook
            with open(demo_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            
            code_lines = []
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    source = cell.get("source", [])
                    if isinstance(source, str):
                        code_lines.append(source)
                    else:
                        code_lines.extend(source)
                    code_lines.append("\n")
            
            with open(exec_path, "w", encoding="utf-8") as f:
                f.write("import matplotlib\n")
                f.write("matplotlib.use('Agg')\n") # Disable GUI
                f.write("".join(code_lines))
        else:
            # For .py files, we can just run them directly (or wrap to disable GUI)
            with open(demo_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            with open(exec_path, "w", encoding="utf-8") as f:
                f.write("import matplotlib\n")
                f.write("matplotlib.use('Agg')\n")
                f.write(content)

        # Run in subprocess
        env = os.environ.copy()
        # Ensure src is in PYTHONPATH
        src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        result = subprocess.run(
            [sys.executable, exec_path],
            capture_output=True,
            text=True,
            env=env,
            timeout=60 # Reasonable timeout for a single demo
        )

        if result.returncode != 0:
            pytest.fail(f"Demo {fname} failed execution:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
