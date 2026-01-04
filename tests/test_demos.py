
import pytest
import os
import sys
import subprocess
import shutil
import tempfile
import json

@pytest.mark.parametrize("backend", ["python", "cosy"])
def test_beginner_demos_backend(backend):
    """Verify beginner demos work with the specified backend."""
    if backend == "cosy":
        try:
            from sandalwood.taylor_function import _COSY_BACKEND_AVAILABLE
            if not _COSY_BACKEND_AVAILABLE:
                pytest.skip("COSY backend not available")
        except ImportError:
            pytest.skip("Could not check COSY availability")

    demo_root = os.path.abspath("demos")
    
    notebooks_to_test = []
    for root, _, files in os.walk(demo_root):
        # Skip checkpoints
        if ".ipynb_checkpoints" in root:
            continue
        for f in files:
            if f.endswith(".ipynb"):
                notebooks_to_test.append(os.path.join(root, f))
    
    if not notebooks_to_test:
        pytest.fail(f"No notebooks found in {demo_root}")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Patch and run each notebook
        for src_path in notebooks_to_test:
            nb_name = os.path.basename(src_path)
            dst_path = os.path.join(temp_dir, nb_name)
            
            # Copy and patch
            with open(src_path, "r") as f:
                nb = json.load(f)
            
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    source = cell["source"]
                    for i, line in enumerate(source):
                        if "mtf.initialize_mtf(max_order=" in line:
                            if "implementation=" in line:
                                 new_line = line.replace('implementation="cosy"', f'implementation="{backend}"')
                                 new_line = new_line.replace('implementation="python"', f'implementation="{backend}"')
                                 new_line = new_line.replace('implementation="cpp"', f'implementation="{backend}"')
                                 source[i] = new_line
                            else:
                                 new_line = line.replace(")", f', implementation="{backend}")')
                                 source[i] = new_line
            
            with open(dst_path, "w") as f:
                json.dump(nb, f)
            
            # Execute
            env = os.environ.copy()
            env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
            
            cmd = [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook", "--execute",
                "--inplace", dst_path
            ]
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if result.returncode != 0:
                pytest.fail(f"Notebook {nb_name} failed with backend {backend}:\n{result.stderr}")
