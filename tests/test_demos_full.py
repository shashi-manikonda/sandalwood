import json
import os
import subprocess
import sys
import tempfile

import pytest


@pytest.mark.demo
@pytest.mark.parametrize("backend", ["python", "cosy"])
def test_all_demos_backend(backend):
    """Verify all demos (.ipynb and .py) work with the specified backend."""
    if backend == "cosy":
        try:
            from sandalwood.taylor_function import _COSY_BACKEND_AVAILABLE

            if not _COSY_BACKEND_AVAILABLE:
                pytest.skip("COSY backend not available")
        except ImportError:
            pytest.skip("Could not check COSY availability")

    demo_root = os.path.abspath("demos")

    files_to_test = []
    for root, _, files in os.walk(demo_root):
        if ".ipynb_checkpoints" in root:
            continue
        for f in files:
            if f.endswith((".ipynb", ".py")):
                files_to_test.append(os.path.join(root, f))

    if not files_to_test:
        pytest.fail(f"No demos found in {demo_root}")

    with tempfile.TemporaryDirectory() as temp_dir:
        for src_path in files_to_test:
            fname = os.path.basename(src_path)
            dst_path = os.path.join(temp_dir, fname)

            if fname.endswith(".ipynb"):
                _test_notebook(src_path, dst_path, backend)
            else:
                _test_script(src_path, dst_path, backend)


def _patch_line(line, backend):
    """Patches mtf.initialize_mtf line to use the specified backend."""
    if "mtf.initialize_mtf(max_order=" in line:
        if "implementation=" in line:
            line = line.replace('implementation="cosy"', f'implementation="{backend}"')
            line = line.replace(
                'implementation="python"', f'implementation="{backend}"'
            )
            line = line.replace('implementation="cpp"', f'implementation="{backend}"')
        else:
            line = line.replace(")", f', implementation="{backend}")')
    return line


def _test_notebook(src_path, dst_path, backend):
    with open(src_path, "r") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = cell["source"]
            for i, line in enumerate(source):
                source[i] = _patch_line(line, backend)

    with open(dst_path, "w") as f:
        json.dump(nb, f)

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        os.path.join(os.getcwd(), "src") + os.pathsep + env.get("PYTHONPATH", "")
    )
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        dst_path,
    ]

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        if ("NotImplementedError" in result.stderr and (
            "implemented for Python backend" in result.stderr
            or "only available for the COSY backend" in result.stderr
        )) or ("RuntimeError" in result.stderr and "COSY backend not initialized" in result.stderr):
            print(f"Skipping notebook {os.path.basename(src_path)} due to unimplemented features.")
            return

        pytest.fail(
            f"Notebook {os.path.basename(src_path)} failed with backend {backend}:\n{result.stderr}"
        )


def _test_script(src_path, dst_path, backend):
    with open(src_path, "r") as f:
        lines = f.readlines()

    with open(dst_path, "w") as f:
        for line in lines:
            f.write(_patch_line(line, backend))

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        os.path.join(os.getcwd(), "src") + os.pathsep + env.get("PYTHONPATH", "")
    )
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    cmd = [sys.executable, dst_path]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        if ("NotImplementedError" in result.stderr and (
            "implemented for Python backend" in result.stderr
            or "only available for the COSY backend" in result.stderr
        )) or ("RuntimeError" in result.stderr and "COSY backend not initialized" in result.stderr):
            print(f"Skipping script {os.path.basename(src_path)} due to unimplemented features.")
            return

        pytest.fail(
            f"Script {os.path.basename(src_path)} failed with backend {backend}:\n{result.stderr}"
        )
