import subprocess
import sys


def test_populate_notebooks():
    """
    Tests that the populate_notebooks.py script runs without errors.
    """
    command = [sys.executable, "scripts/populate_notebooks.py"]
    result = subprocess.run(command, capture_output=True, text=True)

    assert result.returncode == 0, (
        f"populate_notebooks.py failed with exit code {result.returncode}.\\n"
        f"Stderr:\\n{result.stderr}"
    )
