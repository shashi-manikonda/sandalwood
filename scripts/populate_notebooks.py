import os
import shutil
import subprocess
import sys


def get_installer():
    """Returns the installer command (uv or pip)."""
    if shutil.which("uv"):
        return ["uv", "pip", "install"]
    return [sys.executable, "-m", "pip", "install"]


def install_dependencies():
    """Installs the necessary dependencies for running the notebooks."""
    dependencies = [
        "jupytext",
        "matplotlib",
        "pandas",
        "torch",
        "ipython",
        "sympy",
        "nbconvert",
    ]
    for dependency in dependencies:
        print(f"Installing {dependency}...")
        try:
            subprocess.run(get_installer() + [dependency], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {dependency}: {e}")
            sys.exit(1)


def run_notebooks_in_directory(root_directory):
    """
    Finds and executes all .ipynb files in a given directory and its subdirectories.
    """
    # Add the src directory to the python path
    project_root = os.getcwd()
    src_path = os.path.join(project_root, "src")

    # Set up a new environment for the subprocess call
    env = os.environ.copy()
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    # *** FIX for OMP: Error #15 ***
    # This allows the program to continue even with duplicate OpenMP library loads.
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith(".ipynb"):
                print(f"Executing notebook: {os.path.join(dirpath, filename)}")
                filepath = os.path.join(dirpath, filename)
                try:
                    # Construct the nbconvert command
                    command = [
                        sys.executable,
                        "-m",
                        "jupyter",
                        "nbconvert",
                        "--to",
                        "notebook",
                        "--execute",
                        filepath,
                        "--inplace",
                    ]

                    # Run the command with the modified environment
                    subprocess.run(
                        command, check=True, capture_output=True, text=True, env=env
                    )
                    print(f"--- Successfully executed {filename} ---")
                except subprocess.CalledProcessError as e:
                    print(f"*** Failed to execute {filename}:")
                    print(f"Standard Output:\n{e.stdout}")
                    print(f"Standard Error:\n{e.stderr}")
                    # If one notebook fails, we should stop
                    sys.exit(1)


if __name__ == "__main__":
    # install_dependencies()
    # Get the current working directory
    demos_directory = os.path.join(os.getcwd(), "demos")
    if not os.path.exists(demos_directory):
        print(f"Error: The directory '{demos_directory}' does not exist.")
        sys.exit(1)

    try:
        run_notebooks_in_directory(demos_directory)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
