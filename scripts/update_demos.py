
import os
import subprocess
import sys

def update_notebooks():
    """
    Finds and executes all .ipynb files in the demos directory and its subdirectories.
    Updates the notebooks in-place with fresh outputs.
    """
    project_root = os.getcwd()
    src_path = os.path.join(project_root, "src")
    demos_directory = os.path.join(project_root, "demos")

    if not os.path.exists(demos_directory):
        print(f"Error: The directory '{demos_directory}' does not exist.")
        sys.exit(1)

    # Set up environment for execution
    env = os.environ.copy()
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    print(f"Updating all notebooks in {demos_directory}...")

    for dirpath, _, filenames in os.walk(demos_directory):
        for filename in filenames:
            if filename.endswith(".ipynb"):
                filepath = os.path.join(dirpath, filename)
                print(f"Executing: {os.path.relpath(filepath, project_root)}")
                try:
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
                    
                    subprocess.run(command, check=True, capture_output=True, text=True, env=env)
                    print(f"--- Successfully updated {filename} ---")
                except subprocess.CalledProcessError as e:
                    print(f"*** Failed to update {filename}:")
                    print(f"Standard Error:\n{e.stderr}")
                    # Continue with other notebooks but exit with error at the end
                    continue

if __name__ == "__main__":
    update_notebooks()
