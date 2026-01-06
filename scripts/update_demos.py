
import glob
import json
import os
import subprocess
import sys


def update_notebook(filepath):
    print(f"Processing {filepath}...")
    with open(filepath, 'r') as f:
        nb = json.load(f)

    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                original_line = line
                # Replace implementations
                if 'implementation' in line:
                    if '"python"' in line:
                        line = line.replace('"python"', '"cosy"')
                    if "'python'" in line:
                        line = line.replace("'python'", "'cosy'")
                    if '"cpp"' in line:
                        line = line.replace('"cpp"', '"cosy"')
                    if "'cpp'" in line:
                        line = line.replace("'cpp'", "'cosy'")
                
                # If explicit backend print is hardcoded in specific demo text, we might leave it
                # or rely on the actual print output updating.
                
                if line != original_line:
                    modified = True
                new_source.append(line)
            cell['source'] = new_source

    # Explicitly force COSY if not mentioned but initialize is called?
    # No, default is now COSY. We trust the default.
    
    if modified:
        print("  - Modified backend to COSY in source.")
    
    with open(filepath, 'w') as f:
        json.dump(nb, f, indent=1)

    # Execute
    print("  - Executing...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.getcwd(), "src") + os.pathsep + env.get("PYTHONPATH", "")
    # Ensure COSY libs are found if needed (setup usually handles rpath but just in case)
    
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", 
        "--execute", 
        "--inplace", 
        filepath
    ]
    
    try:
        subprocess.run(cmd, env=env, check=True, capture_output=True)
        print("  - Done.")
    except subprocess.CalledProcessError as e:
        print(f"  - FAILED executon: {e}")
        print(e.stderr.decode() if e.stderr else "No stderr")

def main():
    demos_dir = os.path.abspath("demos")
    notebooks = glob.glob(os.path.join(demos_dir, "**/*.ipynb"), recursive=True)
    
    for nb in notebooks:
        if ".ipynb_checkpoints" in nb:
            continue
        update_notebook(nb)

if __name__ == "__main__":
    main()
