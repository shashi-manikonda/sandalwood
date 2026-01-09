
import sys
import os

print(f"Python executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")

try:
    import sandalwood
    print(f"SUCCESS: Imported sandalwood version: {sandalwood.__version__}")
except ImportError as e:
    print(f"FAILURE: Could not import sandalwood: {e}")
    sys.exit(1)

try:
    from sandalwood.backends.cosy import cosy_backend
    print(f"Cosy backend available: {cosy_backend.COSY_AVAILABLE}")
    
    if not cosy_backend.COSY_AVAILABLE:
        print("Note: COSY backend is NOT available. This is expected if MinGW is missing.")
    else:
        print("SUCCESS: COSY backend loaded successfully!")
except Exception as e:
    print(f"FAILURE: Error checking COSY backend: {e}")

# Test basic MTF functionality (Python implementation)
try:
    from sandalwood import mtf
    print("Initializing MTF (Python fallback expected)...")
    mtf.initialize_mtf(max_order=2, max_dimension=1, implementation="python")
    
    x = mtf.var(1)
    f = x + 2.0
    print(f"Created function f = x + 2.0: {f}")
    
    val = f.eval([1.0])
    print(f"Evaluated f(1.0) = {val[0]}")
    
    if abs(val[0] - 3.0) < 1e-9:
        print("SUCCESS: Basic arithmetic calculation correct.")
    else:
        print(f"FAILURE: Basic arithmetic calculation incorrect. Expected 3.0, got {val[0]}")
        
except Exception as e:
    print(f"FAILURE: Basic MTF functionality test failed: {e}")
    import traceback
    traceback.print_exc()

print("Verification script finished.")
