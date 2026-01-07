#!/bin/bash

# ==============================================================================
# Sandalwood Benchmark Wrapper
# ==============================================================================
# This script provides a simplified interface for running the Sandalwood 
# benchmark suite. It handles virtual environment usage and supports 
# multiple benchmarking modes.
#
# Usage:
#   ./benchmark.sh [mode] [options]
#
# Modes:
#   ops     - Individual operation benchmarks (Python vs COSY backend).
#   raw     - Three-way comparison (Python vs COSY backend vs Raw COSY).
#   batch   - Performance of evaluating many points via neval().
#   profile - Run cProfile on core operations.
#   full    - Comprehensive parametric sweep (Orders 2-10, Vars 4-6) + HTML Report.
#
# Options:
#   --order N   - The Taylor expansion order (default: 8).
#   --dims N    - Number of variables/dimensions (default: 4).
#   --iters N   - Number of iterations per benchmark (default: 100).
#   --npoints N - Number of points for batch evaluation (default: 10000).
#
# Results:
#   Results are printed to the console and saved as Markdown tables in
#   the artifacts/ directory.
# ==============================================================================

# Directory where this script resides
BENCH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$BENCH_DIR/../../" && pwd )"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python3"

# Check if .venv exists
if [ ! -f "$PYTHON_BIN" ]; then
    echo "Error: Virtual environment not found at $ROOT_DIR/.venv"
    echo "Please run 'uv venv' and 'uv pip install -e .' first."
    exit 1
fi

# Check if cosy_bin exists, if not compile it
COSY_BIN="$BENCH_DIR/cosy_bin"
COSY_SRC_DIR="$ROOT_DIR/src/sandalwood/backends/cosy/cosy_src"

if [ ! -f "$COSY_BIN" ]; then
    echo "Compiling COSY binary..."
    if command -v gfortran &> /dev/null; then
        # Compilation Flags Explanation:
        # -std=legacy: Downgrade modern strictness to support older Fortran constructs
        # -ffixed-form: Treat source as fixed-form Fortran 77
        # -O3: Maximum stable optimization level
        # -march=native: Optimize for host architecture
        # -flto: Link Time Optimization for cross-file inlining
        # -funroll-loops: Aggressive loop unrolling
        gfortran -std=legacy -ffixed-form -O3 -march=native -flto -funroll-loops -o "$COSY_BIN" \
            "$COSY_SRC_DIR/foxy.f" \
            "$COSY_SRC_DIR/dafox.f" \
            "$COSY_SRC_DIR/foxfit.f" \
            "$COSY_SRC_DIR/foxgraf.f"
        echo "Compilation complete."
    else
        echo "Error: gfortran not found. Cannot compile COSY binary."
        echo "Please install gfortran."
        exit 1
    fi
fi

# Check if COSY.bin exists, if not generate it
COSY_LIB_BIN="$BENCH_DIR/COSY.bin"
if [ ! -f "$COSY_LIB_BIN" ]; then
    echo "Generating COSY.bin..."
    # Create foxyinp.dat for COSY compilation
    echo "COSY" > "$BENCH_DIR/foxyinp.dat"

    # Run cosy_bin to compile COSY.fox -> COSY.bin
    # We must run inside BENCH_DIR so it finds COSY.fox
    (cd "$BENCH_DIR" && ./cosy_bin < foxyinp.dat)

    if [ ! -f "$COSY_LIB_BIN" ]; then
        echo "Error: Failed to generate COSY.bin"
        exit 1
    fi
    echo "COSY.bin generated."
fi

# Default parameters
MODE="raw"
ORDER=8
DIMS=4
ITERS=100
NPOINTS=10000
MEMORY_FLAG=""
FILTER_ARG=""

# Simple argument parsing
while [[ $# -gt 0 ]]; do
  case $1 in
    ops|raw|batch|profile|full)
      MODE="$1"
      shift
      ;;
    --order)
      ORDER="$2"
      shift 2
      ;;
    --dims)
      DIMS="$2"
      shift 2
      ;;
    --iters)
      ITERS="$2"
      shift 2
      ;;
    --npoints)
      NPOINTS="$2"
      shift 2
      ;;
    --memory)
      MEMORY_FLAG="--memory"
      shift 1
      ;;
    --filter)
      FILTER_ARG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [ops|raw|batch|profile] [--order N] [--dims N] [--iters N] [--npoints N] [--memory] [--filter pattern]"
      exit 1
      ;;
  esac
done

echo "--- Sandalwood Benchmark: $MODE ---"
echo "Parameters: Order=$ORDER, Dims=$DIMS, Iters=$ITERS, Points=$NPOINTS"
if [ ! -z "$MEMORY_FLAG" ]; then echo "Memory Profiling: Enabled"; fi
if [ ! -z "$FILTER_ARG" ]; then echo "Filter: $FILTER_ARG"; fi
echo "---"

# Execute the runner
CMD=("$PYTHON_BIN" "$BENCH_DIR/run.py" \
    --mode "$MODE" \
    --order "$ORDER" \
    --dims "$DIMS" \
    --iters "$ITERS" \
    --npoints "$NPOINTS")

if [ ! -z "$MEMORY_FLAG" ]; then CMD+=("--memory"); fi
if [ ! -z "$FILTER_ARG" ]; then CMD+=("--filter" "$FILTER_ARG"); fi

"${CMD[@]}"

echo "--- Done ---"
