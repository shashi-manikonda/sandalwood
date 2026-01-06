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

# Default parameters
MODE="raw"
ORDER=8
DIMS=4
ITERS=100
NPOINTS=10000

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
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [ops|raw|batch|profile] [--order N] [--dims N] [--iters N] [--npoints N]"
      exit 1
      ;;
  esac
done

echo "--- Sandalwood Benchmark: $MODE ---"
echo "Parameters: Order=$ORDER, Dims=$DIMS, Iters=$ITERS, Points=$NPOINTS"
echo "---"

# Execute the runner
"$PYTHON_BIN" "$BENCH_DIR/run.py" \
    --mode "$MODE" \
    --order "$ORDER" \
    --dims "$DIMS" \
    --iters "$ITERS" \
    --npoints "$NPOINTS"

echo "--- Done ---"
