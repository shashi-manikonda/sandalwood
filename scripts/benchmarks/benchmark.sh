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
#   ops       - Individual operation benchmarks (Python vs COSY backend).
#   raw       - Three-way comparison (Python vs COSY backend vs Raw COSY).
#   batch     - Performance of evaluating many points via neval().
#   profile   - Run cProfile on core operations.
#   full      - Comprehensive parametric sweep (Orders 2-10, Vars 4-6) + HTML Report.
#   full_cosy - Same as full, but skips Python backend (COSY vs Raw COSY).
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
COSY_SRC_ORIG="$ROOT_DIR/src/sandalwood/backends/cosy/cosy_src"
CONFIG_FILE="$ROOT_DIR/src/sandalwood/backends/cosy/cosy_config.env"
BUILD_DIR="$BENCH_DIR/build_tmp"

if [ ! -f "$COSY_BIN" ]; then
    echo "Compiling COSY binary..."
    
    # 1. Load Configuration
    if [ -f "$CONFIG_FILE" ]; then
        echo "Loading configuration from $CONFIG_FILE..."
        source "$CONFIG_FILE"
    else
        echo "No config file found. Using defaults."
        export COSY_LMEM=140000000
        export COSY_LVAR=10000000
        export COSY_LDIM=1000
        export COSY_LEA=100000
        export COSY_LIA=1400000
        export COSY_LNO=99
        export COSY_LIA=1400000
        export COSY_LNO=99
        export COSY_LNV=40
        export COSY_COMPILER=gfortran
    fi

    # Detect Compiler
    if [ "$COSY_COMPILER" = "ifx" ]; then
        FC="ifx"
        VERSION_MARKER_OLD="*GFOR"
        VERSION_MARKER_NEW="*IFOR"
        FC_FLAGS="-O3 -march=native -fixed -qopenmp -diag-disable=10448"
    else
        FC="gfortran"
        VERSION_MARKER_OLD="*IFOR"
        VERSION_MARKER_NEW="*GFOR"
        FC_FLAGS="-std=legacy -ffixed-form -O3 -march=native -flto -funroll-loops"
    fi

    if command -v $FC &> /dev/null; then
        # 2. Prepare Build Directory
        echo "Preparing build directory..."
        rm -rf "$BUILD_DIR"
        mkdir -p "$BUILD_DIR"
        cp "$COSY_SRC_ORIG"/*.f "$BUILD_DIR/"

        # 2.5 Compile and Run Version Utility
        echo "Switching code version to $FC and NORM (serial)..."
        $FC "$BUILD_DIR/version.f" -o "$BUILD_DIR/version"

        for f in "$BUILD_DIR"/*.f; do
            if [ "$(basename "$f")" != "version.f" ]; then
                # Switch markers
                printf "$f\n$f.tmp\n$VERSION_MARKER_OLD\n$VERSION_MARKER_NEW\n" | "$BUILD_DIR/version" > /dev/null
                mv "$f.tmp" "$f"
                # Switch MPI to NORM
                printf "$f\n$f.tmp\n*MPI\n*NORM\n" | "$BUILD_DIR/version" > /dev/null
                mv "$f.tmp" "$f"
            fi
        done
        rm "$BUILD_DIR/version"

        # 3. Patch the Source Files
        echo "Patching source files with memory limits..."
        patch_param() {
            local param_name=$1
            local new_value=$2
            sed -i "s/\($param_name *= *\)[0-9]\+/\1$new_value/g" "$BUILD_DIR"/*.f
        }

        patch_param "LMEM" "$COSY_LMEM"
        patch_param "LVAR" "$COSY_LVAR"
        patch_param "LDIM" "$COSY_LDIM"
        patch_param "LEA"  "$COSY_LEA"
        patch_param "LIA"  "$COSY_LIA"
        patch_param "LNO"  "$COSY_LNO"
        patch_param "LNV"  "$COSY_LNV"

        # 5. Compile
        echo "Compiling with $FC..."
        $FC $FC_FLAGS -o "$COSY_BIN" \
            "$BUILD_DIR/foxy.f" \
            "$BUILD_DIR/dafox.f" \
            "$BUILD_DIR/foxfit.f" \
            "$BUILD_DIR/foxgraf.f"
        
        echo "Compilation complete."
        rm -rf "$BUILD_DIR"
    else
        echo "Error: Compiler '$FC' not found. Cannot compile COSY binary."
        echo "Please ensure your Fortran compiler is installed and in your PATH."
        if [ "$FC" = "ifx" ]; then
            echo "Tip: Try sourcing your Intel oneAPI environment (e.g., source /opt/intel/oneapi/setvars.sh)."
        fi
        exit 1
    fi
fi

# Check if COSY.bin exists, if not generate it
COSY_LIB_BIN="$BENCH_DIR/COSY.bin"
if [ ! -f "$COSY_LIB_BIN" ]; then
    echo "Generating COSY.bin..."

    # Ensure COSY.fox is present locally in BENCH_DIR. If not, try to fetch it.
    if [ ! -f "$BENCH_DIR/COSY.fox" ]; then
        if [ ! -z "$SANDALWOOD_COSY_SRC" ] && [ -f "$SANDALWOOD_COSY_SRC/cosy.fox" ]; then
            cp "$SANDALWOOD_COSY_SRC/cosy.fox" "$BENCH_DIR/COSY.fox"
        elif [ ! -z "$SANDALWOOD_COSY_SRC" ] && [ -f "$SANDALWOOD_COSY_SRC/../apps/cosy.fox" ]; then
            cp "$SANDALWOOD_COSY_SRC/../apps/cosy.fox" "$BENCH_DIR/COSY.fox"
        else
            echo "Error: COSY.fox not found in $BENCH_DIR."
            echo "Since COSY Infinity is proprietary, its files are not distributed with Sandalwood."
            echo "Please copy your licensed cosy.fox to $BENCH_DIR/COSY.fox or set the SANDALWOOD_COSY_SRC environment variable."
            exit 1
        fi
    fi

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
TIMEOUT=""

# Simple argument parsing
while [[ $# -gt 0 ]]; do
  case $1 in
    ops|raw|batch|batch_math|profile|full|full_cosy)
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
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [ops|raw|batch|profile|full|full_cosy] [--order N] [--dims N] [--iters N] [--npoints N] [--memory] [--filter pattern] [--timeout N]"
      exit 1
      ;;
  esac
done

echo "--- Sandalwood Benchmark: $MODE ---"
echo "Parameters: Order=$ORDER, Dims=$DIMS, Iters=$ITERS, Points=$NPOINTS"
if [ ! -z "$MEMORY_FLAG" ]; then echo "Memory Profiling: Enabled"; fi
if [ ! -z "$FILTER_ARG" ]; then echo "Filter: $FILTER_ARG"; fi
if [ ! -z "$TIMEOUT" ]; then echo "Timeout: $TIMEOUT"; fi
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
if [ ! -z "$TIMEOUT" ]; then CMD+=("--timeout" "$TIMEOUT"); fi


"${CMD[@]}"

echo "--- Done ---"
