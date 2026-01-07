#!/bin/bash
set -e

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COSY_SRC="$DIR/cosy_src"

# Output library
LIB_NAME="libcosy.so"
OUTPUT="$DIR/$LIB_NAME"

echo "Compiling COSY sources from $COSY_SRC and wrapper.f from $DIR/wrapper.f..."
ls -l "$DIR/wrapper.f"

# Non-monolithic build with -fcommon
# Compilation Flags Explanation:
# -shared: Create a shared library (.so)
# -fPIC: Generate Position Independent Code (required for shared libs)
# -fcommon: Allow multiple definitions of common blocks (legacy Fortran behavior required by COSY)
# -std=legacy: Downgrade modern strictness to support older Fortran constructs
# -O3: Maximum stable optimization level (vectorization, inlining)
# -march=native: Optimize for the host CPU architecture (AVX, etc.)
# -ffixed-form: Treat source as fixed-form Fortran 77 (required for .f files)
# -flto: Link Time Optimization (cross-file inlining)
# -funroll-loops: Aggressive loop unrolling (efficient for large coefficient arrays)
gfortran -shared -fPIC -fcommon -std=legacy -O3 -march=native -ffixed-form -flto -funroll-loops \
    "$COSY_SRC/dafox.f" \
    "$COSY_SRC/foxfit.f" \
    "$COSY_SRC/foxgraf.f" \
    "$COSY_SRC/helper.f" \
    "$DIR/wrapper.f" \
    -fopenmp \
    -o "$OUTPUT"

echo "Compilation successful: $OUTPUT"
