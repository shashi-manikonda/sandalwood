#!/bin/bash
set -e

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COSY_SRC="$DIR/cosy_src"

# Output library
LIB_NAME="libcosy.so"
OUTPUT="$DIR/$LIB_NAME"

echo "Building COSY sources separately..."

# Cleanup old objects
rm -f "$DIR"/*.o "$DIR"/*.so

FFLAGS="-fPIC -fcommon -std=legacy -O3 -march=native -ffixed-form -flto -funroll-loops -fallow-argument-mismatch -fopenmp"

# Compile individual files
gfortran -c $FFLAGS "$COSY_SRC/dafox.f" -o "$DIR/dafox.o"
gfortran -c $FFLAGS "$COSY_SRC/foxfit.f" -o "$DIR/foxfit.o"
gfortran -c $FFLAGS "$COSY_SRC/foxgraf.f" -o "$DIR/foxgraf.o"
gfortran -c $FFLAGS "$COSY_SRC/helper.f" -o "$DIR/helper.o"
gfortran -c $FFLAGS "$DIR/wrapper.f" -o "$DIR/wrapper.o"

echo "Linking..."
gfortran -shared -flto -fopenmp -o "$OUTPUT" \
    "$DIR/dafox.o" \
    "$DIR/foxfit.o" \
    "$DIR/foxgraf.o" \
    "$DIR/helper.o" \
    "$DIR/wrapper.o"

echo "Compilation successful: $OUTPUT"
