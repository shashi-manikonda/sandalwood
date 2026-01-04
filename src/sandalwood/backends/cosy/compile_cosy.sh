#!/bin/bash
set -e

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COSY_SRC="$DIR/cosy_src"

# Output library
LIB_NAME="libcosy.so"
OUTPUT="$DIR/$LIB_NAME"

echo "Compiling COSY sources from $COSY_SRC and wrapper.f..."

gfortran -shared -fPIC -std=legacy -g -O2 \
    "$COSY_SRC/dafox.f" \
    "$COSY_SRC/foxfit.f" \
    "$COSY_SRC/foxgraf.f" \
    "$DIR/wrapper.f" \
    -fopenmp \
    -o "$OUTPUT"

echo "Compilation successful: $OUTPUT"
