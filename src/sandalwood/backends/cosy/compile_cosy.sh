#!/bin/bash
# ==============================================================================
# Sandalwood COSY Builder
# ==============================================================================
# Purpose:
#   Compiles the COSY Infinity Fortran core and the Sandalwood wrapper into a 
#   shared library (libcosy.so).
#
# Logic:
#   1. Internalizes dimensions and memory limits from 'cosy_config.env'.
#   2. Prepares a temporary 'build_tmp/' directory to avoid polluting 'cosy_src/'.
#   3. Patches the Fortran source code PARAMETER statements with configured values.
#   4. Compiles individual object files using gfortran with optimized flags.
#   5. Links everything into a shared object (.so) for Python CTypes consumption.
#
# Usage:
#   ./compile_cosy.sh
# ==============================================================================
set -e

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COSY_SRC_ORIG="$DIR/cosy_src"
BUILD_DIR="$DIR/build_tmp"
CONFIG_FILE="$DIR/cosy_config.env"

# Output library
LIB_NAME="libcosy.so"
OUTPUT="$DIR/$LIB_NAME"

echo "--- Sandalwood COSY Builder ---"

# 1. Load Configuration
if [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from $CONFIG_FILE..."
    source "$CONFIG_FILE"
else
    echo "No config file found. Using defaults."
    # Defaults matching standard COSY
    export COSY_LMEM=140000000
    export COSY_LVAR=10000000
    export COSY_LDIM=1000
    export COSY_LEA=100000
    export COSY_LIA=1400000
    export COSY_LNO=99
    export COSY_LNV=40
fi

echo "Configuration:"
echo "  LMEM: $COSY_LMEM"
echo "  LVAR: $COSY_LVAR"
echo "  LEA:  $COSY_LEA"

# 2. Prepare Build Directory (Copy sources to keep originals safe)
echo "Preparing build directory..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cp "$COSY_SRC_ORIG"/*.f "$BUILD_DIR/"
cp "$DIR/wrapper.f" "$BUILD_DIR/"

# 2.5 Compile and Run Version Utility (Switch code to GFOR and NORM)
echo "Switching code version to GFOR (gfortran) and NORM (serial)..."
gfortran "$BUILD_DIR/version.f" -o "$BUILD_DIR/version"

for f in "$BUILD_DIR"/*.f; do
    if [ "$(basename "$f")" != "version.f" ]; then
        # Switch IFOR to GFOR
        printf "$f\n$f.tmp\n*IFOR\n*GFOR\n" | "$BUILD_DIR/version" > /dev/null
        mv "$f.tmp" "$f"
        # Switch MPI to NORM
        printf "$f\n$f.tmp\n*MPI\n*NORM\n" | "$BUILD_DIR/version" > /dev/null
        mv "$f.tmp" "$f"
    fi
done
rm "$BUILD_DIR/version"

# 3. Patch the Source Files
# We use sed to find the PARAMETER lines and replace the numbers.
# The regex looks for "PARAMETER(LMEM=..." and replaces the value.

echo "Patching source files with new memory limits..."

# Helper function to patch a specific variable across all files
patch_param() {
    local param_name=$1
    local new_value=$2
    # This sed command looks for 'PARAMETER(...,param_name=NUMBER,...)' or 'PARAMETER(param_name=NUMBER)'
    # and replaces the number. It handles spaces and commas.
    # Note: This regex assumes the standard COSY formatting found in dafox.f/wrapper.f
    
    # Linux sed (GNU)
    sed -i "s/\($param_name *= *\)[0-9]\+/\1$new_value/g" "$BUILD_DIR"/*.f
}

patch_param "LMEM" "$COSY_LMEM"
patch_param "LVAR" "$COSY_LVAR"
patch_param "LDIM" "$COSY_LDIM"
patch_param "LEA"  "$COSY_LEA"
patch_param "LIA"  "$COSY_LIA"
patch_param "LNO"  "$COSY_LNO"
patch_param "LNV"  "$COSY_LNV"

# 4. Compile
echo "Compiling..."

# Cleanup old output
rm -f "$DIR"/*.so

FFLAGS="-fPIC -fcommon -std=legacy -O3 -march=native -ffixed-form -flto -funroll-loops -fallow-argument-mismatch -fopenmp"

# Compile individual files from the BUILD_DIR
gfortran -c $FFLAGS "$BUILD_DIR/dafox.f" -o "$BUILD_DIR/dafox.o"
gfortran -c $FFLAGS "$BUILD_DIR/foxfit.f" -o "$BUILD_DIR/foxfit.o"
gfortran -c $FFLAGS "$BUILD_DIR/foxgraf.f" -o "$BUILD_DIR/foxgraf.o"
gfortran -c $FFLAGS "$BUILD_DIR/helper.f" -o "$BUILD_DIR/helper.o"
gfortran -c $FFLAGS "$BUILD_DIR/wrapper.f" -o "$BUILD_DIR/wrapper.o"

echo "Linking..."
gfortran -shared -flto -fopenmp -o "$OUTPUT" \
    "$BUILD_DIR/dafox.o" \
    "$BUILD_DIR/foxfit.o" \
    "$BUILD_DIR/foxgraf.o" \
    "$BUILD_DIR/helper.o" \
    "$BUILD_DIR/wrapper.o"

# Optional: Cleanup build dir
# rm -rf "$BUILD_DIR"

echo "Compilation successful: $OUTPUT"
