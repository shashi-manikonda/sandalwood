#!/bin/bash
for file in dafox.f foxy.f foxfit.f foxgraf.f; do
    echo "Processing $file..."
    # Pass 1: IFOR -> GFOR
    printf "$file\n$file.tmp\n*IFOR\n*GFOR\n" | ./version
    mv $file.tmp $file
    # Pass 2: NORM -> MPI
    printf "$file\n$file.tmp\n*NORM\n*MPI\n" | ./version
    mv $file.tmp $file
done
