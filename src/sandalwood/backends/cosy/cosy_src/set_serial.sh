#!/bin/bash
for file in dafox.f foxy.f foxfit.f foxgraf.f; do
    echo "Processing $file..."
    # Switch MPI -> NORM (Reverting to serial)
    printf "$file\n$file.tmp\n*MPI\n*NORM\n" | ./version
    mv $file.tmp $file
done
