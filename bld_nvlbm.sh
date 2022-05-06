#!/bin/bash

echo "Compile nlvbm.cu with nvcc"
echo "Assumes /usr/local/cuda is sym-linked to desired CUDA release"

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "Version..."
nvcc --version

nvcc -o nvlbm nvlbm.cu -lpthread

echo "Done"


