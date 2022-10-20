CXX_FLAGS=-std=c++11 -O3 -I../include \
		  -use_fast_math -lineinfo \
		  --compiler-options "-Wall -Wextra" \
		  -gencode=arch=compute_80,code=sm_80 \
		  -gencode=arch=compute_86,code=sm_86 \
		  -gencode=arch=compute_70,code=sm_70 \
		  -lcublas

all: matmul

matmul: main.cu
	nvcc ${CXX_FLAGS} main.cu -o matmul 
