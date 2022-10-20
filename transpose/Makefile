CXX_FLAGS=-std=c++11 -O3 -I../include \
		  -use_fast_math -lineinfo \
		  --compiler-options "-Wall -Wextra" \
		  -gencode=arch=compute_86,code=sm_86 \
		  -gencode=arch=compute_80,code=sm_80 \
		  -gencode=arch=compute_70,code=sm_70 

all: transpose transpose_ref transpose_practice

transpose: main.cu
	nvcc main.cu -o transpose ${CXX_FLAGS}
transpose_ref: transpose_ref.cu
	nvcc transpose_ref.cu -o transpose_ref ${CXX_FLAGS}
transpose_practice: transpose_practice.cu
	nvcc transpose_practice.cu -o transpose_practice ${CXX_FLAGS}
