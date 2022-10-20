CXX_FLAGS=-std=c++11 -O3 -I../include \
		  -use_fast_math -lcublas \
		  -lineinfo \
		  --compiler-options "-Wall -Wextra" \
		  -gencode=arch=compute_86,code=sm_86 \
		  -gencode=arch=compute_80,code=sm_80 \
		  -gencode=arch=compute_70,code=sm_70 

all: outer_product batched_outer

outer_product: main.cu
	nvcc main.cu -o outer_product ${CXX_FLAGS}

batched_outer: batched_outer.cu
	nvcc batched_outer.cu -o batched_outer ${CXX_FLAGS}
