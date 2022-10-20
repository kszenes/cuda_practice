CXX_FLAGS=-std=c++11 -O3 -I../include \
		  -use_fast_math -lineinfo -lcublas \
		  --compiler-options "-Wall -Wextra" \
		  -gencode=arch=compute_86,code=sm_86 \
		  -gencode=arch=compute_80,code=sm_80 \
		  -gencode=arch=compute_70,code=sm_70

all: saxpy

saxpy: main.cu
	nvcc main.cu -o saxpy ${CXX_FLAGS}

ptx: main.cu
	nvcc -arch=sm_80 main.cu -c ${CXX_FLAGS}
	cuobjdump -ptx main.o > saxpy.ptx
