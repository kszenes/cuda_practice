CXX_FLAGS=-std=c++11 -O3 -I../include \
		  -use_fast_math -lineinfo -lcublas \
		  --compiler-options "-Wall -Wextra" \
		  -gencode=arch=compute_86,code=sm_86 \
		  -gencode=arch=compute_80,code=sm_80 \
		  -gencode=arch=compute_70,code=sm_70 

.PHONY: all clean
all: vanilla_attention

vanilla_attention: main.cu host_utils.h cuda_kernels.h sort_vector.h
	nvcc main.cu -lcusparse -o vanilla_attention ${CXX_FLAGS}

debug: main.cu cuda_kernels.cu cuda_kernels.h
	nvcc main.cu -g -G -o batched_dot ${CXX_FLAGS}

ptx: main.cu
	nvcc -arch=sm_80 main.cu -c ${CXX_FLAGS}
	cuobjdump -ptx batched_dot.o > batched_dot.ptx

clean:
	rm -rf *.o batched_dot
