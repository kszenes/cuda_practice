CXX_FLAGS=-std=c++11 -O3 -I./include \
		  -use_fast_math -lineinfo -lcublas \
		  --compiler-options "-Wall -Wextra" \
		  -gencode=arch=compute_86,code=sm_86 \
		  -gencode=arch=compute_80,code=sm_80 \
		  -gencode=arch=compute_70,code=sm_70 

.PHONY: all clean
all: vanilla_attention

vanilla_attention: main.cu include
	nvcc main.cu -lcusparse -o vanilla_attention ${CXX_FLAGS}

clean:
	rm -rf vanilla_attention
