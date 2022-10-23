CXX_FLAGS=-std=c++11 -O3 -I./include \
		  -use_fast_math -lineinfo -lcublas \
		  --compiler-options "-Wall -Wextra" \
		  -gencode=arch=compute_86,code=sm_86 \
		  -gencode=arch=compute_80,code=sm_80 \
		  -gencode=arch=compute_70,code=sm_70 \
		  -gencode=arch=compute_60,code=sm_60 


.PHONY: all clean
all: gat

gat: main.cu include
	nvcc main.cu -o gat ${CXX_FLAGS}

clean:
	rm -rf gat
