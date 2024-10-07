CC?=cc
CFLAGS=$(shell pkg-config starpu-1.4 --cflags) -Wall -Wextra -Wno-deprecated-declarations -Ofast
LDFLAGS=$(shell pkg-config starpu-1.4 --libs) -lOpenCL -lcuda -lcudart -lcublas -lcblas

.PHONY: all

all: spmv dw_block_spmv

spmv_cuda.o: spmv_cuda.cu
	nvcc spmv_cuda.cu -c

spmv: spmv.c spmv_cuda.o
	$(CC) spmv.c spmv_kernels.c spmv_cuda.o -o spmv $(CFLAGS) $(LDFLAGS) -lstdc++ -I../

dw_block_spmv: spmv.c
	$(CC) dw_block_spmv.c dw_block_spmv_kernels.c matrix_market/mmio.c matrix_market/mm_to_bcsr.c -o dw_block_spmv $(CFLAGS) $(LDFLAGS) -I../
