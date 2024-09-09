CC?=gcc
CFLAGS=$(shell pkg-config starpu-1.4 --cflags) -Wall -Wextra -Wno-deprecated-declarations
LDFLAGS=$(shell pkg-config starpu-1.4 --libs) -lOpenCL

mandelbrot: mandelbrot.c
	$(CC) mandelbrot.c -o mandelbrot $(CFLAGS) $(LDFLAGS)
