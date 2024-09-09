/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

/*
 * This computes the Mandelbrot set: the output image is split in horizontal
 * stripes, which are computed in parallel.  We also make the same computation
 * several times, so that OpenGL interaction allows to browse through the set.
 */

#include <starpu.h>
#include <math.h>
#include <limits.h>
#include <time.h>

static int nblocks_p = 20;
static int height_p = 400;
static int width_p = 640;
static int maxIt_p = 20000; /* max number of iteration in the Mandelbrot function */
static int niter_p = 10; /* number of timing loops */

static double leftX_p = -0.745;
static double rightX_p = -0.74375;
static double topY_p = .15;
static double bottomY_p = .14875;

/*
 *	OpenCL kernel
 */

char *mandelbrot_opencl_src = "\
#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n				\
#define MIN(a,b) (((a)<(b))? (a) : (b))					\n \
__kernel void mandelbrot_kernel(__global unsigned* a,			\n \
	double leftX, double topY,					\n \
	double stepX, double stepY,					\n \
	int maxIt, int iby, int block_size, int width)			\n \
{									\n \
	size_t id_x = get_global_id(0);	\n				\
	size_t id_y = get_global_id(1);	\n				\
	if ((id_x < width) && (id_y < block_size))			\n \
	{								\n \
	double xc = leftX + id_x * stepX;				\n \
	double yc = topY - (id_y + iby*block_size) * stepY;		\n \
	int it;								\n \
	double x,y;							\n \
	x = y = (double)0.0;						\n \
	for (it=0;it<maxIt;it++)					\n \
	{								\n \
		double x2 = x*x;					\n \
		double y2 = y*y;					\n \
		if (x2+y2 > 4.0) break;					\n \
		double twoxy = (double)2.0*x*y;				\n \
		x = x2 - y2 + xc;					\n \
		y = twoxy + yc;						\n \
	}								\n \
	unsigned int v = MIN((1024*((float)(it)/(2000))), 256);		\n \
	a[id_x + width * id_y] = (v<<16|(255-v)<<8);			\n \
	}								\n \
}";

static struct starpu_opencl_program opencl_programs;

static void compute_block_opencl(void *descr[], void *cl_arg)
{
  int iby, block_size;
  double stepX, stepY;
  int *pcnt; /* unused for CUDA tasks */
  starpu_codelet_unpack_args(cl_arg, &iby, &block_size, &stepX, &stepY, &pcnt);

  cl_mem data = (cl_mem)STARPU_VECTOR_GET_DEV_HANDLE(descr[0]);

  cl_kernel kernel;
  cl_command_queue queue;
  cl_int err;

  int id = starpu_worker_get_id_check();
  int devid = starpu_worker_get_devid(id);

  err = starpu_opencl_load_kernel(&kernel, &queue, &opencl_programs, "mandelbrot_kernel", devid);
  if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

  clSetKernelArg(kernel, 0, sizeof(data), &data);
  clSetKernelArg(kernel, 1, sizeof(leftX_p), &leftX_p);
  clSetKernelArg(kernel, 2, sizeof(topY_p), &topY_p);
  clSetKernelArg(kernel, 3, sizeof(stepX), &stepX);
  clSetKernelArg(kernel, 4, sizeof(stepY), &stepY);
  clSetKernelArg(kernel, 5, sizeof(maxIt_p), &maxIt_p);
  clSetKernelArg(kernel, 6, sizeof(iby), &iby);
  clSetKernelArg(kernel, 7, sizeof(block_size), &block_size);
  clSetKernelArg(kernel, 8, sizeof(width_p), &width_p);

  unsigned dim = 16;
  size_t local[2] = {dim, 1};
  size_t global[2] = {width_p, block_size};
  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
  if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
  starpu_opencl_release_kernel(kernel);
}

/*
 *	CPU kernel
 */

static void compute_block(void *descr[], void *cl_arg)
{
  int iby, block_size;
  double stepX, stepY;
  int *pcnt; /* unused for sequential tasks */

  starpu_codelet_unpack_args(cl_arg, &iby, &block_size, &stepX, &stepY, &pcnt);

  unsigned *data = (unsigned *)STARPU_VECTOR_GET_PTR(descr[0]);

  int local_iy;
  for (local_iy = 0; local_iy < block_size; local_iy++)
    {
      int ix, iy;

      iy = iby*block_size + local_iy;
      for (ix = 0; ix < width_p; ix++)
        {
          double cx = leftX_p + ix * stepX;
          double cy = topY_p - iy * stepY;
          /* Z = X+I*Y */
          double x = 0;
          double y = 0;
          int it;
          for (it = 0; it < maxIt_p; it++)
            {
              double x2 = x*x;
              double y2 = y*y;

              /* Stop iterations when |Z| > 2 */
              if (x2 + y2 > 4.0)
                break;

              double twoxy = 2.0*x*y;

              /* Z = Z^2 + C */
              x = x2 - y2 + cx;
              y = twoxy + cy;
            }

          unsigned int v = STARPU_MIN((1024*((float)(it)/(2000))), 256);
          data[ix + local_iy*width_p] = (v<<16|(255-v)<<8);
        }
    }
}

static struct starpu_codelet mandelbrot_cl =
  {
    .type = STARPU_SEQ,
    .cpu_funcs = {compute_block},
    .where = STARPU_CPU|STARPU_OPENCL,
    .opencl_funcs = {compute_block_opencl},
    .opencl_flags = {STARPU_OPENCL_ASYNC},
    .nbuffers = 1
  };

static void parse_args(int argc, char **argv)
{
  int i;
  for (i = 1; i < argc; i++)
    {
      if (strcmp(argv[i], "-h") == 0) {
        fprintf(stderr, "Usage: %s [-h] [ -width 800] [-height 600] [-nblocks %d] [-pos leftx:rightx:bottomy:topy] [-niter 1000] [-max-iter]\n", argv[0], nblocks_p);
        exit(-1);
      }

      if (strcmp(argv[i], "-width") == 0) {
          char *argptr;
          width_p = strtol(argv[++i], &argptr, 10);
        }

      if (strcmp(argv[i], "-height") == 0) {
          char *argptr;
          height_p = strtol(argv[++i], &argptr, 10);
        }

      if (strcmp(argv[i], "-nblocks") == 0) {
          char *argptr;
          nblocks_p = strtol(argv[++i], &argptr, 10);
        }

      if (strcmp(argv[i], "-niter") == 0) {
          char *argptr;
          niter_p = strtol(argv[++i], &argptr, 10);
        }

      if (strcmp(argv[i], "-pos") == 0) {
        int ret = sscanf(argv[++i], "%lf:%lf:%lf:%lf", &leftX_p, &rightX_p,
                         &bottomY_p, &topY_p);
        assert(ret == 4);
      }

      if (strcmp(argv[i], "-max_iter") == 0) {
        char *argptr;
        maxIt_p = strtol(argv[++i], &argptr, 10);
      }
    }
}

double seconds() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ((double)ts.tv_nsec) / 1e9;
}

int main(int argc, char **argv)
{
  int ret;

  parse_args(argc, argv);

  /* We don't use CUDA in that example */
  struct starpu_conf conf;
  starpu_conf_init(&conf);
  conf.ncuda = 0;

  ret = starpu_init(&conf);
  STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

  unsigned *buffer;
  starpu_malloc((void **)&buffer, height_p*width_p*sizeof(unsigned));

  int block_size = height_p/nblocks_p;
  STARPU_ASSERT((height_p % nblocks_p) == 0);

  starpu_opencl_load_opencl_from_string(mandelbrot_opencl_src, &opencl_programs, NULL);

  starpu_data_handle_t block_handles[nblocks_p];
  int iby;
  for (iby = 0; iby < nblocks_p; iby++) {
    unsigned *data = &buffer[iby*block_size*width_p];
    starpu_vector_data_register(&block_handles[iby], STARPU_MAIN_RAM,
                                (uintptr_t)data, block_size*width_p, sizeof(unsigned));
  }

  double start = seconds();

  for (int i = 0; i < niter_p; i++) {
    double stepX = (rightX_p - leftX_p)/width_p;
    double stepY = (topY_p - bottomY_p)/height_p;

    int per_block_cnt[nblocks_p];

    starpu_iteration_push(i);

    for (iby = 0; iby < nblocks_p; iby++) {
      per_block_cnt[iby] = 0;
      int *pcnt = &per_block_cnt[iby];

      ret = starpu_task_insert(&mandelbrot_cl,
                               STARPU_VALUE, &iby, sizeof(iby),
                               STARPU_VALUE, &block_size, sizeof(block_size),
                               STARPU_VALUE, &stepX, sizeof(stepX),
                               STARPU_VALUE, &stepY, sizeof(stepY),
                               STARPU_W, block_handles[iby],
                               STARPU_VALUE, &pcnt, sizeof(int *),
                               STARPU_TAG_ONLY, ((starpu_tag_t)i)*nblocks_p + iby,
                               0);
      STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
    }

    starpu_iteration_pop();
  }

  starpu_task_wait_for_all();

  double end = seconds();

  printf("Average runtime: %.2f seconds\n", (end-start)/niter_p);

  for (iby = 0; iby < nblocks_p; iby++) {
    starpu_data_unregister(block_handles[iby]);
  }

  starpu_shutdown();

  return 0;
}
