## Compilation notes

If you are lucky, then `pkg-config` may work for you. If not, I found
the following necessary (with StarPU installed in `~/starpu`).

```
STARPU_DIR=~/starpu
export CPATH=$STARPU_DIR/include/starpu/1.4/:$CPATH
export LD_LIBRARY_PATH=$STARPU_DIR/lib/:$LD_LIBARY_PATH
export PKG_CONFIG_PATH=$STARPU_DIR/lib/pkgconfig/:$PKG_CONFIG_PATH
```

## To use only GPU (no hybrid)

Change the `.where` field of the `starpu_codelet` to remove `STARPU_CPU`.

## Results for `mandelbrot`

`STARPU_SCHED=dmdas` works slightly better than the default, although
the difference is not great.

Linear speedup when using two GPUs exclusively (but `-nblocks` must be
pretty low, e.g. 20), but significant slowdown when CPU is also used.

```
$ ./mandelbrot -nblocks 20 -niter 10 -max_iter 10000 -height 20000 -width 20000
```
