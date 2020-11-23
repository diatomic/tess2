## Parallel Delaunay and Voronoi Tessellation and Density Estimation

## Licensing

Tess is released as open source software under a BSD style [license](./COPYING).

## Requirements

- a C++11 compiler
- the [DIY](https://github.com/diatomic/diy) block parallel library
- either the [Qhull](http://qhull.org/) or [CGAL](http://www.cgal.org/) computational geometry library

## Installation

The following instructions assume Qhull is the serial geometry engine.

Build Dependencies

a. [DIY](https://github.com/diatomic/diy)

```
git clone https://github.com/diatomic/diy2
```

b. [Qhull](http://qhull.org/)

```
wget http://www.qhull.org/download/qhull-2020-src-8.0.2.tgz
tar -xvf qhull-2020-src-8.0.2.tgz
cd qhull-2020.2/
make
```

Build Tess

```
git clone https://github.com/diatomic/tess2

cmake /path/to/tess \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_C_COMPILER=mpicc \
-DCMAKE_INSTALL_PREFIX=/path/to/tess2/install \
-Dserial=QHull \
-DDIY_INCLUDE_DIRS=/path/to/diy/include \
-DQHull_INCLUDE_DIRS=/path/to/qhull-2020.2/src/libqhull \
-DQHull_LIBRARY=/path/to/qhull-2020.2/lib/libqhullstatic.a

make
make install
```

Optionally, Tess can use [CGAL](http://www.cgal.org/) instead of Qhull. To do so,
pass `-Dserial=CGAL` as the choice to `cmake`.

## Execution

1. Test tessellation only

```
cd path/to/tess2/install/examples/tess
```

Edit TESS_TEST: select ARCH, num_procs, dsize (number of particles)

```
./TESS_TEST
path/to/tess2/install/tools/draw del.out
```

Mouse interaction with drawing: mouse move to rotate, ‘z’ + mouse up, down to zoom, ‘t’ to toggle voronoi tessellation, ‘y’ to toggle delaunay tessellation, ‘f’ to toggle shaded rendering

2. Test tessellation + density estimator

(from tess top level directory)

```
cd path/to/tess2/install/examples/tess-dense
```

Edit TESS_DENSE_TEST; select ARCH, num_procs, dsize (number of particles), gsize (number of grid points)

```
./TESS_DENSE_TEST
path/to/tess2/install/tools/dense-plot.py --raw=dense.raw --numpts=512
```
(assuming outfile was dense.raw and gsize was 512 512 512 in TESS_DENSE_TEST)

Dense-plot.py is a python script using numpy and matplotlib, but you can use your favorite visualization/plotting tool (VisIt, ParaView, R, Octave, Matlab, etc.) to plot the output. It is just an array of 32-bit floating-point density values listed in C-order (x changes fastest).
