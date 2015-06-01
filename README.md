# Parallel Delaunay and Voronoi Tessellation and Density Estimation

# Licensing

Tess is released as open source software under a BSD style [license](./COPYING).

# Installation

(for tess based on qhull)


1. Install Dependencies

a. DIY

```
git clone https://github.com/diatomic/diy2
```

b. Qhull

```
wget http://www.qhull.org/download/qhull-2012.1-src.tgz
tar -xvf qhull-2012.1-src.tgz
cd qhull-2012.1-src
make
```

2. Install Tess

```
git clone https://github.com/diatomic/tess2
cd tess2
```

Configure using cmake:

```
cmake /path/to/tess \
-DCMAKE_CXX_COMPILER=mpicxx \
-DCMAKE_C_COMPILER=mpicc \
-Dserial=QHull \
-DDIY_INCLUDE_DIRS=/path/to/diy/include \
-DQHull_INCLUDE_DIRS=/path/to/qhull/include \
-DQHull_LIBRARY=/path/to/qhull/lib

make
```

# Execution

1. Test tessellation only

```
cd examples/tess
```

Edit TESS_TEST: select ARCH, num_procs, dsize (number of particles)

```
./TESS_TEST
../../tools/draw del.out.nc 0
```

Mouse interaction with drawing: mouse move to rotate, ‘z’ + mouse up, down to zoom, ‘t’ to toggle voronoi tessellation, ‘y’ to toggle delaunay tessellation, ‘f’ to toggle shaded rendering

2. Test tessellation + density estimator

(from tess top level directory)

```
cd examples/tess-dense
```

Edit TESS_DENSE_TEST; select ARCH, num_procs, dsize (number of particles), gsize (number of grid points)

```
./TESS_DENSE_TEST
../../tools/dense-plot.py --raw=dense.raw --numpts=512
```
(assuming outfile was dense.raw and gsize was 512 512 512 in TESS_DENSE_TEST)

Dense-plot.py is a python script using numpy and matplotlib, but you can use your favorite visualization/plotting tool (VisIt, ParaView, R, Octave, Matlab, etc.) to plot the output. It is just an array of 32-bit floating-point density values listed in C-order (x changes fastest).
