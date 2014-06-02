#----------------------------------------------------------------------------
#
# makefile user definitions
#
# Tom Peterka
# Argonne National Laboratory
# 9700 S. Cass Ave.
# Argonne, IL 60439
# tpeterka@mcs.anl.gov
#
#----------------------------------------------------------------------------
#
# users should set architecture, options, and paths in this file only
# and should not need to touch the other makefiles in the project
#
#----------------------------------------------------------------------------

# 1. Set your architecture here

ARCH = MAC_OSX
#ARCH = LINUX
#ARCH = BGQ

# 2. Set your choice of computational geometry library here (QHULL or CGAL)

CONV = QHULL
#CONV = CGAL

# 3. Select whether and what type of SMP threading is used (optional)

#TESS_THREAD = TESS_OMP
# define new modes, PTHERAD, TBB, CUDA, etc. as they are implemented

# 4. Set your dependency paths and build options here

ifeq ($(ARCH), BGQ) # BG/Q version

ifeq ($(CONV), QHULL)
CC=/soft/compilers/wrappers/xl/mpicc
CXX=/soft/compilers/wrappers/xl/mpicxx
else
CC=/soft/compilers/wrappers/gcc/mpicc
CXX=/soft/compilers/wrappers/gcc/mpicxx
endif

DIY_INC = -I$(HOME)/diy-gcc/include
QHULL_INC =-I$(HOME)/software/qhull-2011.2/src/libqhull
CGAL_INC =-I/$(HOME)/software/CGAL-4.3/include -I/soft/libraries/boost/current/cnk-gcc/current/include -I/usr/include
PNETCDF_INC = -I$(HOME)/software/parallel-netcdf-1.3.0-gcc/include
DIY_LIB = $(HOME)/diy-gcc/lib/libdiy.a
QHULL_LIB = -L$(HOME)/software/qhull-2011.2/lib -lqhullstatic
CGAL_LIB = -dynamic \
	/home/tpeterka/software/gmp-5.1.3/lib/libgmp.a \
	/home/tpeterka/software/CGAL-4.3-install/lib/libCGAL.a \
	/soft/libraries/boost/current/cnk-gcc/current/lib/libboost_thread-mt.a \
	/soft/libraries/boost/current/cnk-gcc/current/lib/libboost_system-mt.a \
	-lmpich-gcc -lopa-gcc -lmpl-gcc -lpami-gcc -Wl,-Bstatic -lSPI \
	-lSPI_cnk -Wl,-Bdynamic -lrt -lpthread -lstdc++ -lmpichcxx-gcc
PNETCDF_LIB = -L$(HOME)/software/parallel-netcdf-1.3.0-gcc/lib -lpnetcdf

else

CC=mpicc
CXX=mpicxx
DIY_INC = -I$(HOME)/diy/include
QHULL_INC =-I$(HOME)/software/qhull-2011.2/src/libqhull
CGAL_INC =-I/opt/local/include
PNETCDF_INC = -I$(HOME)/software/parallel-netcdf-1.3.0/include
DIY_LIB = $(HOME)/diy/lib/libdiy.a
QHULL_LIB = -L$(HOME)/software/qhull-2011.2/lib -lqhullstatic
CGAL_LIB = -L/opt/local/lib -lgmp -lCGAL  \
	$(HOME)/software/mpich-3.0.4-install/lib/libpmpich.a \
	$(HOME)/software/mpich-3.0.4-install/lib/libmpich.a
PNETCDF_LIB = -L$(HOME)/software/parallel-netcdf-1.3.0/lib -lpnetcdf

endif

TIMING = -DTIMING
CCFLAGS = -DPNETCDF_IO

#----------------------------------------------------------------------------
#
# users should not need to edit beyond this point
#
#----------------------------------------------------------------------------
ifeq ($(TESS_THREAD), TESS_OMP)
CCFLAGS += -DTESS_OMP
endif

ifeq ($(CONV), QHULL) # qhull version

DELAUNAY_INC = ${QHULL_INC}
DELAUNAY_LIB = ${QHULL_LIB}
TESS_CONV_OBJ = tess-qhull.o

endif

ifeq ($(CONV), CGAL) # cgal version

ifeq ($(ARCH), BGQ)
CGAL_CCFLAGS=-frounding-math -DCGAL_USE_MPFR -DCGAL_USE_GMP \
	-DTESS_CGAL_ALLOW_SPATIAL_SORT -DNDEBUG -O3
else
CGAL_CCFLAGS=-frounding-math -DCGAL_USE_MPFR -DCGAL_USE_GMP \
	-DTESS_CGAL_ALLOW_SPATIAL_SORT -pipe -Os -arch x86_64 \
	-fno-strict-aliasing -O3
endif

DELAUNAY_INC=${CGAL_INC}
DELAUNAY_LIB=${CGAL_LIB}
DELAUNAY_CCFLAGS=${CGAL_CCFLAGS}
TESS_CONV_OBJ = tess-cgal.o

endif

#----------------------------------------------------------------------------
