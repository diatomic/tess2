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

#ARCH = MAC_OSX
#ARCH = LINUX
#ARCH = BGP
ARCH = BGQ

# 2. Set your choice of computational geometry library here (QHULL or CGAL)

#CONV = QHULL
CONV = CGAL

# 3. Set your dependency paths here

DIY_INC = -I$(HOME)/diy-gcc/include
QHULL_INC =-I$(HOME)/software/qhull-2011.2/src/libqhull
CGAL_INC =-I/$(HOME)/software/CGAL-4.3/include -I/soft/libraries/boost/current/cnk-gcc/current/include -I/usr/include
PNETCDF_INC = -I$(HOME)/software/parallel-netcdf-1.3.0/include
DIY_LIB = -L$(HOME)/diy-gcc/lib -ldiy
QHULL_LIB = -L$(HOME)/software/qhull-2011.2/lib -lqhullstatic
CGAL_LIB = -L$(HOME)/software/CGAL-4.3/lib -L$(HOME)/software/gmp-5.1.3/lib -lgmp -lCGAL -L$/soft/libraries/boost/current/cnk-gcc/current/lib
PNETCDF_LIB = -L$(HOME)/software/parallel-netcdf-1.3.0/lib -lpnetcdf

# 4. Set your build options here

CC=/soft/compilers/wrappers/gcc/mpicxx
CXX=/soft/compilers/wrappers/gcc/mpicxx
TIMING = -DTIMING

#----------------------------------------------------------------------------
#
# users should not need to edit beyond this point
#
#----------------------------------------------------------------------------

ifeq ($(CONV), QHULL) # qhull version

DELAUNAY_INC = ${QHULL_INC}
DELAUNAY_LIB = ${QHULL_LIB}
TESS_CONV_OBJ = tess-qhull.o

endif

ifeq ($(CONV), CGAL) # cgal version

CGAL_CCFLAGS=-frounding-math -DCGAL_DISABLE_ROUNDING_MATH_CHECK \
	     -DTESS_CGAL_ALLOW_SPATIAL_SOR # should be based on compiler check
#CGAL_CCFLAGS=-DCGAL_DISABLE_ROUNDING_MATH_CHECK \
	     -DTESS_CGAL_ALLOW_SPATIAL_SOR # should be based on compiler check
DELAUNAY_INC=${CGAL_INC}
DELAUNAY_LIB=${CGAL_LIB}
DELAUNAY_CCFLAGS=${CGAL_CCFLAGS}
TESS_CONV_OBJ = tess-cgal.o

endif

#----------------------------------------------------------------------------
