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
# All rights reserved. May not be used, modified, or copied
# without permission
#
#----------------------------------------------------------------------------
#
# users: set your architecture, options, and paths in this file only
# you should not need to touch the other makefiles in the project
#
#----------------------------------------------------------------------------

# 1. Set your architecture here

ARCH = MAC_OSX
#ARCH = LINUX
#ARCH = BGP

# 2. Set your dependency paths here

DIY_INC = -I$(HOME)/diy/include
QHULL_INC =-I$(HOME)/software/qhull-2011.2/src/libqhull
CGAL_INC =-I/usr/include
PNETCDF_INC = -I$(HOME)/software/parallel-netcdf-1.3.0/include
DIY_LIB = -L$(HOME)/diy/lib -ldiy
QHULL_LIB = -L$(HOME)/software/qhull-2011.2/lib -lqhullstatic
CGAL_LIB = -L/usr/lib -lgmp -lCGAL
PNETCDF_LIB = -L$(HOME)/software/parallel-netcdf-1.3.0/lib -lpnetcdf

CGAL_CCFLAGS=-frounding-math -DCGAL_DISABLE_ROUNDING_MATH_CHECK

DELAUNAY_INC=${QHULL_INC}
DELAUNAY_LIB=${QHULL_LIB}
#DELAUNAY_INC=${CGAL_INC}
#DELAUNAY_LIB=${CGAL_LIB}
#DELAUNAY_CCFLAGS=${CGAL_CCFLAGS}

# 3. Set your build options here

TIMING = -DTIMING
