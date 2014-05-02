#----------------------------------------------------------------------------
#
# Tom Peterka
# Argonne National Laboratory
# 9700 S. Cass Ave.
# Argonne, IL 60439
# tpeterka@mcs.anl.gov
#
# (C) 2011 by Argonne National Laboratory.
# See COPYRIGHT in top-level directory.
#
#----------------------------------------------------------------------------

include ../user_defs.mk

INCLUDES = $(DIY_INC) $(DELAUANAY_INC) $(PNETCDF_INC) -I../lib
LIBS = -L../lib -ltess $(DIY_LIB) $(DELAUNAY_LIB) $(PNETCDF_LIB)

###########

ifeq ($(ARCH), MAC_OSX) # mac version

CCFLAGS = -g
CCFLAGS += -DMAC_OSX -fopenmp
CCFLAGS += ${DELAUNAY_CCFLAGS}

endif

###########

ifeq ($(ARCH), LINUX) # linux version

CCFLAGS = -g
CCFLAGS += -DLINUX
CCFLAGS += ${DELAUNAY_CCFLAGS}

endif

###########


ifeq ($(ARCH), BGQ) # BG/Q version

CCFLAGS = -O3
#CCFLAGS = -g
CCFLAGS += -DBGQ $(TIMING)
CCFLAGS += ${DELAUNAY_CCFLAGS}
GL_LIBS =

endif

###########

OBJS = main.o


.SUFFIXES: .cpp

.cpp.o:
	$(CXX) -c $(CCFLAGS) $(INCLUDES) $<

default: all

all: post-voronoi

post-voronoi: $(OBJS) ../lib/libtess.a
	$(CXX) -o post-voronoi $(OBJS) $(LIBS)

clean:
	rm -f *.o 
