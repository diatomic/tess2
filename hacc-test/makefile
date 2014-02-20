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
TESS = $(HOME)/tess
#TESS = $(HOME)/tess-cgal-install

include ../user_defs.mk

ifeq ($(ARCH), MAC_OSX) # mac version
GENERICIO = $(HOME)/software/genericio/
endif
ifeq ($(ARCH), LINUX) # linux version
GENERICIO = $(HOME)/software/genericio/
endif
ifeq ($(ARCH), BGQ) # BG/Q version
GENERICIO = $(HOME)/software/genericio-gcc/
endif

INCLUDES = $(DIY_INC) $(DELAUANAY_INC) $(PNETCDF_INC) -I../lib -I$(GENERICIO)
LIBS = -L$(TESS)/lib -ltess $(DIY_LIB) $(DELAUNAY_LIB) $(PNETCDF_LIB) -L$(GENERICIO)/genericio.build/libs -lGenericIO

###########

ifeq ($(ARCH), MAC_OSX) # mac version

CCFLAGS = -g
CCFLAGS += -DMAC_OSX
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

all: hacc-voronoi

hacc-voronoi: $(OBJS)
	$(CXX) -o hacc-voronoi $(OBJS) $(LIBS)

clean:
	rm -f *.o 
