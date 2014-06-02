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

INCLUDES = -I../include $(DIY_INC) $(DELAUNAY_INC) $(PNETCDF_INC)
LIBS = $(DIY_LIB) $(DELAUNAY_LIB) $(PNETCDF_LIB) -L. -ltess

###########

ifeq ($(ARCH), MAC_OSX) # mac version

CCFLAGS += -g -Wall -Wextra
CCFLAGS += -DMAC_OSX $(TIMING) -fopenmp
CCFLAGS += ${DELAUNAY_CCFLAGS}
GL_LIBS = -framework GLUT -framework OpenGL

endif

###########

ifeq ($(ARCH), LINUX) # linux version

CCFLAGS += -g
CCFLAGS += -DLINUX $(TIMING)
CCFLAGS += ${DELAUNAY_CCFLAGS}
GL_LIBS = -lglut -lGLU -lGL

endif

###########

ifeq ($(ARCH), BGQ) # BG/Q version

CCFLAGS += -O3
#CCFLAGS += -g
CCFLAGS += -DBGQ $(TIMING)
CCFLAGS += ${DELAUNAY_CCFLAGS}
GL_LIBS =

endif

###########

TESS_OBJS = tess.o io.o swap.o tet.o utils.o dense.o ser_io.o $(TESS_CONV_OBJ)

.SUFFIXES: .c .cpp

.c.o:
	$(CC) -c $(CCFLAGS) $(INCLUDES) $<

.cpp.o:
	$(CXX) -c $(CCFLAGS) $(INCLUDES) $<

default: all

all: libtess.a

libtess.a: $(TESS_OBJS)
	rm -f $@
	ar -cru $@ $(TESS_OBJS)
	mv libtess.a ../lib

clean:
	rm -f *.o *.a
