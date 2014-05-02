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

INCLUDES = $(DIY_INC) $(DELAUNAY_INC) $(PNETCDF_INC)
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

TESS_OBJS = tess.o io.o swap.o tet.o utils.o dense.o $(TESS_CONV_OBJ)
DRAW_OBJS = draw.o ser_io.o swap.o io.o
DUMP_OBJS = dump.o ser_io.o swap.o

.SUFFIXES: .c .cpp

.c.o:
	$(CC) -c $(CCFLAGS) $(INCLUDES) $<

.cpp.o:
	$(CXX) -c $(CCFLAGS) $(INCLUDES) $<

default: all

ifeq ($(ARCH), BGP)
all: libtess.a dump
else
ifeq ($(ARCH), BGQ)
all: libtess.a dump
else
all: libtess.a draw dump
endif
endif

libtess.a: $(TESS_OBJS)
	rm -f $@
	ar -cru $@ $(TESS_OBJS)

draw: $(DRAW_OBJS) 
	$(CXX) -o draw $(DRAW_OBJS) $(GL_LIBS) $(LIBS)

dump: $(DUMP_OBJS) 
	$(CXX) -o dump $(DUMP_OBJS)

clean:
	rm -f *.o 
