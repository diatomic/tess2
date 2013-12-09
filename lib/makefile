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

INCLUDES = $(DIY_INC) $(QHULL_INC) $(PNETCDF_INC)
LIBS = $(DIY_LIB) $(QHULL_LIB) $(PNETCDF_LIB)

###########

ifeq ($(ARCH), MAC_OSX) # mac version

CCFLAGS = -g -Wall -Wextra
CCFLAGS += -DMAC_OSX $(TIMING) -fopenmp
GL_LIBS = -framework GLUT -framework OpenGL
CC=mpicc
CXX=mpicxx

endif

###########

ifeq ($(ARCH), LINUX) # linux version

CCFLAGS = -g
CCFLAGS += -DLINUX $(TIMING)
GL_LIBS = -lglut -lGLU -lGL
CC=mpicc
CXX=mpicxx

endif

###########

ifeq ($(ARCH), BGP) # BG/P version

CCFLAGS = -O3 -qarch=450d -qtune=450
#CCFLAGS = -g
CCFLAGS += -DBGP $(TIMING)
GL_LIBS =
CC=mpixlc_r
CXX=mpixlcxx_r
#CC=mpicxx
#CXX=mpicxx

endif

###########

TESS_OBJS = tess.o io.o
DRAW_OBJS = draw.o ser_io.o swap.o io.o
DUMP_OBJS = dump.o ser_io.o swap.o
DENSE_OBJS = dense.o io.o

.SUFFIXES: .c .cpp

.c.o:
	$(CC) -c $(CCFLAGS) $(INCLUDES) $<

.cpp.o:
	$(CXX) -c $(CCFLAGS) $(INCLUDES) $<

default: all

ifeq ($(ARCH), BGP)
all: libtess.a dump
else
all: libtess.a draw dump dense
endif

libtess.a: $(TESS_OBJS)
	rm -f $@
	ar -cru $@ $(TESS_OBJS)

draw: $(DRAW_OBJS) 
	$(CXX) -o draw $(DRAW_OBJS) $(GL_LIBS) $(LIBS)
	mv draw ..

dump: $(DUMP_OBJS) 
	$(CXX) -o dump $(DUMP_OBJS)
	mv dump ..

dense: $(DENSE_OBJS) 
	mpicxx -o dense $(DENSE_OBJS) $(DIY_LIB) $(PNETCDF_LIB)
	mv dense ..

clean:
	rm -f *.o 
