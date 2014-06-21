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
LIBS = $(DIY_LIB) $(DELAUNAY_LIB) $(PNETCDF_LIB) -L../lib -ltess

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

DRAW_OBJS = draw.o

.SUFFIXES: .c .cpp

.c.o:
	$(CC) -c $(CCFLAGS) $(INCLUDES) $<

.cpp.o:
	$(CXX) -c $(CCFLAGS) $(INCLUDES) $<

default: all

ifeq ($(ARCH), BGQ)
all:
else
all: draw
endif

draw: $(DRAW_OBJS) 
	$(CXX) -o draw $(DRAW_OBJS) $(GL_LIBS) $(LIBS)

clean:
	rm -f *.o draw
