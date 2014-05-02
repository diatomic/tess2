#----------------------------------------------------------------------------
#
# top-level makefile
#
# Tom Peterka
# Argonne National Laboratory
# 9700 S. Cass Ave.
# Argonne, IL 60439
# tpeterka@mcs.anl.gov
#
# (C) 2012 by Argonne National Laboratory.
# See COPYRIGHT in top-level directory.
#
#----------------------------------------------------------------------------
default: all

all clean:
	cd lib && $(MAKE) $@
	cd tess-driver && $(MAKE) $@
	cd dense-driver && $(MAKE) $@
	cd post-tess-driver && $(MAKE) $@


