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
	cd src && $(MAKE) $@
	cd examples/test-tess && $(MAKE) $@
	cd examples/dense && $(MAKE) $@
	cd examples/post-tess && $(MAKE) $@
	cd examples/tess-dense && $(MAKE) $@


