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
distdir=tess-0.1

default: all

all clean: FORCE
	cd lib && $(MAKE) $@
	cd driver && $(MAKE) $@
	cd postprocessor && $(MAKE) $@
	rm -rf $(distdir)
	rm -f $(distdir).tar.gz

FORCE:
	cd lib && $(MAKE) clean
	cd driver && $(MAKE) clean
	cd postprocessor && $(MAKE) clean
	rm -rf $(distdir)
	rm -f $(distdir).tar.gz

test: FORCE all
	make
	./DIST_TEST
	./draw vor.out.nc 0

dist: $(distdir).tar.gz distcheck

$(distdir).tar.gz: FORCE $(distdir)
	tar chof - $(distdir) | gzip -9 -c > $@
	rm -rf $(distdir)

$(distdir):
	mkdir -p $(distdir)/driver
	mkdir -p $(distdir)/lib
	mkdir -p $(distdir)/postprocessor
	cp DENSE_TEST $(distdir)
	cp POST_TEST $(distdir)
	cp VOR_TEST $(distdir)
	cp DIST_TEST $(distdir)
	cp makefile $(distdir)
	cp user_defs.mk $(distdir)
	cp driver/*.cpp $(distdir)/driver
	cp driver/makefile $(distdir)/driver
	cp postprocessor/*.cpp $(distdir)/postprocessor
	cp postprocessor/makefile $(distdir)/postprocessor
	cp lib/*.c $(distdir)/lib
	cp lib/*.cpp $(distdir)/lib
	cp lib/*.h $(distdir)/lib
	cp lib/*.hpp $(distdir)/lib
	cp lib/makefile $(distdir)/lib

distcheck: $(distdir).tar.gz
	gzip -cd $(distdir).tar.gz | tar xvf -
	cd $(distdir) && $(MAKE) test
	rm -rf $(distdir)
	echo "*** Package $(distdir).tar.gz is ready for distribution"

