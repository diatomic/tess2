# Buld tess library

set			(TESS_SOURCES tess.cpp swap.cpp# io.cpp
				      tet.cpp utils.c dense.cpp volume.cpp)

if			(${serial} MATCHES "CGAL")
  add_library		(tess SHARED ${TESS_SOURCES} tess-cgal.cpp)
elseif			(${serial} MATCHES "QHull")
  add_library		(tess SHARED ${TESS_SOURCES} tess-qhull.c)
else			()
    message		("Uknown serial library: ${serial}")
endif			()
