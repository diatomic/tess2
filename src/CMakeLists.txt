# Buld tess library

set			(TESS_SOURCES tess.cpp swap.cpp# io.cpp
				      tet.cpp utils.c dense.cpp volume.cpp)

if			(${serial} MATCHES "CGAL")
  add_library		(tess SHARED ${TESS_SOURCES} tess-cgal.cpp)
  target_link_libraries	(tess ${libraries})
elseif			(${serial} MATCHES "QHull")
  add_library		(tess SHARED ${TESS_SOURCES} tess-qhull.c)
  target_link_libraries	(tess ${libraries})
else			()
    message		("Uknown serial library: ${serial}")
endif			()
