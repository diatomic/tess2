# Buld tess library

set			(TESS_SOURCES tess.cpp swap.cpp tet.cpp dense.cpp volume.cpp)

if			(${serial} MATCHES "CGAL")
  add_library		(tess SHARED ${TESS_SOURCES} tess-cgal.cpp)
  target_link_libraries	(tess ${libraries})
elseif			(${serial} MATCHES "QHull")
  add_library		(tess SHARED ${TESS_SOURCES} tess-qhull.c)
  target_link_libraries	(tess ${libraries})
else			()
    message		("Uknown serial library: ${serial}")
endif			()

install                 (TARGETS tess
                        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
                        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
                        GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)
