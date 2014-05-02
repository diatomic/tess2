if			(use_cgal)
    add_library		(tess tess.cpp tess-cgal.cpp io.c swap.cpp
    tet.cpp utils.c dense.cpp)
else			(use_cgal)
    add_library		(tess tess.cpp tess-qhull.c io.c swap.cpp
    tet.cpp utils.c dense.cpp)
endif			(use_cgal)

if			(draw)
include_directories	(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR})
add_executable		(draw draw.cpp ser_io.cpp swap.cpp io.c tet.cpp)
target_link_libraries	(draw ${libraries} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
endif			(draw)

add_executable		(dump dump.cpp ser_io.cpp swap.cpp)
target_link_libraries	(dump ${libraries})

