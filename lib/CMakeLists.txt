if			(use_cgal)
    add_library		(tess tess.c tess-cgal.cpp io.c)
else			(use_cgal)
    add_library		(tess tess.c tess-qhull.c io.c)
endif			(use_cgal)

include_directories	(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR})
add_executable		(draw draw.cpp ser_io.cpp swap.cpp io.c)
target_link_libraries	(draw ${libraries} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

add_executable		(dump dump.cpp ser_io.cpp swap.cpp)
target_link_libraries	(dump ${libraries})

add_executable		(dense dense.cpp io.c)
target_link_libraries	(dense ${libraries})
