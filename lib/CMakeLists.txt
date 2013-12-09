add_library		(tess tess.c io.c)

include_directories	(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR})
add_executable		(draw draw.cpp ser_io.cpp swap.cpp io.c)
target_link_libraries	(draw ${libraries} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

add_executable		(dump dump.cpp ser_io.cpp swap.cpp)

add_executable		(dense dense.cpp io.c)
target_link_libraries	(dense ${libraries})
