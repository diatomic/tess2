if			(draw)
include_directories	(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR})
add_executable		(draw draw.cpp)
target_link_libraries	(draw tess ${libraries} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
endif			(draw)

