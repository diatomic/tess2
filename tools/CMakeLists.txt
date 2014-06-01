set			(source_dir ../src)
# TODO: why is ser_io.cpp not part of tess library? If it were, can't we just
#	link against tess here and be done with specifying explicit paths?

if			(draw)
include_directories	(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR})
add_executable		(draw draw.cpp ${source_dir}/ser_io.cpp ${source_dir}/swap.cpp ${source_dir}/io.c ${source_dir}/tet.cpp)
target_link_libraries	(draw ${libraries} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
endif			(draw)

add_executable		(dump dump.cpp ${source_dir}/ser_io.cpp ${source_dir}/swap.cpp)
target_link_libraries	(dump ${libraries})
