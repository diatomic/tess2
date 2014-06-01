if			(hacc_test)
add_executable          (hacc-delaunay main.cpp ${DEBUG_SOURCES})
target_link_libraries   (hacc-delaunay tess ${libraries})
endif			(hacc_test)
