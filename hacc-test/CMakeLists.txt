if			(hacc_test)
add_executable          (hacc-voronoi main.cpp ${DEBUG_SOURCES})
target_link_libraries   (hacc-voronoi tess ${libraries})
endif			(hacc_test)
