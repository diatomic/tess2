if			(hacc_test)
add_executable          (hacc-delaunay main.cpp ${DEBUG_SOURCES})
target_link_libraries   (hacc-delaunay tess ${libraries})
endif			(hacc_test)

install                 (TARGETS hacc-delaunay
                        DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/hacc-post-tess/
                        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
                        GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)

install                 (FILES HACC_TEST
                        DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/hacc-post-tess/
                        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
                        GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)
