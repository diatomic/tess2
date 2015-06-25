add_executable          (delaunay main.cpp)
target_link_libraries   (delaunay tess ${libraries})

install                 (TARGETS delaunay
                        DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/tess/
                        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
                        GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)

install                 (FILES TESS_TEST
                        DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/tess/
                        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
                        GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)
