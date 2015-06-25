add_executable          (post-delaunay main.cpp)
target_link_libraries   (post-delaunay tess ${libraries})

install                 (TARGETS post-delaunay
                        DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/post-tess/
                        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
                        GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)

install                (FILES POST_TEST
                       DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/post-tess/
                       PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
                       GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)
