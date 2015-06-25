add_executable          (tess-dense main.cpp)
target_link_libraries   (tess-dense tess ${libraries})

install                 (TARGETS tess-dense
                        DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/tess-dense/
                        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
                        GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)

install                 (FILES TESS_DENSE_TEST
                        DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/tess-dense/
                        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
                        GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)
