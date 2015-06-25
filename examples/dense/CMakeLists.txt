add_executable          (dense main.cpp)
target_link_libraries   (dense tess ${libraries})

install                 (TARGETS dense
                        DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/dense/
                        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
                        GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)

install                 (FILES DENSE_TEST
                        DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/dense/
                        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
                        GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)
