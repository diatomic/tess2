if			(draw)

  include_directories	(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR})
  add_executable	(draw draw.cpp)
  target_link_libraries	(draw tess ${libraries} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
  install               (TARGETS draw
                        DESTINATION ${CMAKE_INSTALL_PREFIX}/tools/
                        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
                        GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)

endif			(draw)

add_executable		(stats stats.cpp)
target_link_libraries	(stats tess ${libraries})

install                 (TARGETS stats
                        DESTINATION ${CMAKE_INSTALL_PREFIX}/tools/
                        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
                        GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)

install                 (FILES dense-plot.py
                        DESTINATION ${CMAKE_INSTALL_PREFIX}/tools/
                        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
                        GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)
