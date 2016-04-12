option                      (gadget             "Gadget IO support"      on)
option                      (hacc               "HACC IO support"        off)
option                      (moab               "MOAB IO support"        off)

# gadget
if                          (gadget)
  find_package              (Boost COMPONENTS filesystem)
  include_directories       (SYSTEM ${Boost_INCLUDE_DIR})
  set                       (io_sources ${io_sources} io/gadget/particles.cpp)
  add_definitions           (-DTESS_GADGET_IO)
endif                       (gadget)

# HDF5 is always required
if                          (HDF5_LIBRARY)
  message                   ("HDF5_LIBRARY = " ${HDF5_LIBRARY})
else                        ()
  message                   ("Looking for HDF5")
  find_package              (HDF5 QUIET)
endif                       ()

if                          (HDF5_FOUND OR ${HDF5_LIBRARY})
  include_directories       (SYSTEM ${HDF5_INCLUDE_DIR})
  add_library               (hdf5 STATIC IMPORTED)
  set_target_properties     (hdf5 PROPERTIES IMPORTED_LOCATION ${HDF5_hdf5_LIBRARY})
  set                       (HDF5_libraries     hdf5)
else                        ()
  find_path                 (HDF5_INCLUDE_DIR   hdf5.h)
  find_library              (HDF5_LIBRARY NAMES hdf5)
  include_directories       (SYSTEM ${HDF5_INCLUDE_DIR})
  message                   ("HDF5_INCLUDE_DIR=${HDF5_INCLUDE_DIR} HDF5_LIBRARY=${HDF5_LIBRARY}")
  if                        (HDF5_LIBRARY)
    if                      (ZLIB_LIBRARIES)
      message               ("ZLIB_LIBRARIES = " ${ZLIB_LIBRARIES})
    else                    ()
      message               ("Looking for Zlib")
      find_package          (ZLIB REQUIRED)
    endif                   ()
    include_directories     (SYSTEM ${ZLIB_INCLUDE_DIRS})
    set                     (HDF5_libraries ${HDF5_LIBRARY} ${ZLIB_LIBRARIES})
  endif                     ()
endif                       ()

# pread-voronoi
set                         (io_sources ${io_sources} io/hdf5/pread-hdf5.cpp)

add_executable              (pread-voronoi pread-voronoi.cpp ${io_sources})
target_link_libraries       (pread-voronoi tess ${libraries} ${HDF5_libraries})
if                          (gadget)
  target_link_libraries     (pread-voronoi ${Boost_FILESYSTEM_LIBRARY})
endif                       (gadget)

# # balance
# add_executable              (balance balance.cpp ${io_sources})
# set_target_properties       (balance PROPERTIES COMPILE_FLAGS "-std=c++11")
# target_link_libraries       (balance tess ${libraries} ${HDF5_libraries})
# if                          (gadget)
#   target_link_libraries     (balance ${Boost_FILESYSTEM_LIBRARY})
# endif                       (gadget)

install                 (TARGETS pread-voronoi
  DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/pread-voronoi/
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
  GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)

install                 (FILES PREAD_VORONOI unit-cube.h5
  DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/pread-voronoi/
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
  GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)

# bov
add_executable        (pread-bov pread-bov.cpp)
target_link_libraries (pread-bov tess ${libraries} ${HDF5_libraries})

install               (TARGETS pread-bov
  DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/pread-voronoi/
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
  GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)

# moab
if                          (moab)
  find_path                 (MOAB_INCLUDE_DIR   moab_mpi.h)
  find_library              (MOAB_LIBRARY NAMES MOAB)
  include_directories       (SYSTEM ${MOAB_INCLUDE_DIR})

  # zlib is required for HDF5
  find_package              (ZLIB REQUIRED)
  include_directories       (SYSTEM ${ZLIB_INCLUDE_DIRS})
  set                       (HDF5_libraries ${HDF5_LIBRARY} ${ZLIB_LIBRARIES})

  # finally, moab
  set                       (io_sources io/moab/particles.cpp)
  add_definitions           (-DTESS_MOAB_IO)
  add_executable            (moab-test pread-moab.cpp ${io_sources})
  target_link_libraries     (moab-test tess ${libraries} ${MOAB_LIBRARY} ${HDF5_libraries})

  install                   (TARGETS moab-test
                            DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/pread-voronoi/
                            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
                            GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)

  install                   (FILES MOAB_TEST
                            DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/pread-voronoi/
                            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
                            GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)
endif                       (moab)

# hacc
if                          (hacc)

  find_path                 (GenericIO_INCLUDE_DIR      GenericIO.h)
  find_library              (GenericIO_LIBRARY NAMES    GenericIO)
  include_directories       (SYSTEM ${GenericIO_INCLUDE_DIR})
  set                       (io_sources io/hacc/particles.cpp)

  add_executable            (hacc-test hacc-voronoi.cpp ${io_sources})
  target_link_libraries     (hacc-test tess ${libraries} ${GenericIO_LIBRARY})

  install                   (TARGETS hacc-test
                            DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/pread-voronoi/
                            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
                            GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)

  install                   (FILES HACC_TEST
                            DESTINATION ${CMAKE_INSTALL_PREFIX}/examples/pread-voronoi/
                            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE
                            GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)

endif                       (hacc)
