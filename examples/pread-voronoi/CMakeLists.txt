find_package                (HDF5 QUIET)

if                          (HDF5_FOUND)

  include_directories       (SYSTEM ${HDF5_INCLUDE_DIR})

  add_library		    (hdf5 STATIC IMPORTED)
  set_target_properties	    (hdf5 PROPERTIES IMPORTED_LOCATION ${HDF5_hdf5_LIBRARY})
  set			    (HDF5_libraries     hdf5)

  add_executable            (pread-voronoi pread-voronoi.cpp pread-hdf5.cpp)
  target_link_libraries	    (pread-voronoi tess ${libraries} ${HDF5_libraries})

else			    (HDF5_FOUND)

  find_path                 (HDF5_INCLUDE_DIR   hdf5.h)
  find_library              (HDF5_LIBRARY NAMES     hdf5)
  include_directories       (SYSTEM ${HDF5_INCLUDE_DIR})

  if                        (HDF5_LIBRARY)
    find_package            (ZLIB REQUIRED)
    include_directories     (SYSTEM ${ZLIB_INCLUDE_DIRS})
    set			    (HDF5_libraries ${HDF5_LIBRARY} ${ZLIB_LIBRARIES})

    add_executable          (pread-voronoi pread-voronoi.cpp pread-hdf5.cpp)
    target_link_libraries   (pread-voronoi tess ${libraries} ${HDF5_libraries})
  endif                     (HDF5_LIBRARY)

endif			    (HDF5_FOUND)
