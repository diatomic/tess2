find_package                (HDF5 QUIET COMPONENTS CXX)

if                          (HDF5_FOUND)

  include_directories       (SYSTEM ${HDF5_CXX_INCLUDE_DIR})

  add_library		    (hdf5 STATIC IMPORTED)
  set_target_properties	    (hdf5 PROPERTIES IMPORTED_LOCATION ${HDF5_hdf5_LIBRARY})
  add_library		    (hdf5_cpp STATIC IMPORTED)
  set_target_properties	    (hdf5_cpp PROPERTIES IMPORTED_LOCATION ${HDF5_hdf5_cpp_LIBRARY})
  set			    (HDF5_libraries     hdf5 hdf5_cpp)

  add_executable            (pread-voronoi pread-voronoi.cpp pread-hdf5.cpp)
  target_link_libraries	    (pread-voronoi tess ${libraries} ${HDF5_libraries})

else			    (HDF5_FOUND)

  find_path                 (HDF5_CXX_INCLUDE_DIR   H5Cpp.h)
  find_library              (HDF5_LIBRARY NAMES     hdf5)
  find_library              (HDF5_CXX_LIBRARY NAMES hdf5_cpp)
  include_directories       (SYSTEM ${HDF5_CXX_INCLUDE_DIR})

  if                        (HDF5_CXX_LIBRARY)
    find_package            (ZLIB REQUIRED)
    include_directories     (SYSTEM ${ZLIB_INCLUDE_DIRS})
    set			    (HDF5_libraries ${HDF5_LIBRARY} ${HDF5_CXX_LIBRARY} ${ZLIB_LIBRARIES})

    add_executable          (pread-voronoi pread-voronoi.cpp pread-hdf5.cpp)
    target_link_libraries   (pread-voronoi tess ${libraries} ${HDF5_libraries})
  endif                     (HDF5_CXX_LIBRARY)

endif			    (HDF5_FOUND)
