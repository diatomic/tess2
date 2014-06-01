find_package                (HDF5 QUIET COMPONENTS CXX)

if                          (HDF5_FOUND)

  include_directories         (SYSTEM ${HDF5_CXX_INCLUDE_DIR})

  add_library		    (hdf5 STATIC IMPORTED)
  set_target_properties	    (hdf5 PROPERTIES IMPORTED_LOCATION ${HDF5_hdf5_LIBRARY})
  add_library		    (hdf5_cpp STATIC IMPORTED)
  set_target_properties	    (hdf5_cpp PROPERTIES IMPORTED_LOCATION ${HDF5_hdf5_cpp_LIBRARY})
  set			    (HDF5_libraries     hdf5 hdf5_cpp)


  add_executable		    (pread-voronoi pread-voronoi.cpp pread-hdf5.cpp)
  target_link_libraries	    (pread-voronoi tess ${libraries} ${HDF5_libraries})
else			    (HDF5_FOUND)
  message("HDF5 not found, pread-voronoi won't be built.")
endif			    (HDF5_FOUND)
