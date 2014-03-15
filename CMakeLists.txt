project                     (tess)
cmake_minimum_required      (VERSION 2.8)

option                      (optimize       "Build tess with optimization"		    ON)
option                      (debug          "Build tess with debugging on"		    OFF)
option                      (timing	    "Build tess with timing"			    ON)
option                      (memory	    "Build tess with memory profiling"		    OFF)
option			    (use_cgal	    "Use CGAL"					    OFF)
option			    (draw	    "Build draw"				    ON)
option			    (hacc_test	    "Build HACC test"				    OFF)
option			    (bgq	    "Build on BG/Q"				    OFF)

if			    (timing)
    add_definitions	    (-DTIMING)
endif			    (timing)

if			    (memory)
  add_definitions	    (-DMEMORY)
endif			    (memory)

if			    (bgq)
  add_definitions	    (-DBGQ)
endif			    (bgq)

# Debugging
if                          (debug)
    if                      (optimize)
            set             (cxx_flags                  ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
    else                    (optimize)
            set             (cxx_flags                  ${CMAKE_CXX_FLAGS_DEBUG})
    endif                   (optimize)
    add_definitions         (-DDEBUG)
else                        (debug)
    if                      (optimize)
            set             (cxx_flags                  ${CMAKE_CXX_FLAGS_RELEASE})
    else                    (optimize)
            set             (cxx_flags                  ${CMAKE_CXX_FLAGS})
    endif                   (optimize)
endif                       (debug)
add_definitions             (${cxx_flags})

# OSX flags
IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_definitions	    (-DMAC_OSX)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin") 

# QHull vs. CGAL
if			    (use_cgal)
    find_package            (CGAL)
    include                 (${CGAL_USE_FILE})
    add_definitions         (${CGAL_CXX_FLAGS_INIT})
    include_directories     (${CGAL_INCLUDE_DIRS})
    set			    (libraries ${libraries} ${CGAL_LIBRARY} ${CGAL_3RD_PARTY_LIBRARIES})
    set 		    (CMAKE_EXE_LINKER_FLAGS "-dynamic ${CMAKE_EXE_LINKER_FLAGS}")
    # TODO: this should really be based on whether the compiler suffers from the bug in std::nth_element()
    add_definitions	    (-DTESS_CGAL_ALLOW_SPATIAL_SORT)
else			    (use_cgal)
    find_path               (QHull_INCLUDE_DIRS		libqhull.h)
    find_library            (QHull_LIBRARY NAMES	qhullstatic)
    include_directories	    (${QHull_INCLUDE_DIRS})
    set			    (libraries ${libraries} ${QHull_LIBRARY})
endif			    (use_cgal)

# MPI
find_package                (MPI REQUIRED)

# Threads
find_package		    (Threads)

# OpenGL
if			    (draw)
find_package		    (GLUT)
find_package		    (OpenGL)
endif			    (draw)

# DIY
find_path                   (DIY_INCLUDE_DIRS		diy.h)
find_library                (DIY_LIBRARY NAMES		diy)

# PNetCDF
find_path                   (PNetCDF_INCLUDE_DIR	pnetcdf.h)
find_library                (PNetCDF_LIBRARY NAMES	pnetcdf)
if			    (PNetCDF_LIBRARY)
  add_definitions	    (-DPNETCDF_IO)
endif			    (PNetCDF_LIBRARY)

# GenericIO
if			    (hacc_test)
find_path                   (GenericIO_INCLUDE_DIR	GenericIO.h)
find_library                (GenericIO_LIBRARY NAMES	GenericIO)
endif			    (hacc_test)

set                         (CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem")
include_directories         (${CMAKE_CURRENT_SOURCE_DIR}/lib
			     ${DIY_INCLUDE_DIRS}
			     ${PNetCDF_INCLUDE_DIR}
			     ${GenericIO_INCLUDE_DIR}
                             SYSTEM ${MPI_INCLUDE_PATH}
                            )
		
set			    (libraries
			     ${libraries}
			     ${DIY_LIBRARY}
			     ${PNetCDF_LIBRARY}
			     ${GenericIO_LIBRARY}
			     ${MPI_C_LIBRARIES}
			     ${MPI_CXX_LIBRARIES}
			     ${CMAKE_THREAD_LIBS_INIT}
			    )

if                          (debug)
    find_library            (LIBDW_LIBRARY NAMES dw)
    if                      (LIBDW_LIBRARY)
        set                 (DEBUG_SOURCES ${CMAKE_SOURCE_DIR}/lib/backward.cpp)
        add_definitions     (-DBACKWARD_HAS_DW=1)
        set                 (libraries ${libraries} ${LIBDW_LIBRARY})
    else                    (LIBDW_LIBRARY)
        message             (STATUS "LibDW not found; backward.cpp won't be used")
    endif                   (LIBDW_LIBRARY)
endif                       (debug)

add_subdirectory	    (lib)
add_subdirectory	    (driver)
add_subdirectory	    (postprocessor)
add_subdirectory	    (pread-voronoi)
add_subdirectory	    (hacc-test)
