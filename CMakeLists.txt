project                     (tess)
cmake_minimum_required      (VERSION 2.8)

option                      (optimize       "Build tess with optimization"		    ON)
option                      (debug          "Build tess with debugging on"		    OFF)
option                      (timing	    "Build tess with timing"			    ON)
option			    (use_cgal	    "Use CGAL"					    OFF)

if			    (timing)
    add_definitions	    (-DTIMING)
endif			    (timing)

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
find_package		    (GLUT)
find_package		    (OpenGL)

# DIY
find_path                   (DIY_INCLUDE_DIRS		diy.h)
find_library                (DIY_LIBRARY NAMES		diy)

# PNetCDF
find_path                   (PNetCDF_INCLUDE_DIR	pnetcdf.h)
find_library                (PNetCDF_LIBRARY NAMES	pnetcdf)

set                         (CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem")
include_directories         (${CMAKE_CURRENT_SOURCE_DIR}/lib
			     ${DIY_INCLUDE_DIRS}
			     ${PNetCDF_INCLUDE_DIR}
                             SYSTEM ${MPI_INCLUDE_PATH}
                            )
		
set			    (libraries
			     ${libraries}
			     ${DIY_LIBRARY}
			     ${PNetCDF_LIBRARY}
			     ${MPI_C_LIBRARIES}
			     ${MPI_CXX_LIBRARIES}
			     ${CMAKE_THREAD_LIBS_INIT}
			    )

add_subdirectory	    (lib)
add_subdirectory	    (driver)
add_subdirectory	    (postprocessor)
