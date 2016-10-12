project                     (tess)
cmake_minimum_required      (VERSION 2.8)

option			    (BUILD_SHARED_LIBS "Build shared libraries"                        OFF)
option                      (optimize          "Build tess with optimization"		       ON)
option                      (debug             "Build tess with debugging on"		       OFF)
option                      (timing	       "Build tess with timing"			       ON)
option                      (memory	       "Build tess with memory profiling"	       OFF)
option			    (draw	       "Build draw"				       ON)
option			    (bgq	       "Build on BG/Q"				       OFF)
option			    (pread             "Build pread-voronoi example (requires HDF5)"   OFF)
option                      (diy_thread        "Enable diy threading"                          OFF)
option                      (build_examples    "Build examples"                                ON)
option                      (build_tools       "Build tools"                                   ON)

set			    (serial	       "QHull" CACHE STRING "serial Delaunay library to use")
set_property		    (CACHE serial PROPERTY STRINGS CGAL QHull)

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
    set			    (CMAKE_MACOSX_RPATH	    on)
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

# QHull vs. CGAL
if			    (${serial} MATCHES "CGAL")
  message		    ("Using CGAL")
  find_package              (CGAL)
  find_package              (Boost)
  include                   (${CGAL_USE_FILE})
  add_definitions           (${CGAL_CXX_FLAGS_INIT})
  include_directories       (${CGAL_INCLUDE_DIRS} SYSTEM ${Boost_INCLUDE_DIR})
  set                       (libraries ${libraries} ${CGAL_LIBRARY} ${CGAL_3RD_PARTY_LIBRARIES})
  set                       (CMAKE_EXE_LINKER_FLAGS "-dynamic ${CMAKE_EXE_LINKER_FLAGS}")
  # TODO: this should really be based on whether the compiler suffers from the bug in std::nth_element()
  add_definitions           (-DTESS_CGAL_ALLOW_SPATIAL_SORT)
  add_definitions           (-DTESS_USE_CGAL)
elseif                      (${serial} MATCHES "QHull")
  message		    ("Using QHull")
  find_path                 (QHull_INCLUDE_DIRS		libqhull.h)
  find_library              (QHull_LIBRARY NAMES	qhullstatic)
  include_directories	    (${QHull_INCLUDE_DIRS})
  set			    (libraries ${libraries} ${QHull_LIBRARY})
  add_definitions	    (-DTESS_USE_QHull)
else			    ()
  message		    ("Uknown serial library: ${serial}")
endif			    ()

# MPI
find_package                (MPI REQUIRED)
set                         (libraries ${libraries}    ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})

# Threads
find_package		    (Threads)
find_package		    (OpenMP)
if 			    (OPENMP_FOUND)
  set                       (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set                       (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else			    ()
  message		    ("OpenMP not found")
  add_definitions	    (-DTESS_NO_OPENMP)
endif                       ()
if                          (NOT diy_thread)
    message                 ("Diy threading is disabled; setting diy threads will have no effect")
    add_definitions         (-DDIY_NO_THREADS)
endif                       (NOT diy_thread)

# OpenGL
if			    (draw)
find_package		    (GLUT)
find_package		    (OpenGL)
endif			    (draw)

# DIY
find_path                   (DIY_INCLUDE_DIRS		diy/types.h)

# Include path
set                         (CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem")
include_directories         (${CMAKE_CURRENT_SOURCE_DIR}/include
			     ${DIY_INCLUDE_DIRS}
                             SYSTEM ${MPI_INCLUDE_PATH}
                            )

# Libraries
set			    (libraries
			     ${libraries}
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

# Subdirectories
add_subdirectory	    (src)
if                          (build_examples)
add_subdirectory	    (examples)
endif                       ()
if                          (build_tools)
add_subdirectory	    (tools)
endif                       ()

# Install the headers
file                        (GLOB DEPLOY_FILES_AND_DIRS "${PROJECT_SOURCE_DIR}/include/*")
foreach                     (ITEM ${DEPLOY_FILES_AND_DIRS})
   if                       (IS_DIRECTORY "${ITEM}" )
      list                  (APPEND DIRS_TO_DEPLOY "${ITEM}")
   else                     ()
      list                  (APPEND FILES_TO_DEPLOY "${ITEM}")
   endif                    ()
endforeach                  ()
install                     (FILES ${FILES_TO_DEPLOY} DESTINATION ${CMAKE_INSTALL_PREFIX}/include )
install                     (DIRECTORY ${DIRS_TO_DEPLOY} DESTINATION ${CMAKE_INSTALL_PREFIX}/include )
