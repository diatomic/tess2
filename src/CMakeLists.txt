# Buld tess library
if			    (${serial} MATCHES "CGAL")
    add_library		(tess tess.cpp tess-cgal.cpp io.cpp swap.cpp
    tet.cpp utils.c dense.cpp volume.cpp)
elseif		    (${serial} MATCHES "QHull")
    add_library		(tess tess.cpp tess-qhull.c io.cpp swap.cpp
    tet.cpp utils.c dense.cpp volume.cpp)
else			    ()
    message		    ("Uknown serial library: ${serial}")
endif			    ()
