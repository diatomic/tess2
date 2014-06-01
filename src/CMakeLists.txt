# Buld tess library
if			(use_cgal)
    add_library		(tess tess.cpp tess-cgal.cpp io.c swap.cpp
    tet.cpp utils.c dense.cpp)
else			(use_cgal)
    add_library		(tess tess.cpp tess-qhull.c io.c swap.cpp
    tet.cpp utils.c dense.cpp)
endif			(use_cgal)
