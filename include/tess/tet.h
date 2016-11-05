#ifndef _TESS_TET_H
#define _TESS_TET_H

struct tet_t {
    int verts[4];		/* indices of the vertices */
    int tets[4];		/* indices of the neighbors; tets[i] lies opposite verts[i] */
};

#endif
