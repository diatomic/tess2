#define qh_QHimport
#include "qhull_a.h"
#include "tet.h"
#include "delaunay.h"

int gen_voronoi_output(facetT *facetlist, struct vblock_t *vblock,
		       int num_particles);
int gen_delaunay_output(facetT *facetlist, int **tet_verts);
void gen_d_delaunay_output(facetT *facetlist, struct dblock_t *dblock);
int bin_search(int *tbl, int key, int size);

/* copied from tet-neighbors.h because that header has other functions using
   vectors that are not compatible with C*/
int complete(int v, struct tet_t* tets, int t);

