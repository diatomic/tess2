#define qh_QHimport
#include "qhull_a.h"

int gen_voronoi_output(facetT *facetlist, struct vblock_t *vblock,
		       int num_particles);
int gen_delaunay_output(facetT *facetlist, struct vblock_t *vblock,
			int *gids, int *nids, unsigned char *dirs,
			struct remote_ic_t *rics, int lid, int num_recvd);
