#include "tess/tess.h"
#include "tess/tess-qhull.h"
#include <assert.h>

/*--------------------------------------------------------------------------*/
/* Initialization and destruction of Delaunay data structures is not used with
 * qhull, since it doesn't support incremental updates.
 */
void init_delaunay_data_structure(struct dblock_t* b)
{
  b->Dt = NULL;
}

void clean_delaunay_data_structure(struct dblock_t* b)
{
  b->Dt = NULL;
}
/*--------------------------------------------------------------------------*/
/*
  creates local delaunay cells

  dblock: local block
*/
void local_cells(struct dblock_t *dblock) 
{
  boolT ismalloc = False;    /* True if qhull should free points in
				qh_freeqhull() or reallocation */
  char flags[250];          /* option flags for qhull, see qh-quick.htm */
  int exitcode;             /* 0 if no error from qhull */
  int curlong, totlong;     /* memory remaining after qh_memfreeshort */
  FILE *dev_null; /* file descriptor for writing to /dev/null */
  int i, j;
  int dim = 3; /* 3d */

  dev_null = fopen("/dev/null", "w");
  assert(dev_null != NULL);

  /* deep copy from float to double (qhull API is double) */
  double *pts = 
    (double *)malloc(dblock->num_particles * 3 * sizeof(double));
  for (j = 0; j < 3 * dblock->num_particles; j++)
    pts[j] = dblock->particles[j];

  /* compute delaunay */
  /*     sprintf(flags, "qhull v d o Fv Fo"); /\* Fv prints voronoi faces *\/ */
  sprintf (flags, "qhull d Qt"); /* print delaunay cells */

  /* eat qhull output by sending it to dev/null */
  exitcode = qh_new_qhull(dim, dblock->num_particles, pts, ismalloc,
                          flags, dev_null, stderr);

  free(pts);

  /* process delaunay output */
  if (!exitcode)
    gen_delaunay_output(qh facet_list, dblock);

  /* qhull does not order verts and neighbor tets such that the ith
     neighbor is opposite the ith vertex; so need to reorder neighbors */
  reorder_neighbors(dblock);

  fill_vert_to_tet(dblock);

  /* mem check */
#ifdef MEM
  int dwell = 10;
  get_mem(-1, dwell);
#endif

  /* clean up qhull */
  qh_freeqhull(!qh_ALL);                 /* free long memory */
  qh_memfreeshort(&curlong, &totlong);  /* free short memory */
  if (curlong || totlong)
    fprintf (stderr, "qhull internal warning: did not free %d bytes of "
             "long memory (%d pieces)\n", totlong, curlong);

  fclose(dev_null);

}
/*--------------------------------------------------------------------------*/
/*
  generates delaunay output from qhull

  facetlist: qhull list of convex hull facets
  tet_verts: pointer to array of tet vertex indices for this block 
  (allocated by this function, user's responsibility to free)

*/
void gen_delaunay_output(facetT *facetlist, struct dblock_t *dblock) {

  facetT *facet, *neighbor, **neighborp;
  vertexT *vertex, **vertexp;
  int numfacets = 0;
  int t, v, n; /* index in tets, tet verts, tet neighbors */

  /* count number of tets (facets to qhull) */
  FORALLfacet_(facetlist) {
    if ((facet->visible && qh NEWfacets) || (qh_skipfacet(facet)))
      facet->visitid= 0;
    else
      facet->visitid= ++numfacets;
  }

  dblock->num_tets = numfacets;
  dblock->tets = (struct tet_t *)malloc(numfacets * sizeof(struct tet_t));

  /* for all tets, get vertices */
  t = 0;
  FORALLfacet_(facetlist) {

    if (qh_skipfacet(facet) || (facet->visible && qh NEWfacets))
      continue;

    if (qh_setsize(facet->vertices) != 4) {
      fprintf(stderr, "tet has %d vertices; skipping.\n",
	      qh_setsize(facet->vertices));
      continue;
    }

    /* for all vertices */
    v = 0;
    if ((facet->toporient ^ qh_ORIENTclock)
	|| (qh hull_dim > 2 && !facet->simplicial)) {
      FOREACHvertex_(facet->vertices)
	dblock->tets[t].verts[v++] = qh_pointid(vertex->point);
    } else {
      FOREACHvertexreverse12_(facet->vertices)
	dblock->tets[t].verts[v++] = qh_pointid(vertex->point);
    }
    ++t;
  }

  /* for all tets, get neighbors */
  t = 0;
  FORALLfacet_(facetlist) {

    if (qh_skipfacet(facet) || (facet->visible && qh NEWfacets))
      continue;

    if (qh_setsize(facet->vertices) != 4) {
      fprintf(stderr, "tet has %d vertices; skipping.\n",
	      qh_setsize(facet->vertices));
      continue;
    }

    /* for all neighbor tets */
    n = 0;
    FOREACHneighbor_(facet) {
      if (neighbor->visitid)
	dblock->tets[t].tets[n++] = neighbor->visitid - 1;
      else
	dblock->tets[t].tets[n++] = -1;
    }
    assert(n == 4); /* sanity */

    t++;

  } /* for all tets */

  assert(numfacets == t); /* sanity */

}
/*--------------------------------------------------------------------------*/
/*
reorders neighbors in dblock such that ith neighbor is opposite ith vertex
*/
void reorder_neighbors(struct dblock_t *dblock) {

  int t, v, n, nv; /*indices into tets, verts, neighbors, neighbor verts */
  int nbr; /* one neighbor tet */
  int done; /* this neighbor is done */
  int tets[4];  /* newly ordered neighbors */

  /* tets */
  for (t = 0; t < dblock->num_tets; t++) {

    /* verts */
    for (v = 0; v < 4; v++) {

      done = 0;

      /* neighbor tets */
      for (n = 0; n < 4; n++) {

	nbr = dblock->tets[t].tets[n];
	if (nbr == -2) /* done already */
	  continue;

	/* neighbor tet verts */
	if (nbr > -1) {
	  for (nv = 0; nv < 4; nv++) {
	    if (dblock->tets[nbr].verts[nv] == dblock->tets[t].verts[v])
	      break; /* nbr is the wrong neighbor */
	  }
	}

	if (nbr > -1 && nv == 4) {/* nbr is the right neighbor */
	  done = 1;
	  break;
	}

      }

      if (done) {
	tets[v] = nbr;
	dblock->tets[t].tets[n] = -2; /* mark this neighbor as done */
      }
      else
	tets[v] = -1;

    } /* verts */

    /* copy reordered neighbors back to dblock */
    for (n = 0; n < 4; n++)
      dblock->tets[t].tets[n] = tets[n];

    /* sanity check */
    for (v = 0; v < 4; ++v) {
      int nbr = dblock->tets[t].tets[v]; /* opposite neighbor */
      if (nbr > -1) {
	for (nv = 0; nv < 4; nv++) { /* verts in opposite neighbor */
	  if (dblock->tets[nbr].verts[nv] == dblock->tets[t].verts[v])
	    fprintf(stderr, "Neighbor of tet %d can't have a vertex opposite "
		    "to it\n", t);
	}
      }
    }

  } /* tets */

}
/*--------------------------------------------------------------------------*/
