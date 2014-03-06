#include "tess.h"
#include "tess-qhull.h"

/*--------------------------------------------------------------------------*/
/* Initialization and destruction of Delaunay data structures is not used with
 * qhull, since it doesn't support incremental updates.
 */
void* init_delaunay_data_structures(int nblocks) {
  nblocks = nblocks; /* quiet compiler warning about unused parameter */
  return 0;
}

void clean_delaunay_data_structures(void* ds) {
  ds = ds; /* quiet compiler warning about unused parameter */
}

/*--------------------------------------------------------------------------*/
/*
  creates local voronoi cells

  nblocks: number of blocks
  tblocks: pointer to array of temporary vblocks
  dim: number of dimensions (eg. 3)
  num_particles: number of particles in each block
  particles: particles in each block, particles[block_num][particle]
  where each particle is 3 values, px, py, pz
  times: timing
  ds: the delaunay data structures; unused in qhull
*/
void local_cells(int nblocks, struct vblock_t *tblocks, int dim,
		 int *num_particles, float **particles, void* ds,
		 struct tet_t** tets, int* ntets) {

  boolT ismalloc = False;    /* True if qhull should free points in
				qh_freeqhull() or reallocation */
  char flags[250];          /* option flags for qhull, see qh-quick.htm */
  int exitcode;             /* 0 if no error from qhull */
  int curlong, totlong;     /* memory remaining after qh_memfreeshort */
  FILE *dev_null; /* file descriptor for writing to /dev/null */
  int i, j;

  ds = ds; /* quiet compiler warning about unused parameter */

  dev_null = fopen("/dev/null", "w");
  assert(dev_null != NULL);

  /* for all blocks */
  for (i = 0; i < nblocks; i++) {

    // This obviously needs to be fixed; for now it's a fakeout to allow for
    // free() upstream
    tets[i]  = NULL;
    ntets[i] = 0;

    /* fprintf(stderr, "Num particles (local): %d\n", num_particles[i]); */

    /* deep copy from float to double (qhull API is double) */
    double *pts = (double *)malloc(num_particles[i] * 3 * sizeof(double));
    for (j = 0; j < 3 * num_particles[i]; j++)
      pts[j] = particles[i][j];

    /* compute voronoi */
/*     sprintf(flags, "qhull v o Fo"); /\* print voronoi cells *\/ */
    sprintf (flags, "qhull d Qt"); /* print delaunay cells */

    /* eat qhull output by sending it to dev/null
       need to see how this behaves on BG/P, will get I/O forwarded but will
       stop there and not proceed to storage */
    exitcode = qh_new_qhull(dim, num_particles[i], pts, ismalloc,
			    flags, dev_null, stderr);

    free(pts);

    /* allocate number of verts for original particles */
    tblocks[i].num_cell_verts = (int *)malloc(sizeof(int) * num_particles[i]);
    memset(tblocks[i].num_cell_verts, 0, sizeof(int) * num_particles[i]);

    /* process voronoi output */
    if (!exitcode)
      gen_voronoi_output(qh facet_list, &tblocks[i], num_particles[i]);

    /* allocate cell sites for original particles */
    tblocks[i].num_orig_particles = num_particles[i];
    tblocks[i].sites =
      (float *)malloc(3 * sizeof(float) * tblocks[i].num_orig_particles);
    for (j = 0; j < tblocks[i].num_orig_particles; j++) {
      tblocks[i].sites[3 * j] = particles[i][3 * j];
      tblocks[i].sites[3 * j + 1] = particles[i][3 * j + 1];
      tblocks[i].sites[3 * j + 2] = particles[i][3 * j + 2];
    }

    /* clean up qhull */
    qh_freeqhull(!qh_ALL);                 /* free long memory */
    qh_memfreeshort(&curlong, &totlong);  /* free short memory */
    if (curlong || totlong)
      fprintf (stderr, "qhull internal warning: did not free %d bytes of "
	       "long memory (%d pieces)\n", totlong, curlong);

  } /* for all blocks */

  fclose(dev_null);

}
/*--------------------------------------------------------------------------*/
/*
  creates local delaunay cells

  nblocks: number of blocks
  dblocks: pointer to array of dblocks
  dim: number of dimensions (eg. 3)
  times: timing
  ds: the delaunay data structures; unused in qhull
*/
void local_dcells(int nblocks, struct dblock_t *dblocks, int dim, void* ds) {

  boolT ismalloc = False;    /* True if qhull should free points in
				qh_freeqhull() or reallocation */
  char flags[250];          /* option flags for qhull, see qh-quick.htm */
  int exitcode;             /* 0 if no error from qhull */
  int curlong, totlong;     /* memory remaining after qh_memfreeshort */
  FILE *dev_null; /* file descriptor for writing to /dev/null */
  int i, j;

  ds = ds; /* quiet compiler warning about unused parameter */

  dev_null = fopen("/dev/null", "w");
  assert(dev_null != NULL);

  /* for all blocks */
  for (i = 0; i < nblocks; i++) {

    /* deep copy from float to double (qhull API is double) */
    double *pts = 
      (double *)malloc(dblocks[i].num_particles * 3 * sizeof(double));
    for (j = 0; j < 3 * dblocks[i].num_particles; j++)
      pts[j] = dblocks[i].particles[j];

    /* compute delaunay */
/*     sprintf(flags, "qhull v d o Fv Fo"); /\* Fv prints voronoi faces *\/ */
    sprintf (flags, "qhull d Qt"); /* print delaunay cells */

    /* eat qhull output by sending it to dev/null */
    exitcode = qh_new_qhull(dim, dblocks[i].num_particles, pts, ismalloc,
			    flags, dev_null, stderr);

    free(pts);

    /* process delaunay output */
    if (!exitcode)
      gen_d_delaunay_output(qh facet_list, &dblocks[i]);
    fill_vert_to_tet(&dblocks[i]);

    /* clean up qhull */
    qh_freeqhull(!qh_ALL);                 /* free long memory */
    qh_memfreeshort(&curlong, &totlong);  /* free short memory */
    if (curlong || totlong)
      fprintf (stderr, "qhull internal warning: did not free %d bytes of "
	       "long memory (%d pieces)\n", totlong, curlong);

  } /* for all blocks */

  fclose(dev_null);

}
/*--------------------------------------------------------------------------*/
/*
  creates all final voronoi and delaunay cells

  nblocks: number of blocks
  vblocks: pointer to array of vblocks
  dim: number of dimensions (eg. 3)
  num_particles: number of particles in each block
  num_orig_particles: number of original particles in each block, before any
  neighbor exchange
  particles: particles in each block, particles[block_num][particle]
  where each particle is 3 values, px, py, pz
  gids: global block ids of owners of received particles in each of my blocks
  nids: native particle ids of received particles in each of my blocks
  dirs: wrapping directions of received particles in each of my blocks
  times: timing
  ds: the delaunay data structures; unused in qhull
*/
void all_cells(int nblocks, struct vblock_t *vblocks, int dim,
		int *num_particles, int *num_orig_particles, 
		float **particles, int **gids, int **nids, 
		unsigned char **dirs, double *times, void* ds,
		struct tet_t** tets, int* ntets) {

  boolT ismalloc = False;    /* True if qhull should free points in
				qh_freeqhull() or reallocation */
  char flags[250];          /* option flags for qhull, see qh-quick.htm */
  int exitcode;             /* 0 if no error from qhull */
  int curlong, totlong;     /* memory remaining after qh_memfreeshort */
  FILE *dev_null; /* file descriptor for writing to /dev/null */
  int num_recvd; /* number of received particles in current block */
  int i, j;

  /* quiet compiler warnings about unused parameters */
  times = times;
  ds = ds;

  dev_null = fopen("/dev/null", "w");
  assert(dev_null != NULL);

  /* is_complete status of received particles */
  struct remote_ic_t **rics =
    (struct remote_ic_t **)malloc(nblocks * 
				  sizeof(struct remote_ic_t *));
  /* delaunay vertices */
  int **tet_verts = (int **)malloc(nblocks * sizeof(int *));
  int *num_tets = (int *)malloc(nblocks * sizeof(int));
  for (i = 0; i < nblocks; i++) {
    tet_verts[i] =  NULL;
    num_tets[i] = 0;
  }

  /* for all blocks */
  for (i = 0; i < nblocks; i++) {
    
    // This obviously needs to be fixed; for now it's a fakeout to allow for
    // free() upstream
    tets[i]  = NULL;
    ntets[i] = 0;

    /* number of received particles */
    num_recvd = num_particles[i] - num_orig_particles[i];

    /* deep copy from float to double (qhull API is double) */
    double *pts = (double *)malloc(num_particles[i] * 3 * sizeof(double));
    for (j = 0; j < 3 * num_particles[i]; j++)
      pts[j] = particles[i][j];

    /* compute voronoi */
/*     sprintf(flags, "qhull v o Fv Fo"); /\* Fv prints voronoi faces *\/ */
/*     sprintf(flags, "qhull d Qt"); /\* print delaunay cells *\/ */
    sprintf(flags, "qhull v d Fv Qt"); /* Fv prints voronoi faces */

    /* eat qhull output by sending it to dev/null
       need to see how this behaves on BG/P, will get I/O forwarded but will 
       stop there and not proceed to storage */
    exitcode = qh_new_qhull(dim, num_particles[i], pts, ismalloc,
			    flags, dev_null, stderr);

    free(pts);

    /* allocate number of verts for original particles */
    vblocks[i].num_cell_verts = (int *)malloc(sizeof(int) * num_particles[i]);
    memset(vblocks[i].num_cell_verts, 0, sizeof(int) * num_particles[i]);

    /* process voronoi output */
    if (!exitcode)
      gen_voronoi_output(qh facet_list, &vblocks[i], num_particles[i]);

    /* allocate cell sites for original particles */
    vblocks[i].num_orig_particles = num_orig_particles[i];
    vblocks[i].sites =
      (float *)malloc(3 * sizeof(float) * vblocks[i].num_orig_particles);
    for (j = 0; j < vblocks[i].num_orig_particles; j++) {
      vblocks[i].sites[3 * j] = particles[i][3 * j];
      vblocks[i].sites[3 * j + 1] = particles[i][3 * j + 1];
      vblocks[i].sites[3 * j + 2] = particles[i][3 * j + 2];
    }

    /* allocate lookup table for cells completion status */
    vblocks[i].is_complete = 
      (unsigned char *)malloc(vblocks[i].num_orig_particles);

    /* determine complete cells */
    complete_cells(&vblocks[i], i);

    /* process delaunay output */
    if (!exitcode)
      num_tets[i] = gen_delaunay_output(qh facet_list, &tet_verts[i]);

    /* clean up qhull */
    qh_freeqhull(!qh_ALL);                 /* free long memory */
    qh_memfreeshort(&curlong, &totlong);  /* free short memory */
    if (curlong || totlong)
      fprintf (stderr, "qhull internal warning: did not free %d bytes of "
	       "long memory (%d pieces)\n", totlong, curlong);

    /* connectivity of faces in voronoi cells */
    cell_faces(&vblocks[i]);

  } /* for all blocks */

  /* exchange complete cell status for exchanged particles */
  neighbor_is_complete(nblocks, vblocks, rics);

  /* convert delaunay output to vblock for all blocks */
  for (i = 0; i < nblocks; i++)
    gen_tets(tet_verts[i], num_tets[i], &vblocks[i], gids[i], nids[i], 
	     dirs[i], rics[i], i, num_particles[i] - num_orig_particles[i]);

  /* cleanup */
  for (i = 0; i < nblocks; i++) {
    free(rics[i]);
    if (tet_verts[i])
      free(tet_verts[i]);
  }
  free(rics);
  free(tet_verts);
  free(num_tets);
  fclose(dev_null);

}
/*--------------------------------------------------------------------------*/
/*
  generates voronoi output from qhull

  facetlist: qhull list of convex hull facets
  vblock: pointer to one voronoi block, allocated by caller
  num_particles: number of particles used to generate the tessellation
  side effects: allocates data structures inside of vblock, caller's
  responsibility to free

  returns: number of cells found (<= original number of particles)
*/
int gen_voronoi_output(facetT *facetlist, struct vblock_t *vblock, 
		       int num_particles) {

  int i, numcenters, numvertices= 0, numneighbors, numinf, vertex_i, vertex_n;
  facetT *facet, *neighbor, **neighborp;
  setT *vertices;
  vertexT *vertex, **vertexp;
  boolT isLower;
  unsigned int numfacets= (unsigned int) qh num_facets;
  int *skip_cells = NULL; /* cells (input particles) skipped by qhull */
  int num_skip_cells = 0; /* number of skipped cells */
  int alloc_skip_cells = 0; /* allocated number of skipped cells */
  int chunk_size = 128; /* chunk size for allocating skip_cells */

  /* init, get counts */
  int cell = 0; /* index of cell being processed */
  vertices = qh_markvoronoi(facetlist, NULL, 0, &isLower, &numcenters);
  FOREACHvertex_i_(vertices) {
    if (vertex) {
      numvertices++;
      numneighbors = numinf = 0;
      FOREACHneighbor_(vertex) {
        if (neighbor->visitid == 0)
          numinf= 1;
        else if (neighbor->visitid < numfacets)
          numneighbors++;
      }
      if (numinf && !numneighbors) {
        SETelem_(vertices, vertex_i)= NULL;
        numvertices--;
	add_int(cell, &skip_cells, &num_skip_cells, &alloc_skip_cells,
		chunk_size);
      }
    }
    else
	add_int(cell, &skip_cells, &num_skip_cells, &alloc_skip_cells,
		chunk_size);
    cell++;
  }

  /* number of verts and cells may appear to be reversed, but this is
     qhull's nomenclature (makes sense for delaunay) and is actually correct
  */
  vblock->num_verts = numcenters;
  int temp_num_cells = numvertices;

  /* vertices */
  vblock->verts = (double *)malloc(sizeof(double) * 3 * vblock->num_verts);

  vblock->verts[0] = vblock->verts[1] = vblock->verts[2] = qh_INFINITE;
  i = 1; /* already did the infinity vertex, index 0 */
  FORALLfacet_(facetlist) {
    if (facet->visitid && facet->visitid < numfacets) {
      if (!facet->center)
	facet->center = qh_facetcenter(facet->vertices);
      vblock->verts[3 * i] = facet->center[0];
      vblock->verts[3 * i + 1] = facet->center[1];
      vblock->verts[3 * i + 2] = facet->center[2];
      i++;
    }
  }

  /* number of vertices in each cell; size is number of particles; 
     if a cell is skipped, the number of vertices will be 0 */
  cell = 0;
  FOREACHvertex_i_(vertices) {
    numneighbors = 0;
    numinf = 0;
    if (vertex) {
      FOREACHneighbor_(vertex) {
        if (neighbor->visitid == 0)
          numinf= 1;
        else if (neighbor->visitid < numfacets)
          numneighbors++;
      }
      if (numinf)
	numneighbors++;
      vblock->num_cell_verts[cell++] = numneighbors;
    }
    else
      cell++;
  }

  /* allocate the cell vertices */
  vblock->tot_num_cell_verts = 0;
  for (i = 0; i < num_particles; i++)
    vblock->tot_num_cell_verts += vblock->num_cell_verts[i];
  vblock->cells = (int *)malloc(sizeof(int) * vblock->tot_num_cell_verts);

  /* cell vertices */
  i = 0;
  FOREACHvertex_i_(vertices) {
    if (vertex) {
      numinf = 0;
      FOREACHneighbor_(vertex) {
	if (neighbor->visitid < numfacets) {
	  if (!numinf || neighbor->visitid > 0) {
	    vblock->cells[i++] = neighbor->visitid;
	    if (neighbor->visitid == 0)
	      numinf++;
	  }
	}
	else if (numinf && neighbor->visitid < numfacets)
	  vblock->cells[i++] = neighbor->visitid;
      }
    }
  }

  /* voronoi faces */
  int tot_faces = qh_printvdiagram2 (NULL, NULL, vertices, qh_RIDGEall, False);
  vblock->faces = (struct vface_t*)malloc(tot_faces * sizeof(struct vface_t));
  memset(vblock->faces, 0, tot_faces * sizeof(struct vface_t));
  int num_faces = 0;

  FORALLvertices
    vertex->seen= False;
  FOREACHvertex_i_(vertices) {
    if (vertex) {
      if (qh GOODvertex > 0 && qh_pointid(vertex->point)+1 != qh GOODvertex)
        continue;

      /* following is equivalent to calling
	 qh_eachvoronoi(stderr, qh_printvridge, vertex, !qh_ALL, qh_RIDGEall,
	 True); */

      vertexT *atvertex = vertex;
      boolT unbounded;
      int count;
      facetT *neighbor, **neighborp, *neighborA, **neighborAp;
      setT *centers;
      setT *tricenters= qh_settemp(qh TEMPsize);
      boolT firstinf;
      unsigned int numfacets= (unsigned int)qh num_facets;
      int totridges= 0;

      qh vertex_visit++;
      atvertex->seen= True;
      FOREACHneighbor_(atvertex) {
	if (neighbor->visitid < numfacets)
	  neighbor->seen= True;
      }
      FOREACHneighbor_(atvertex) {
	if (neighbor->seen) {

	  FOREACHvertex_(neighbor->vertices) {

	    if (vertex->visitid != qh vertex_visit && !vertex->seen) {
	      vertex->visitid= qh vertex_visit;
	      count= 0;
	      firstinf= True;
	      qh_settruncate(tricenters, 0);

	      FOREACHneighborA_(vertex) {
		if (neighborA->seen) {
		  if (neighborA->visitid) {
		    if (!neighborA->tricoplanar ||
			qh_setunique(&tricenters, neighborA->center))
		      count++;
		  }else if (firstinf) {
		    count++;
		    firstinf= False;
		  }
		}
	      } /* FOREACHneighborA */

	      if (count >= qh hull_dim - 1) {  /* e.g., 3 for 3-d Voronoi */
		if (firstinf)
		  unbounded= False;
		else
		  unbounded= True;
		totridges++;
		trace4((qh ferr, 4017,
			"qh_eachvoronoi: Voronoi ridge of %d vertices "
			"between sites %d and %d\n",
			count, qh_pointid(atvertex->point),
			qh_pointid(vertex->point)));

		if (qh hull_dim == 3+1) /* 3-d Voronoi diagram */
		  centers= qh_detvridge3 (atvertex, vertex);
		else
		  centers= qh_detvridge(vertex);

		/* following is equivvalent to calling
		   qh_printvridge(fp, atvertex, vertex, centers, unbounded) */

		facetT *facet, **facetp;
		QHULL_UNUSED(unbounded);
		vblock->faces[num_faces].num_verts = qh_setsize(centers);
		vblock->faces[num_faces].cells[0] = qh_pointid(atvertex->point);
		vblock->faces[num_faces].cells[1] = qh_pointid(vertex->point);
		int num_verts = 0;
		FOREACHfacet_(centers) {
		  assert(num_verts < MAX_FACE_VERTS);
		  vblock->faces[num_faces].verts[num_verts++] = facet->visitid;
		}
		num_faces++;

		/* end of equivalence of calling qh_printvridge */

		qh_settempfree(&centers);

	      } /* if count >= ... */

	    } /* if (vertex->visitid ... */

	  } /* FOREACHvertex */

	} /* if (neighbor_seen) */

      } /* FOREACHneighbor */

      FOREACHneighbor_(atvertex)
	neighbor->seen= False;
      qh_settempfree(&tricenters);

      /* end of equivalence of calling qh_eachvoronoi() */

    } /* if (vertex) */

  } /* FOREACHvertex */

  vblock->num_faces = num_faces;
  assert(vblock->num_faces == tot_faces); /* sanity */

  /* clean up */
  qh_settempfree(&vertices);
  if (skip_cells)
    free(skip_cells);

  return temp_num_cells;

}
/*--------------------------------------------------------------------------*/
/*
  generates delaunay output from qhull

  facetlist: qhull list of convex hull facets
  tet_verts: pointer to array of tet vertex indices for this block 
  (allocated by this function, user's responsibility to free)

  returns: number of tets
*/
int gen_delaunay_output(facetT *facetlist, int **tet_verts) {

  facetT *facet;
  vertexT *vertex, **vertexp;
  int numfacets = 0;
  int v = 0; /* index in tets */

  /* count number of facets */
  FORALLfacet_(facetlist) {
    if ((facet->visible && qh NEWfacets) || (qh_skipfacet(facet)))
      facet->visitid= 0;
    else
      facet->visitid= ++numfacets;
  }

  *tet_verts = (int *)malloc(numfacets * 4 * sizeof(int));

  /* for all tets (facets to qhull) */
  FORALLfacet_(facetlist) {

    if (qh_skipfacet(facet) || (facet->visible && qh NEWfacets))
      continue;

    if (qh_setsize(facet->vertices) != 4) {
      fprintf(stderr, "tet has %d vertices; skipping.\n",
	      qh_setsize(facet->vertices));
      continue;
    }

    if ((facet->toporient ^ qh_ORIENTclock)
	|| (qh hull_dim > 2 && !facet->simplicial)) {
      FOREACHvertex_(facet->vertices)
	(*tet_verts)[v++] = qh_pointid(vertex->point);
    } else {
      FOREACHvertexreverse12_(facet->vertices)
	(*tet_verts)[v++] = qh_pointid(vertex->point);
    }

  } /* for all tets */

  assert(numfacets == v / 4); /* sanity */

  return numfacets;

}
/*--------------------------------------------------------------------------*/
#if 1
/*
  generates delaunay output from qhull

  facetlist: qhull list of convex hull facets
  tet_verts: pointer to array of tet vertex indices for this block 
  (allocated by this function, user's responsibility to free)

*/
void gen_d_delaunay_output(facetT *facetlist, struct dblock_t *dblock) {

  facetT *facet, *neighbor, **neighborp;
  vertexT *vertex, **vertexp;
  int numfacets = 0;
  int t, v, n; /* index in tets, tet verts, tet neighbors */
  int i, nbr;

  /* count number of tets (facets to qhull) */
  FORALLfacet_(facetlist) {
    if ((facet->visible && qh NEWfacets) || (qh_skipfacet(facet)))
      facet->visitid= 0;
    else
      facet->visitid= ++numfacets;
  }

  dblock->num_tets = numfacets;
  dblock->tets = (struct tet_t *)malloc(numfacets * sizeof(struct tet_t));

  /* for all tets (facets to qhull) */
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
      if (neighbor->visitid) {
	nbr = neighbor->visitid - 1;
	dblock->tets[t].tets[n] = neighbor->visitid - 1;
	if (nbr != -1)
	{
	  for (i = 0; i < 4; ++i)
	    if (dblock->tets[nbr].verts[i] == dblock->tets[t].verts[n])
	      fprintf(stderr, "Neighboring tet can't have a vertex it's opposite of: %d %d %d %d %d\n", t, nbr, i, n, dblock->tets[t].verts[n]);
	}
	++n;
      }
      else
	dblock->tets[t].tets[n++] = -1;
    }
    assert(n == 4); /* sanity */

    t++;

  } /* for all tets */

  assert(numfacets == t); /* sanity */

}
/*--------------------------------------------------------------------------*/
#else
/* based on Steve's version of neighbor ids, same results as mine
   DEPRECATED */
/*
  generates delaunay output from qhull

  facetlist: qhull list of convex hull facets
  tet_verts: pointer to array of tet vertex indices for this block 
  (allocated by this function, user's responsibility to free)

*/
void gen_d_delaunay_output(facetT *facetlist, struct dblock_t *dblock) {

  facetT *facet, *neighbor, **neighborp;
  ridgeT *ridge, **ridgep;
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
  int *id_map = (int *)malloc(numfacets * sizeof(int));

  dblock->num_tets = numfacets;
  dblock->tets = (struct tet_t *)malloc(numfacets * sizeof(struct tet_t));

  /* for all tets (facets to qhull) */
  t = 0;
  FORALLfacet_(facetlist) {

    if (qh_skipfacet(facet) || (facet->visible && qh NEWfacets))
      continue;

    if (qh_setsize(facet->vertices) != 4) {
      fprintf(stderr, "tet has %d vertices; skipping.\n",
	      qh_setsize(facet->vertices));
      continue;
    }
    id_map[t++] = facet->id;

  }

  /* for all tets (facets to qhull) */
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

    /* for all neighbor tets */
    n = 0;
    qh_makeridges(facet);
    FOREACHridge_(facet->ridges) {
      neighbor = otherfacet_(ridge, facet);

      int neigh_id = bin_search(id_map, neighbor->id, numfacets);
      if (neigh_id >= 0)
	dblock->tets[t].tets[n++] = neigh_id;
      else
	dblock->tets[t].tets[n++] = -1;
    }
    assert(n == 4); /* sanity */

/*     debug */
/*         fprintf(stderr, "1: tet %d verts [%d %d %d %d] neigh_tets [%d %d %d %d]\n", */
/*     	    t, dblock->tets[t].verts[0], dblock->tets[t].verts[1], */
/*     	    dblock->tets[t].verts[2], dblock->tets[t].verts[3], */
/*     	    dblock->tets[t].tets[0], dblock->tets[t].tets[1], */
/*     	    dblock->tets[t].tets[2], dblock->tets[t].tets[3]); */

    t++;

  } /* for all tets */

  free(id_map);
  assert(numfacets == t); /* sanity */

}
/*--------------------------------------------------------------------------*/
#endif
/*--------------------------------------------------------------------------*/
/* used with Steve's version of neighbor ids, DEPRECATED */
/*
  binary search
  tbl: lookup table
  key: search key
  size: number of table elements

  returns: index of key, -1 if not found
*/
int bin_search(int *tbl, int key, int size) {

  int max = size - 1;
  int min = 0;
  int mid;

  while (max >= min) {
    mid = (min + max) / 2;
    if (tbl[mid] < key )
      min = mid + 1;
    else if (tbl[mid] > key)
      max = mid - 1;
    else
      return mid;
  }

  return -1; /* not found */

}
/*--------------------------------------------------------------------------*/
