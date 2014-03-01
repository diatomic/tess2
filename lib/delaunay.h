/*---------------------------------------------------------------------------
 *
 * delaunay block data structures
 *
 * Tom Peterka
 * Argonne National Laboratory
 * 9700 S. Cass Ave.
 * Argonne, IL 60439
 * tpeterka@mcs.anl.gov
 *
 * (C) 2011 by Argonne National Laboratory.
 * See COPYRIGHT in top-level directory.
 *
--------------------------------------------------------------------------*/
#ifndef _DELAUNAY_H
#define _DELAUNAY_H

#include "tet.h"

/* delaunay tessellation for one DIY block */
struct dblock_t {

  float mins[3]; /* block minimum corner */

  /* input particles */
  int num_orig_particles; /* number of original particles in this block
  			     before any neighbor exhcange */
  float *particles; /* original input points */
  unsigned char *is_complete; /* whether each oiginal particle is the site of
				 a complete voronoi cell or not */

  /* tets */
  int num_tets; /* number of delaunay tetrahedra */
  struct tet_t *tets; /* delaunay tets */
  int num_rem_tet_verts; /* number of remote delaunay vertices */
  struct remote_vert_t *rem_tet_verts; /* remote tet vertex (particle) info */
  int* vert_to_tet; /* a tet that contains the vertex */


  /* min and max are separated so that DIY can omit some fields above in its
     datatype; not allowed to omit first or last field */
  float maxs[3]; /* block maximum corner */

};

#endif
