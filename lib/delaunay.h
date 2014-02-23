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
  int num_complete_cells; /* number of complete cells */
  int *complete_cells; /* sorted list of complete cells from raw tesselation
			  can be used to index into sites 
			  (ie, order of sites is same) */
  unsigned char *is_complete; /* each cell's completion status 0 or 1 */

  /* info for delaunay tetrahedralization */
  int num_loc_tets; /* number of delaunay tets with all local vertices */
  struct tet_t *loc_tets; /* strictly local delaunay tetrahedra */
  int num_rem_tets; /* number of  delaunay tets with >= 1 remote vertex */
  struct tet_t *rem_tets; /* remote (at least one particle) 
			     delaunay tetrahedra */
  int *rem_tet_gids; /* global ids of owners of vertices in remote tets */
  int *rem_tet_nids; /* native particle ids of vertices in remote tets */
  unsigned char *rem_tet_wrap_dirs; /* wrapping directions of vertices in
				       remote tets */

  /* min and max are separated so that DIY can omit some fields above in its
     datatype; not allowed to omit first or last field */
  float maxs[3]; /* block maximum corner */

};

#endif
