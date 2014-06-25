/*---------------------------------------------------------------------------
 *
 * delaunay block
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
#include <stddef.h>

/* header */
#define NUM_VERTS 0 /* indices of header elements */
#define TOT_NUM_CELL_VERTS 1
#define NUM_COMPLETE_CELLS 2
#define TOT_NUM_CELL_FACES 3
#define NUM_ORIG_PARTICLES 4
#define NUM_LOC_TETS 5
#define NUM_REM_TETS 6
#define NUM_FACES 7
#define NUM_TETS 8
#define NUM_REM_TET_VERTS 9
#define NUM_PARTICLES 10

#define MAX_HIST_BINS 256 /* maximum number of bins in cell volume histogram */
#define MAX_NEIGHBORS 27 /* maximum number of neighbor blocks */

/* timing info */
#define TESS_MAX_TIMES 9 /* max number of timing components */
#define TOT_TIME 0 /* total time */
#define LOC1_TIME 1 /* first local delaunay */
#define LOC2_TIME 2 /* second local delaunay */
#define LOC3_TIME  3 /* third local delaunay*/
#define NEIGH1_TIME  4 /* first neighbor exchange */
#define NEIGH2_TIME  5 /* second neighbor exchange */
#define INC1_TIME 6 /* incomplete_cells_initial */
#define INC2_TIME 7 /* incomplete_cells_final */
#define OUT_TIME 8 /* file output */

/* remote particle */
struct RemotePoint {
  float x, y, z; /* coordinates */
  int gid; /* owner block global id */
  int nid; /* native index of particle in owner block */
  unsigned char dir; /*wrapping direction */
};

/* completion status for cell associated with remote particle */
struct remote_ic_t {
  int is_complete; /* 0 or 1 indicates whether remote particle is complete */
  int gid; /* owner block global id */
  int nid; /* native index of particle in owner block */
};

/* remote tet vertex */
struct remote_vert_t {
  int gid; /* global ids of owner block */
  int nid; /* native particle id in owner block */
  unsigned char dir; /* wrapping direction */
};

/* CLP  - struct walls - using general equation of plane 
   per http://mathworld.wolfram.com/Plane.html */
struct wall_t {
float a;
float b;
float c;
float d;
};

/* delaunay tessellation for one DIY block */
struct dblock_t {

  int gid; /* global block id */
  float mins[3]; /* block minimum corner */
  void *Dt; /* native delaunay data structure */

  /* input particles */
  int num_orig_particles; /* number of original particles in this block
  			     before any neighbor exhcange */
  int num_particles; /* current number of particles in this block after any
			neighbor exchange; original particles appear first 
			followed by received particles; 
			only original particles are saved to disk */
  float *particles; /* original input points */

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
