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
  float mins[3], maxs[3]; /* block extents */
  void *Dt; /* native delaunay data structure */

  /* input particles */
  int num_orig_particles; /* number of original particles in this block
  			     before any neighbor exhcange */
  int num_particles; /* current number of particles in this block after any
			neighbor exchange; original particles appear first 
			followed by received particles */
  float *particles; /* all particles, original plus those received from neighbors */

  /* tets */
  int num_tets; /* number of delaunay tetrahedra */
  struct tet_t *tets; /* delaunay tets */
  int* vert_to_tet; /* a tet that contains the vertex */

  /* sent particles and convex hull particles
     these persist between phases of the algorithm but ar not saved in the final output
     using void* for each so that C files tess-qhull.c and io.c can see them */
  void* convex_hull_particles;
  void* sent_particles;

};

#endif
