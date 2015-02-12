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
#include "diy/types.h"

#define MAX_HIST_BINS 256     /* maximum number of bins in cell volume histogram */
#define MAX_NEIGHBORS 27      /* maximum number of neighbor blocks */

/* remote particle */
struct point_t {
  float x, y, z;              /* coordinates */
  int gid;                    /* owner block global id */
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

  int gid;                   /* global block id */
  float mins[3], maxs[3];    /* block extents */
  struct bb_c_t data_bounds; /* global data extents */
  struct bb_c_t box;	     /* current box; used in swap-reduce() when distributing particles */
  void* Dt;                  /* native delaunay data structure */

  /* input particles */
  int num_orig_particles;    /* number of original particles in this block
                                before any neighbor exchange */
  int num_particles;         /* current number of particles in this block after any
                                neighbor exchange; original particles appear first
                                followed by received particles */
  float* particles;          /* all particles, original plus those received from neighbors */

  /* tets */
  int num_tets;              /* number of delaunay tetrahedra */
  struct tet_t* tets;        /* delaunay tets */
  int* rem_gids;             /* owners of remote particles */
  int* vert_to_tet;          /* a tet that contains the vertex */

  /* sent particles and convex hull particles
     these persist between phases of the algorithm but ar not saved in the final output
     using void* for each so that C files tess-qhull.c and io.c can see them */
  void* convex_hull_particles;
  void* sent_particles;

  /* estimated density field */
  float* density;            /* density field */
  int num_grid_pts;          /* total number of density grid points */
};

#endif
