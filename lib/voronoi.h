/*---------------------------------------------------------------------------
 *
 * voronoi data structures
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
#ifndef _VORONOI_H
#define _VORONOI_H

/* header */
#define NUM_VERTS 0 /* indices of header elements */
#define TOT_NUM_CELL_VERTS 1
#define NUM_COMPLETE_CELLS 2
#define TOT_NUM_CELL_FACES 3
#define NUM_ORIG_PARTICLES 4
#define NUM_LOC_TETS 5
#define NUM_REM_TETS 6
#define NUM_FACES 7

#define MAX_HIST_BINS 256 /* maximum number of bins in cell volume histogram */
#define MAX_FACE_VERTS 24 /* maximum number of vertices per face */

/* timing info */
#define MAX_TIMES 8 /* max number of timing components */
#define EXCH_TIME 0 /* particle exchange */
#define CELL_TIME 1 /* voronoi / delaunay computation */
#define VOL_TIME 2 /* volumes and areas of voronoi cells */
#define OUT_TIME  3 /* file output */
#define LOCAL_TIME  4 /* local voronoi / delaunay computation */

/* remote particle */
struct remote_particle_t {
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

/* sent particle */
struct sent_t {
  int particle; /* site (particle) index sent to neighbor */
  float ghost; /* ghost size used to send this particle */
};

/* face between up to two voronoi cells */
struct vface_t {
  int cells[2]; /* up to two voronoi cells, or -1 if no nneighbo cell */
  int num_verts; /* number of vertices in the face */
  int verts[MAX_FACE_VERTS]; /* vertex indices for the face */
};

/* voronoi and delaunay tessellation for one DIY block */
struct vblock_t {

  float mins[3]; /* block minimum corner */

  /* raw voronoi tesselation */
  int num_verts; /* number of vertices, including "infinite" vertex */
  double *verts; /* double version of vertices x0, y0, z0, x1, y, z1, ...
		    infinite vertex is x0, y0, z0, for in-core computations */
  float *save_verts; /* float version of verts for permananent storage */
  int *num_cell_verts; /* number of vertices in each cell */
  int tot_num_cell_verts; /* sum of all num_cell_verts */
  int *cells; /* vertices in cells, in order of cells */
  float *sites; /* original input points = sites of voronoi cells in same
		   order as (original, not necessarily complete) cells */

  /* temporary complete cells */
  int temp_num_complete_cells; /* number of complete cells */
  int *temp_complete_cells; /* sorted list of complete cells from 
			       raw tesselation */

  /* final complete cells */
  int num_complete_cells; /* number of complete cells */
  int *complete_cells; /* sorted list of complete cells from raw tesselation
			  can be used to index into sites 
			  (ie, order of sites is same) */
  unsigned char *is_complete; /* each cell's completion status 0 or 1 */
  float *areas; /* surface areas of complete cells */
  float *vols; /* volumes of complete cells */
  float *face_areas; /* area of each face in faces array */
  int num_faces; /* total number of unique faces */
  struct vface_t *faces; /* faces between up to two voronoi cells */
  int *cell_faces_start; /* starting offset in faces for each cell, both
			    complete and incomplete, in order of
			    originalk particles */
  int tot_num_cell_faces; /* total number of faces in all cells,
				 both complete and incomplete */
  int *cell_faces; /* faces in each cell, both complete and incomplete,
		      as indices into faces array */

  /* info for delaunay tetrahedralization */
  int num_loc_tets; /* number of delaunay tets with all local vertices */
  int *loc_tets; /* strictly local delaunay tetrahedra
  		each four site ids are another tetrahedron
  		(site_id0, site_id1, site_id2, site_id3), (site_id0, ... */
  int num_rem_tets; /* number of  delaunay tets with >= 1 remote vertex */
  int *rem_tet_gids; /* global ids of owners of vertices in remote tets */
  int *rem_tet_nids; /* native particle ids of vertices in remote tets */
  unsigned char *rem_tet_wrap_dirs; /* wrapping directions of vertices in
				       remote tets */

  /* additional info */
  int num_orig_particles; /* number of original particles in this block
  			     before any neighbor exhcange */
  int num_sent_particles; /* number of particles sent to neighbors */
  int alloc_sent_particles; /* allocated number of sent particles */
  struct sent_t *sent_particles; /* sent particle ids and neighbor directions */

  float maxs[3]; /* block maximum corner */

};

/* statistical summary */
struct stats_t {
  int tot_tets; /* total number of delaunay tetrahedra found */
  int tot_cells; /* total number of complete voronoi cells found */
  int tot_faces; /* total number of faces in all complete voronoi cells */
  int tot_verts; /* total number of vertices in all complete voronoi cells */
  float avg_cell_verts; /* average number of vertices per cell */
  float avg_cell_faces; /* average number of faces per cell */
  float avg_face_verts; /* average number of vertices per face */
  float min_cell_vol; /* minimum cell volume */
  float max_cell_vol; /* maximum cell volume */
  float avg_cell_vol; /* average cell volume */
  float min_cell_dense; /* minimum cell density */
  float max_cell_dense; /* maximum cell density */
  float avg_cell_dense; /* average cell density */
  float min_cell_time; /* minimum voronoi cell time */
  float max_cell_time; /* maximum voronoi cell time */
  float min_vol_time; /* minimum convex hull time */
  float max_vol_time; /* maximum convex hull time */
  int num_vol_bins; /* number of bins in cell volume histogram */
  int num_dense_bins; /* number of bins in cell density histogram */
  int vol_hist[MAX_HIST_BINS]; /* cell volume histogram */
  int dense_hist[MAX_HIST_BINS]; /* cell density histogram */
};

#endif
