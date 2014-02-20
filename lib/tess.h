/*---------------------------------------------------------------------------
 *
 * parallel tesselation
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
#ifndef _TESS_H
#define _TESS_H

#include <stdlib.h>	// needed for RAND_MAX

#include "voronoi.h"
#include "swap.hpp"
#include "diy.h"

extern MPI_Comm comm; /* MPI communicator */

/* public */

#ifdef __cplusplus
extern "C"
#endif
void tess_test(int tot_blocks, int *data_size, float jitter,
	       float minvol, float maxvol, int wrap, int twalls_on, double *all_times,
	       char *outfile);

#ifdef __cplusplus
extern "C"
#endif
void tess_init(int num_blocks, int *gids, 
	       struct bb_t *bounds, struct gb_t **neighbors, 
	       int *num_neighbors, float *global_mins, float *global_maxs, 
	       int wrap, int twalls_on, float minvol, float maxvol, 
	       MPI_Comm mpi_comm, double *times);

#ifdef __cplusplus
extern "C"
#endif
void tess_init_diy_exist(int num_blocks, float *global_mins, 
			 float *global_maxs, int wrap, int twalls_on,
			 float minvol, float maxvol, 
			 MPI_Comm mpi_comm, double *all_times);

#ifdef __cplusplus
extern "C"
#endif
void tess_finalize();

#ifdef __cplusplus
extern "C"
#endif
void tess(float **particles, int *num_particles, char *out_file);

/* private */

void voronoi_delaunay(int nblocks, float **particles, int *num_particles, 
		      double *times, char *out_file);
int gen_particles(int lid, float **particles, float jitter);
#ifdef __cplusplus
extern "C"
#endif
void complete_cells(struct vblock_t *vblock, int lid);
void incomplete_cells_initial(struct vblock_t *tblock, struct vblock_t *vblock, 
			      int lid, int** convex_hull_particles, 
			      int* num_convex_hull_particles);
void incomplete_cells_final(struct vblock_t *tblock, struct vblock_t *vblock, 
			    int lid, int* convex_hull_particles, 
			    int num_convex_hull_particles,
			    struct wall_t *walls,
			    int num_walls,
			    float** mirror_particles,
			    int*  num_mirror_particles);
#ifdef __cplusplus
extern "C"
#endif
void all_cells(int nblocks, struct vblock_t *vblocks, int dim,
	       int *num_particles, int *num_orig_particles, 
	       float **particles, int **gids, int **nids, 
	       unsigned char **dirs, double *times, void* ds);
#ifdef __cplusplus
extern "C"
#endif
void cell_faces(struct vblock_t *vblock);
void create_blocks(int num_blocks, struct vblock_t **vblocks, int ***hdrs);
#ifdef __cplusplus
extern "C"
#endif
void local_cells(int nblocks, struct vblock_t *tblocks, int dim,
		 int *num_particles, float **particles, void* ds);
#ifdef __cplusplus
extern "C"
#endif
void* init_delaunay_data_structures(int nblocks);
#ifdef __cplusplus
extern "C"
#endif
void clean_delaunay_data_strucutres(void* ds);

void neighbor_particles(int nblocks, float **particles,
			int *num_particles, int *num_orig_particles,
			int **gids, int **nids, unsigned char **dirs);
#ifdef __cplusplus
extern "C"
#endif
void neighbor_is_complete(int nblocks, struct vblock_t *vblocks,
			  struct remote_ic_t **rics);
void item_type(DIY_Datatype *type);
void ic_type(DIY_Datatype *dtype);
void destroy_blocks(int num_blocks, struct vblock_t *vblocks, int **hdrs);
void reset_blocks(int num_blocks, struct vblock_t *vblocks);
void collect_stats(int nblocks, struct vblock_t *vblocks, double *times);
void aggregate_stats(int nblocks, struct vblock_t *vblocks, 
		     struct stats_t *loc_stats);
void average(void *in, void *inout, int *len, MPI_Datatype *type);
void histogram(void *in, void *inout, int *len, MPI_Datatype *type);
void print_block(struct vblock_t *vblock, int gid);
void print_particles(float *particles, int num_particles, int gid);
void prep_out(int nblocks, struct vblock_t *vblocks);
void save_headers(int nblocks, struct vblock_t *vblocks, int **hdrs);
void transform_particle(char *p, unsigned char wrap_dir);
void delaunay(int num_blocks, struct vblock_t *vblocks);
void neigh_cells(struct vblock_t *vblock, int cell, int face, int cur_vert);
int compare(const void *a, const void *b);
#ifdef __cplusplus
extern "C"
#endif
void add_int(int val, int **vals, int *numvals, int *maxvals, int chunk_size);
void add_float(float val, float **vals, int *numvals, int *maxvals, 
	       int chunk_size);
void add_pt(float *val, float **vals, int *numvals, int *maxvals, 
	    int chunk_size);
void add_empty_int(int **vals, int index, int *numitems, int *maxitems, 
		  int chunk_size, int init_val);
void add_sent(struct sent_t val, struct sent_t **vals, int *numvals, 
	      int *maxvals, int chunk_size);
int cell_bounds(struct vblock_t *vblock, int cell, int vert);
void cell_vols(int nblocks, struct vblock_t *vblocks, float **particles);
void face_areas(int nblocks, struct vblock_t *vblocks);
void write_particles(int nblocks, float **particles, int *num_particles, 
		     char *outfile);
void handle_error(int errcode, MPI_Comm comm, char *str);
#ifdef __cplusplus
extern "C"
#endif
void gen_tets(int *tet_verts, int num_tets, struct vblock_t *vblock,
	      int *gids, int *nids, unsigned char *dirs,
	      struct remote_ic_t *rics, int lid, int num_recvd);

void create_walls(int *num_walls, struct wall_t **walls);
void destroy_walls(int num_walls, struct wall_t *walls);
int test_outside(const float * pt, const struct wall_t *wall);
void generate_mirror(float *rpt, const float *pt, const struct wall_t *wall);
void add_mirror_particles(int nblocks, float **mirror_particles, 
			  int *num_mirror_particles, float **particles,
			  int *num_particles, int *num_orig_particles,
			  int **gids, int **nids, unsigned char **dirs);
#endif
