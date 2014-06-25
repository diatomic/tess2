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

#include "mpi.h"
#include <stdlib.h>	// needed for RAND_MAX
#include "delaunay.h"
#include "swap.hpp"
#include "utils.h"

/* public */

#ifdef __cplusplus
extern "C"
#endif
void tess_test(int tot_blocks, int *data_size, float jitter,
	       float minvol, float maxvol, int wrap, int twalls_on, 
	       double *all_times, char *outfile, MPI_Comm mpi_comm);

#if 0

#ifdef __cplusplus
extern "C"
#endif
struct dblock_t *tess_test_diy_exist(int nblocks, int *data_size, float jitter, 
				     float minvol, float maxvol, int wrap,
				     int twalls_on, double *times,
				     MPI_Comm mpi_comm);

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

#endif

/* private */

struct dblock_t *delaunay(int nblocks, float **particles, int *num_particles,
			  double *times, char *out_file);

#ifdef __cplusplus
extern "C"
#endif
void destroy_blocks(int num_blocks, struct dblock_t *dblocks, int **hdrs);

#ifdef __cplusplus
extern "C"
#endif
void local_cells(int nblocks, struct dblock_t *dblocks, void *ds);

#ifdef __cplusplus
extern "C"
#endif
void d_local_cells(struct dblock_t *b);

#ifdef __cplusplus
extern "C"
#endif
void* init_delaunay_data_structures(int nblocks);
#ifdef __cplusplus
extern "C"
#endif
void init_delaunay_data_structure(struct dblock_t* b);
#ifdef __cplusplus
extern "C"
#endif
void clean_delaunay_data_structures(void* ds);

#ifdef __cplusplus
extern "C"
#endif
void fill_vert_to_tet(struct dblock_t *dblock);

#ifdef __cplusplus
extern "C"
#endif
void get_mem(int breakpoint, int dwell);

void neighbor_particles(int nblocks, struct dblock_t *dblocks);
/* void item_type(DIY_Datatype *type); */
/* void ic_type(DIY_Datatype *dtype); */
void collect_stats(int nblocks, struct dblock_t *dblocks, double *times);
void print_block(struct dblock_t *dblock, int gid);
void print_particles(float *particles, int num_particles, int gid);
void prep_out(int nblocks, struct dblock_t *dblocks, int **hdrs);
void transform_particle(char *p, unsigned char wrap_dir);
int compare(const void *a, const void *b);
void write_particles(int nblocks, float **particles, int *num_particles, 
		     char *outfile);
void handle_error(int errcode, MPI_Comm comm, char *str);
void gen_tets(struct dblock_t *dblock,
	      struct remote_ic_t *rics, int lid, int num_recvd);
void create_walls(int *num_walls, struct wall_t **walls);
void destroy_walls(int num_walls, struct wall_t *walls);
int test_outside(const float * pt, const struct wall_t *wall);
void generate_mirror(float *rpt, const float *pt, const struct wall_t *wall);
void add_mirror_particles(int nblocks, float **mirror_particles, 
			  int *num_mirror_particles, float **particles,
			  int *num_particles, int *num_orig_particles,
			  int **gids, int **nids, unsigned char **dirs);
void timing(double *times, int start, int stop);

#endif
