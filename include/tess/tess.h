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
#include <stdlib.h>
#include "delaunay.h"
#include "swap.hpp"
#include "utils.h"

/* timing and quantity stats */
enum {TOT_TIME, DEL1_TIME, DEL2_TIME, DEL3_TIME, NEIGH1_TIME, NEIGH2_TIME, OUT_TIME, MAX_TIMES};
enum {NUM_ORIG_PTS, NUM_FINAL_PTS, NUM_TETS, MAX_QUANTS};

/* public */

#ifdef __cplusplus
extern "C"
#endif
void tess_test(int tot_blocks, int mem_blocks, int *data_size, float jitter, float minvol, 
               float maxvol, int wrap, int twalls_on,  double *all_times, char *outfile, 
               MPI_Comm mpi_comm);

#if 0

#ifdef __cplusplus
extern "C"
#endif
struct dblock_t *tess_test_diy_exist(int nblocks, int *data_size, float jitter, float minvol, 
                                     float maxvol, int wrap, int twalls_on, double *times,
				     MPI_Comm mpi_comm);

#ifdef __cplusplus
extern "C"
#endif
void tess_init(int num_blocks, int *gids, struct bb_t *bounds, struct gb_t **neighbors, 
	       int *num_neighbors, float *global_mins, float *global_maxs, int wrap, 
               int twalls_on, float minvol, float maxvol, MPI_Comm mpi_comm, double *times);

#ifdef __cplusplus
extern "C"
#endif
void tess_init_diy_exist(int num_blocks, float *global_mins, float *global_maxs, int wrap, 
                         int twalls_on, float minvol, float maxvol, MPI_Comm mpi_comm, 
                         double *all_times);

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

#ifdef __cplusplus
extern "C"
#endif
void local_cells(struct dblock_t *b);

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
void clean_delaunay_data_structure(struct dblock_t* b);

#ifdef __cplusplus
extern "C"
#endif
void fill_vert_to_tet(struct dblock_t *dblock);

#ifdef __cplusplus
extern "C"
#endif
void get_mem(int breakpoint, int dwell);

void collect_stats();
void print_block(struct dblock_t *dblock, int gid);
void print_particles(float *particles, int num_particles, int gid);
void transform_particle(char *p, unsigned char wrap_dir);
int compare(const void *a, const void *b);
void write_particles(int nblocks, float **particles, int *num_particles, char *outfile);
void handle_error(int errcode, MPI_Comm comm, char *str);
void create_walls(int *num_walls, struct wall_t **walls);
void destroy_walls(int num_walls, struct wall_t *walls);
int test_outside(const float * pt, const struct wall_t *wall);
void generate_mirror(float *rpt, const float *pt, const struct wall_t *wall);
void add_mirror_particles(int nblocks, float **mirror_particles, int *num_mirror_particles, 
                          float **particles, int *num_particles, int *num_orig_particles,
			  int **gids, int **nids, unsigned char **dirs);
void timing(int start, int stop);

#endif
