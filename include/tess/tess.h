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
--------------------------------------------------------------------------*/
#ifndef _TESS_H
#define _TESS_H

#include "mpi.h"
#include <stdlib.h>
#include "delaunay.h"
#include "swap.hpp"

/* timing and quantity stats */
enum
{
  TOT_TIME,
  EXCH_TIME,
  LOC1_TIME,
  INC1_TIME,
  DEL1_TIME,
  DEL2_TIME,
  DEL3_TIME,
  NEIGH1_TIME,
  NEIGH2_TIME,
  OUT_TIME,
  TESS_MAX_TIMES
};

enum
{
  NUM_ORIG_PTS,
  NUM_FINAL_PTS,
  NUM_TETS,
  MAX_QUANTS
};

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
void get_mem(int breakpoint);

void print_block(struct dblock_t *dblock, int gid);
void print_particles(float *particles, int num_particles, int gid);
void write_particles(int nblocks, float **particles, int *num_particles, char *outfile);
void handle_error(int errcode, MPI_Comm comm, char *str);
void create_walls(int *num_walls, struct wall_t **walls);
void destroy_walls(int num_walls, struct wall_t *walls);
int test_outside(const float * pt, const struct wall_t *wall);
void generate_mirror(float *rpt, const float *pt, const struct wall_t *wall);
void add_mirror_particles(int nblocks, float **mirror_particles, int *num_mirror_particles,
                          float **particles, int *num_particles, int *num_orig_particles,
			  int **gids, int **nids, unsigned char **dirs);
void timing(double* times, int start, int stop);

#endif
