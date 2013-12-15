/*---------------------------------------------------------------------------
 *
 * parallel netcdf I/O for voronoi tesselation
 *
 * Wei-keng Liao (Northwestern University)
 * Tom Peterka
 * Argonne National Laboratory
 * 9700 S. Cass Ave.
 * Argonne, IL 60439
 * tpeterka@mcs.anl.gov
 *
 * (C) 2013 by Argonne National Laboratory.
 * See COPYRIGHT in top-level directory.
 *
--------------------------------------------------------------------------*/
#ifndef _IO_H
#define _IO_H

#include "diy.h"


#define MIN(a,b) (((a)<(b))?(a):(b))

#define ERR {if(err!=NC_NOERR)printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err));}

/* quantity vector */
#define NUM_QUANTS 9
#define NUM_VERTICES 0
#define NUM_COMP_CELLS 1
#define NUM_CELL_FACES 2
#define NUM_ORIG_PARTS 3
#define NUM_NEIGHBORS 4
#define NUM_BLOCKS 5
#define NUM_LOC_TETRAS 6
#define NUM_REM_TETRAS 7
#define NUM_UNIQUE_FACES 8

#ifdef __cplusplus
extern "C"
#endif
void diy_write(int nblocks, struct vblock_t *vblocks, int **hdrs,
	       char *out_file);

#ifdef __cplusplus
extern "C"
#endif
void pnetcdf_write(int nblocks, struct vblock_t *vblocks, 
		   char *out_file, MPI_Comm comm);

#ifdef __cplusplus
extern "C"
#endif
void pnetcdf_read(int *nblocks, int *tot_blocks, struct vblock_t ***vblocks, 
		  char *in_file, MPI_Comm comm, int **gids, 
		  int **num_neighbors, int ***neighbors, int ***neigh_procs);

#ifdef __cplusplus
extern "C"
#endif
void create_datatype(void *vblock, int did, int lid, DIY_Datatype *dtype);

#endif
