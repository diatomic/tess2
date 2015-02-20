/*---------------------------------------------------------------------------
 *
 * parallel netcdf I/O for tesselation
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

/* DEPRECATED */

/* #include "delaunay.h" */
/* #include <diy/types.h> */

/* #define MIN(a,b) (((a)<(b))?(a):(b)) */

/* #define ERR {if(err!=NC_NOERR)printf("Error at line %d in %s: %s\n", __LINE__, __FILE__, ncmpi_strerror(err));} */

/* /\* quantity vector *\/ */
/* #define NUM_QUANTS 5 */
/* #define NUM_NEIGHBORS 0 */
/* #define NUM_BLOCKS 1 */
/* #define NUM_TETRAS 2 */
/* #define NUM_REM_GIDS 3 */
/* #define NUM_PARTS 4 */

/* #ifdef __cplusplus */
/* extern "C" */
/* #endif */
/* void diy_write(int nblocks, struct dblock_t *dblocks, int **hdrs, char *out_file); */

/* #ifdef __cplusplus */
/* extern "C" */
/* #endif */
/* void pnetcdf_write(int nblocks, struct dblock_t **dblocks, */
/* 		   char *out_file, MPI_Comm comm, int *num_nbrs, struct gb_t **nbrs); */

/* #ifdef __cplusplus */
/* extern "C" */
/* #endif */
/* void pnetcdf_read(int *nblocks, int *tot_blocks, struct dblock_t **dblocks, char *in_file, */
/*                   MPI_Comm comm, int **num_neighbors, int ***neighbors, int ***neigh_procs); */

#endif
