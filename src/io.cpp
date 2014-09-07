/*---------------------------------------------------------------------------
 *
 * parallel netcdf I/O for voronoi tesselation
 *
 * Tom Peterka
 * Argonne National Laboratory
 * 9700 S. Cass Ave.
 * Argonne, IL 60439
 * tpeterka@mcs.anl.gov
 *
--------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <pnetcdf.h>
#include "tess/delaunay.h"
#include "tess/io.h"

/*---------------------------------------------------------------------------
/*
  writes output in pnetcdf format

  nblocks: local number of blocks
  dblocks: array of pointers to dblocks
  out_file: output file name
  comm: MPI communicator
  num_nbrs: number of neighbors for each local block
  nbrs: neighbors of each local block
*/
void pnetcdf_write(int nblocks, struct dblock_t **dblocks,
		   char *out_file, MPI_Comm comm, int *num_nbrs, struct gb_t **nbrs)
{
  int err;
  int ncid, cmode, varids[41], dimids[14], dimids_2D[2];
  MPI_Offset start[2], count[2];

  MPI_Offset proc_quants[NUM_QUANTS]; /* quantities per process */
  MPI_Offset tot_quants[NUM_QUANTS]; /* total quantities all global blocks */
  MPI_Offset block_ofsts[NUM_QUANTS]; /* starting offsets for each block */

  /* init */
  int i;
  for (i = 0; i < NUM_QUANTS; i++) {
    proc_quants[i] = 0;
    tot_quants[i] = 0;
    block_ofsts[i] = 0;
  }

  /* sum quantities over local blocks */
  int b;
  for (b = 0; b < nblocks; b++) {
    proc_quants[NUM_PARTS] += dblocks[b]->num_particles;
    proc_quants[NUM_NEIGHBORS] += num_nbrs[b];
    /* tets are written as a 2d array of ints [(verts, tets) x 4 indices]
       the 2 below refers to the two rows: verts and tets
       the 4 indices (2nd dimension) is defined later */
    proc_quants[NUM_TETRAS] += 2 * dblocks[b]->num_tets;
    proc_quants[NUM_REM_GIDS] += (dblocks[b]->num_particles - dblocks[b]->num_orig_particles);
  }
  proc_quants[NUM_BLOCKS] = nblocks;

  /* sum per process values to be global ones */
  MPI_Allreduce(proc_quants, tot_quants, NUM_QUANTS, MPI_OFFSET, MPI_SUM, comm);

  /* prefix sum proc offsets */
  MPI_Exscan(proc_quants, &block_ofsts, NUM_QUANTS, MPI_OFFSET, MPI_SUM, comm);

  /* create a new file for writing */
  cmode = NC_CLOBBER | NC_64BIT_DATA;
  err = ncmpi_create(comm, out_file, cmode, MPI_INFO_NULL, &ncid); ERR;

  /* --- define dimensions --- */

  err = ncmpi_def_dim(ncid, "num_g_blocks", tot_quants[NUM_BLOCKS],
		      &dimids[0]); ERR;
  err = ncmpi_def_dim(ncid, "XYZ", 3, &dimids[1]); ERR;
  err = ncmpi_def_dim(ncid, "num_g_particles", tot_quants[NUM_PARTS],
		      &dimids[6]); ERR;
  err = ncmpi_def_dim(ncid, "num_g_neighbors", tot_quants[NUM_NEIGHBORS],
		      &dimids[7]); ERR;
  err = ncmpi_def_dim(ncid, "num_g_tets", tot_quants[NUM_TETRAS],
		      &dimids[8]); ERR;
  err = ncmpi_def_dim(ncid, "V0V1V2V3", 4, &dimids[9]); ERR;
  err = ncmpi_def_dim(ncid, "num_g_rem_gids", tot_quants[NUM_REM_GIDS], &dimids[10]); ERR;

  /* --- define variables --- */

  /* quantities */
  err = ncmpi_def_var(ncid, "num_orig_particles", NC_INT, 1, &dimids[0],
		      &varids[4]); ERR;
  err = ncmpi_def_var(ncid, "num_particles", NC_INT, 1, &dimids[0],
		      &varids[5]); ERR;
  err = ncmpi_def_var(ncid, "num_tets", NC_INT, 1, &dimids[0],
		      &varids[25]); ERR;

  /* block offsets
     encode the offset in the full array of each variable wheree the first
     block of each process starts */
  err = ncmpi_def_var(ncid, "block_off_num_particles", NC_INT64, 1,
		      &dimids[0], &varids[9]); ERR;
  err = ncmpi_def_var(ncid, "block_off_num_tets", NC_INT64, 1,
		      &dimids[0], &varids[26]); ERR;
  err = ncmpi_def_var(ncid, "block_off_num_rem_gids", NC_INT64, 1,
		      &dimids[0], &varids[29]); ERR;
  err = ncmpi_def_var(ncid, "block_off_num_neighbors", NC_INT64, 1,
		      &dimids[0], &varids[40]); ERR;

  /* other data */
  dimids_2D[0] = dimids[0];
  dimids_2D[1] = dimids[1];
  err = ncmpi_def_var(ncid, "mins", NC_FLOAT, 2, dimids_2D, &varids[11]); ERR;
  err = ncmpi_def_var(ncid, "maxs", NC_FLOAT, 2, dimids_2D, &varids[12]); ERR;
  dimids_2D[0] = dimids[6];
  dimids_2D[1] = dimids[1];
  err = ncmpi_def_var(ncid, "particles", NC_FLOAT, 2, dimids_2D,
		      &varids[14]); ERR;
  err = ncmpi_def_var(ncid, "num_neighbors", NC_INT, 1, &dimids[0],
		      &varids[24]); ERR;
  err = ncmpi_def_var(ncid, "neighbors", NC_INT, 1, &dimids[7],
		      &varids[21]); ERR;
  err = ncmpi_def_var(ncid, "neigh_procs", NC_INT, 1, &dimids[7],
		      &varids[22]); ERR;
  err = ncmpi_def_var(ncid, "gids", NC_INT, 1, &dimids[0],
		      &varids[23]); ERR;
  dimids_2D[0] = dimids[8];
  dimids_2D[1] = dimids[9];
  err = ncmpi_def_var(ncid, "tets", NC_INT, 2, dimids_2D,
		      &varids[27]); ERR;
  err = ncmpi_def_var(ncid, "rem_gids", NC_INT, 1, &dimids[10],
		      &varids[30]); ERR;
  err = ncmpi_def_var(ncid, "vert_to_tet", NC_INT, 1, &dimids[6],
		      &varids[33]); ERR;

  /* exit define mode */
  err = ncmpi_enddef(ncid); ERR;

  /* write all variables.
     to improve: we can try nonblocking I/O to aggregate small requests */

  for (b = 0; b < nblocks; b++) {

    struct dblock_t *d = dblocks[b];

    /* quantities */
    start[0] = block_ofsts[NUM_BLOCKS];
    count[0] = 1;
    err = ncmpi_put_vara_int_all(ncid, varids[4], start, count,
				 &d->num_orig_particles); ERR;
    err = ncmpi_put_vara_int_all(ncid, varids[5], start, count,
				 &d->num_particles); ERR;
    err = ncmpi_put_vara_int_all(ncid, varids[25], start, count,
				 &d->num_tets); ERR;

    /* block offsets */
    err = ncmpi_put_vara_longlong_all(ncid, varids[9], start, count,
				      &block_ofsts[NUM_PARTS]); ERR;
    err = ncmpi_put_vara_longlong_all(ncid, varids[26], start, count,
				      &block_ofsts[NUM_TETRAS]); ERR;
    err = ncmpi_put_vara_longlong_all(ncid, varids[29], start, count,
				      &block_ofsts[NUM_REM_GIDS]); ERR;
    err = ncmpi_put_vara_longlong_all(ncid, varids[40], start, count,
				      &block_ofsts[NUM_NEIGHBORS]); ERR;

    /* block bounds */
    start[0] = block_ofsts[NUM_BLOCKS];
    count[0] = 1;
    start[1] = 0;
    count[1] = 3;
    err = ncmpi_put_vara_float_all(ncid, varids[11], start, count,
				   d->mins); ERR;
    err = ncmpi_put_vara_float_all(ncid, varids[12], start, count,
				   d->maxs); ERR;

    /* particles */
    start[0] = block_ofsts[NUM_PARTS];
    start[1] = 0;
    count[0] = d->num_particles;
    count[1] = 3;
    err = ncmpi_put_vara_float_all(ncid, varids[14], start, count,
				   d->particles); ERR;

    /* num_neighbors, neighbors, neigh_procs */
    struct gb_t *neigh_gbs =
      (struct gb_t *)malloc(num_nbrs[b] * sizeof(struct gb_t));
    int *neighbors = (int*)malloc(num_nbrs[b] * sizeof(int));
    int *neigh_procs = (int*)malloc(num_nbrs[b] * sizeof(int));
    for (i = 0; i < num_nbrs[b]; i++) {
      neighbors[i] = nbrs[b][i].gid;
      neigh_procs[i] = nbrs[b][i].proc;
    }
    start[0] = block_ofsts[NUM_BLOCKS];
    count[0] = 1;
    err = ncmpi_put_vara_int_all(ncid, varids[24], start, count,
				 &num_nbrs[b]); ERR;
    start[0] = block_ofsts[NUM_NEIGHBORS];
    count[0] = num_nbrs[b];
    err = ncmpi_put_vara_int_all(ncid, varids[21], start, count, neighbors);
    ERR;
    err = ncmpi_put_vara_int_all(ncid, varids[22], start, count, neigh_procs);
    ERR;

    /* gid */
    start[0] = block_ofsts[NUM_BLOCKS];
    count[0] = 1;
    err = ncmpi_put_vara_int_all(ncid, varids[23], start, count, &d->gid); ERR;

    /* tets */
    count[0] = d->num_tets * 2; /* verts and neighbors combined */
    count[1] = (count[0] ? 4 : 0);
    start[0] = (count[0] ? block_ofsts[NUM_TETRAS] : 0);
    start[1] = 0;
    /* casting array of struct of ints to simple 2D array of ints
       because pnetcdf likes arrays */
    err = ncmpi_put_vara_int_all(ncid, varids[27], start, count,
				 (int *)(d->tets)); ERR;

    /* remote gids */
    count[0] = d->num_particles - d->num_orig_particles;
    start[0] = (count[0] ? block_ofsts[NUM_REM_GIDS] : 0);
    err = ncmpi_put_vara_int_all(ncid, varids[30], start, count, d->rem_gids); ERR;

    /* vert_to_tet */
    start[0] = block_ofsts[NUM_PARTS];
    count[0] = d->num_particles;
    err = ncmpi_put_vara_int_all(ncid, varids[33], start, count,
				 d->vert_to_tet); ERR;

    /* update block offsets */
    block_ofsts[NUM_PARTS] += d->num_particles;
    block_ofsts[NUM_NEIGHBORS] += num_nbrs[b];
    block_ofsts[NUM_BLOCKS]++;
    /* 2x because tets are written as a 2d array of ints with row dimension 2 */
    block_ofsts[NUM_TETRAS] += 2 * d->num_tets;
    block_ofsts[NUM_REM_GIDS] += (d->num_particles - d->num_orig_particles);

    /* cleanup */
    free(neighbors);
    free(neigh_procs);
    free(neigh_gbs);

  }

  err = ncmpi_close(ncid); ERR;

}
/*--------------------------------------------------------------------------*/
/*
  reads input in pnetcdf format

  nblocks: (output) local number of blocks
  tot_blocks: (output) total number of blocks
  dblocks: (output) pointer to array of dblocks
  in_file: input file name
  comm: MPI communicator
  num_neighbors: (output) pointer to array of number of neighbors for
   each local block (allocated by this function)
  neighbors: (output) pointer to 2D array of gids of neighbors of each
   local block (allocated by this function)
  neigh_procs: (output) pointer to 2D array of procs of neighbors of each
   local block (allocated by this function)

  side effects: allocates dblocks, num_neighbors, neighbors, neigh_procs

*/
void pnetcdf_read(int *nblocks, int *tot_blocks, struct dblock_t **dblocks,
		  char *in_file, MPI_Comm comm, int **num_neighbors,
		  int ***neighbors, int ***neigh_procs) {

  int err;
  int ncid, varids[41], dimids[14];
  MPI_Offset start[2], count[2];
  nc_type type;
  int ndims, natts;
  int rank, groupsize; /* MPI usual */
  int i;

  /* open file for reading */
  err = ncmpi_open(comm, in_file, NC_NOWRITE, MPI_INFO_NULL, &ncid); ERR;

  err = ncmpi_inq_varid(ncid, "block_off_num_particles", &varids[9]); ERR;
  err = ncmpi_inq_varid(ncid, "block_off_num_tets", &varids[26]); ERR;
  err = ncmpi_inq_varid(ncid, "block_off_num_rem_gids", &varids[29]); ERR;
  err = ncmpi_inq_varid(ncid, "block_off_num_neighbors", &varids[40]); ERR;

  /* get number of blocks */
  MPI_Offset num_g_blocks; /* 64 bit version of tot_blcoks */
  err = ncmpi_inq_varid(ncid, "mins", &varids[11]); ERR;
  err = ncmpi_inq_var(ncid, varids[11], 0, &type, &ndims,
		      dimids, &natts); ERR;
  err = ncmpi_inq_dimlen(ncid, dimids[0], &num_g_blocks); ERR;
  *tot_blocks = num_g_blocks;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &groupsize);

  /* todo: following only works for same number of blocks for all procs */
  int bp = *tot_blocks / groupsize; /* nominal blocks per process */
  int start_block_ofst =  rank * bp;
  /* end of restriction */

  *nblocks = (rank < groupsize - 1 ? bp : *tot_blocks - (rank * bp));

  /* all gids and block index in file of each gid
     todo: not scalable*/
  int *all_gids = (int *)malloc(*tot_blocks * sizeof(int));
  int *gid2idx = (int *)malloc(*tot_blocks * sizeof(int));
  start[0] = 0;
  count[0] = *tot_blocks;
  err = ncmpi_inq_varid(ncid, "gids", &varids[23]); ERR;
  err = ncmpi_get_vara_int_all(ncid, varids[23], start, count,
			       all_gids); ERR;
  for (i = 0; i < *tot_blocks; i++)
    gid2idx[all_gids[i]] = i;

  /* block offsets */
  int64_t *block_ofsts = (int64_t*)malloc(*tot_blocks * sizeof(int64_t));
  *dblocks = (struct dblock_t*)malloc(*nblocks * sizeof(struct dblock_t));

  /* read all blocks */
  *num_neighbors = (int *)malloc(*nblocks * sizeof(int));
  *neighbors = (int **)malloc(*nblocks * sizeof(int *));
  *neigh_procs = (int **)malloc(*nblocks * sizeof(int *));

  /* for all blocks */
  int b;
  for (b = 0; b < *nblocks; b++) {

    struct dblock_t *d = &((*dblocks)[b]);

    /* quantities */
    start[0] = start_block_ofst + b;
    count[0] = 1;
    err = ncmpi_inq_varid(ncid, "num_orig_particles", &varids[4]); ERR;
    err = ncmpi_get_vara_int_all(ncid, varids[4], start, count,
				 &(d->num_orig_particles)); ERR;
    err = ncmpi_inq_varid(ncid, "num_particles", &varids[5]); ERR;
    err = ncmpi_get_vara_int_all(ncid, varids[5], start, count,
				 &(d->num_particles)); ERR;
    err = ncmpi_inq_varid(ncid, "num_tets", &varids[25]); ERR;
    err = ncmpi_get_vara_int_all(ncid, varids[25], start, count,
				 &(d->num_tets)); ERR;

    /* block bounds */
    start[0] = start_block_ofst + b;
    start[1] = 0;
    count[0] = 1;
    count[1] = 3;
    err = ncmpi_inq_varid(ncid, "mins", &varids[11]); ERR;
    err = ncmpi_get_vara_float_all(ncid, varids[11], start, count,
				   d->mins); ERR;
    err = ncmpi_inq_varid(ncid, "maxs", &varids[12]); ERR;
    err = ncmpi_get_vara_float_all(ncid, varids[12], start, count,
				   d->maxs); ERR;

    /* particles */
    start[0] = 0;
    count[0] = *tot_blocks;
    err = ncmpi_get_vara_longlong_all(ncid, varids[9], start, count,
				      (long long *)block_ofsts); ERR;
    d->particles =
      (float *)malloc(d->num_particles * 3 * sizeof(float));
    start[0] = block_ofsts[start_block_ofst + b];
    start[1] = 0;
    count[0] = d->num_particles;
    count[1] = 3;
    err = ncmpi_inq_varid(ncid, "particles", &varids[14]); ERR;
    err = ncmpi_get_vara_float_all(ncid, varids[14], start, count,
				   d->particles); ERR;

    /* neighbors */
    err = ncmpi_inq_varid(ncid, "neighbors", &varids[21]); ERR;
    err = ncmpi_inq_varid(ncid, "num_neighbors", &varids[24]); ERR;
    start[0] = start_block_ofst + b;
    count[0] = 1;
    err = ncmpi_get_vara_int_all(ncid, varids[24], start, count,
				 &((*num_neighbors)[b])); ERR;
    if ((*num_neighbors)[b]) {
      (*neighbors)[b] = (int *)malloc((*num_neighbors)[b] * sizeof(int));
      (*neigh_procs)[b] = (int *)malloc((*num_neighbors)[b] * sizeof(int));
      start[0] = 0;
      count[0] = *tot_blocks;
      err = ncmpi_get_vara_longlong_all(ncid, varids[40], start, count,
					(long long *) block_ofsts); ERR;
      start[0] = block_ofsts[start_block_ofst + b];
      count[0] = (*num_neighbors)[b];
      err = ncmpi_get_vara_int_all(ncid, varids[21], start, count,
				   (*neighbors)[b]); ERR;
    }

    /* neigh_procs is not read from the file; rather distributed anew
       according to the number of processes being used now
       for now relies on blocks being in gid order in file */
    int n;
    for (n = 0; n < (*num_neighbors)[b]; n++) {
      (*neigh_procs)[b][n] = gid2idx[(*neighbors)[b][n]] / bp;
      if ((*neigh_procs)[b][n] >= groupsize)
	(*neigh_procs[b][n] = groupsize - 1);
    }

    /* gid */
    count[0] = 1;
    start[0] = (count[0] ? start_block_ofst + b: 0);
    err = ncmpi_inq_varid(ncid, "gids", &varids[23]); ERR;
    err = ncmpi_get_vara_int_all(ncid, varids[23], start, count, &d->gid); ERR;

    /* tets */
    start[0] = 0;
    count[0] = *tot_blocks;
    err = ncmpi_get_vara_longlong_all(ncid, varids[26], start, count, (long long *) block_ofsts);
    ERR;
    d->tets = (struct tet_t*)malloc(d->num_tets * sizeof(struct tet_t));
    /* tets are written as a 2d array of ints [(verts, tets) x 4 indices]
       the 2 below refers to the two rows: verts and tets
       the 4 indices (2nd dimension) is defined later */
    count[0] = 2 * d->num_tets;
    count[1] = (count[0] ? 4 : 0);
    start[0] = (count[0] ? block_ofsts[start_block_ofst + b] : 0);
    start[1] = 0;
    err = ncmpi_inq_varid(ncid, "tets", &varids[27]); ERR;
    err = ncmpi_get_vara_int_all(ncid, varids[27], start, count,
				 (int *)d->tets); ERR;

    /* remote gids */
    start[0] = 0;
    count[0] = *tot_blocks;
    err = ncmpi_get_vara_longlong_all(ncid, varids[29], start, count,
                                      (long long *) block_ofsts); ERR;
    d->rem_gids = (int*)malloc((d->num_particles - d->num_orig_particles) * sizeof(int));
    count[0] = d->num_particles - d->num_orig_particles;
    start[0] = (count[0] ? block_ofsts[start_block_ofst + b] : 0);
    err = ncmpi_inq_varid(ncid, "rem_gids", &varids[30]); ERR;
    /* it's possible no remote gids exist, and reading them would be an error; test for this */
    int dimid;
    err = ncmpi_inq_vardimid(ncid, varids[30], &dimid); ERR;
    MPI_Offset dim_length;
    err = ncmpi_inq_dimlen(ncid, dimid, &dim_length); ERR;
    if (dim_length) /* only read remote gids if they exist */
      err = ncmpi_get_vara_int_all(ncid, varids[30], start, count, d->rem_gids); ERR;

    /* vert_to_tet */
    start[0] = 0;
    count[0] = *tot_blocks;
    err = ncmpi_get_vara_longlong_all(ncid, varids[9], start, count,
				      (long long *)block_ofsts); ERR;
    d->vert_to_tet =
      (int *)malloc(d->num_particles * sizeof(int));
    start[0] = block_ofsts[start_block_ofst + b];
    count[0] = d->num_particles;
    err = ncmpi_inq_varid(ncid, "vert_to_tet", &varids[33]); ERR;
    err = ncmpi_get_vara_int_all(ncid, varids[33], start, count,
				   d->vert_to_tet); ERR;

  } /* for all blocks */

  /* cleanup */
  err = ncmpi_close(ncid); ERR;
  free(block_ofsts);
  free(all_gids);
  free(gid2idx);

}
/*--------------------------------------------------------------------------*/
