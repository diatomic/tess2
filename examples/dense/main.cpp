//---------------------------------------------------------------------------
//
// density field regular grid computation from voronoi tessellation
//
// Tom Peterka
// Argonne National Laboratory
// 9700 S. Cass Ave.
// Argonne, IL 60439
// tpeterka@mcs.anl.gov
//
// (C) 2013 by Argonne National Laboratory.
// See COPYRIGHT in top-level directory.
//
//--------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <vector>
#include "tess/dense.hpp"

using namespace std;

//--------------------------------------------------------------------------
//
// function prototypes
//
void ParseArgs(int argc, char **argv, int *num_given_bounds,
	       float *given_mins, float *given_maxs, bool &project, 
	       float *proj_plane, float &mass, int *glo_num_idx);

//--------------------------------------------------------------------------

int main(int argc, char** argv) {

  int dim = 3;
  int tot_blocks; // global number of blocks
  int nblocks; // my local number of blocks
  int num_threads = 1; // number of threads DIY can use
  int rank, groupsize; // MPI usual
  int did; // domain id
  MPI_Comm comm = MPI_COMM_WORLD; // MPI communicator
  dblock_t *dblocks; // delaunay local blocks
  float eps = 0.0001; // epsilon for floating point values to be equal
  float data_mins[3], data_maxs[3]; // data global bounds

  // grid bounds
  int num_given_bounds; // number of given bounds
  float given_mins[3], given_maxs[3]; // the given bounds
  int glo_num_idx[3]; // global grid number of points
  float grid_phys_mins[3], grid_phys_maxs[3]; // grid physical bounds
  float grid_step_size[3]; // physical size of one grid space

  // 2D projection
  bool project; // whether to project to 2D
  float proj_plane[3]; // normal to projection plane

  // particle mass
  float mass;

  ParseArgs(argc, argv, &num_given_bounds, given_mins, 
	    given_maxs, project, proj_plane, mass, glo_num_idx);

  // ensure projection plane normal vector is unit length
  float length = sqrt(proj_plane[0] * proj_plane[0] + 
		      proj_plane[1] * proj_plane[1] +
		      proj_plane[2] * proj_plane[2]);
  proj_plane[0] /= length;
  proj_plane[1] /= length;
  proj_plane[2] /= length;

  // init
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &groupsize);

  // timing
  double times[DENSE_MAX_TIMES]; // timing
  MPI_Barrier(comm);
  times[TOTAL_TIME] = MPI_Wtime();
  times[INPUT_TIME] = MPI_Wtime();

  // read the tessellation
  // pnetcdf is the only version for the density estimator (no diy version)

  int *gids; // block global ids
  int *num_neighbors; // number of neighbors for each local block
  int **neighbors; // neighbors of each local block
  int **neigh_procs; // processes of neighbors of each local block
  gb_t **diy_neighs; // neighbors in diy global block format

  // read tessellation
  pnetcdf_read(&nblocks, &tot_blocks, &dblocks, argv[1], MPI_COMM_WORLD,
	       &gids, &num_neighbors, &neighbors, &neigh_procs);

  int wrap = 0; // todo: make wrap an input program argument
  bb_t bounds[nblocks]; // block bounds
  for (int i = 0; i < nblocks; i++) {
    for (int j = 0; j < dim; j++) {
      bounds[i].min[j] = dblocks[i].mins[j];
      bounds[i].max[j] = dblocks[i].maxs[j];
    }
  }
  int maxblocks; // max blocks in any process
  MPI_Allreduce(&nblocks, &maxblocks, 1, MPI_INT, MPI_MAX, comm);

  MPI_Barrier(comm);
  times[INPUT_TIME] = MPI_Wtime() - times[INPUT_TIME];
  times[COMP_TIME] = MPI_Wtime();

  // init diy
  diy_neighs = new gb_t*[nblocks];
  for (int i = 0; i < nblocks; i++) {
    if (num_neighbors[i])
      diy_neighs[i] = new gb_t[num_neighbors[i]];
    for (int j = 0; j < num_neighbors[i]; j++) {
      diy_neighs[i][j].gid = neighbors[i][j];
      diy_neighs[i][j].proc = neigh_procs[i][j];
    }
  }
  DIY_Init(dim, num_threads, comm);
  did = DIY_Decomposed(nblocks, gids, bounds, NULL, NULL, NULL, NULL, 
		       diy_neighs, num_neighbors, wrap);

  // cleanup temporary data
  for (int i = 0; i < nblocks; i++) {
    if (num_neighbors[i]) {
      delete[] diy_neighs[i];
      free(neighbors[i]);
      free(neigh_procs[i]);
    }
  }
  delete[] diy_neighs;
  free(neighbors);
  free(neigh_procs);
  free(num_neighbors);
  free(gids);


  // compute the density
  float *density[nblocks];
  dense(density, nblocks, comm, num_given_bounds, given_mins, 
	given_maxs, project, proj_plane, mass, data_mins, data_maxs, dblocks,
	grid_phys_mins, grid_phys_maxs, grid_step_size, eps, glo_num_idx);

  MPI_Barrier(comm);
  times[COMP_TIME] = MPI_Wtime() - times[COMP_TIME];

  // write file
  times[OUTPUT_TIME] = MPI_Wtime();
  WriteGrid(density, comm, nblocks, maxblocks, argv[2], project, glo_num_idx,
	    dblocks, eps, data_mins, data_maxs, num_given_bounds, given_mins,
	    given_maxs);
  MPI_Barrier(comm);
  times[OUTPUT_TIME] = MPI_Wtime() - times[OUTPUT_TIME];

  // cleanup
  for (int i = 0; i < nblocks; i++)
    delete[] density[i];

  MPI_Barrier(comm);
  times[TOTAL_TIME] = MPI_Wtime() - times[TOTAL_TIME];

  SummaryStats(times, comm, grid_step_size, grid_phys_mins, glo_num_idx);

  DIY_Finalize();
  MPI_Finalize();

}
//--------------------------------------------------------------------------
//
// parse args
//
void ParseArgs(int argc, char ** argv, int *num_given_bounds,
	       float *given_mins, float *given_maxs, bool &project, 
	       float *proj_plane, float &mass, int *glo_num_idx) {

  if (argc < 9) {
    fprintf(stderr, "Usage: <exe> <infile> <outfile>"
	    "<resample grid size x y z> <projection plane ! or x y z> <mass> <swap (0 or 1)> <cell-centered densities (0 or 1)>");
    exit(0);
  }

  glo_num_idx[0] = atoi(argv[3]);
  glo_num_idx[1] = atoi(argv[4]);
  glo_num_idx[2] = atoi(argv[5]);
  if (!strcmp(argv[6], "!")) {
    project = false;
    mass = atof(argv[7]);
    *num_given_bounds = atoi(argv[8]);
    if (*num_given_bounds == 1) {
      given_mins[0] = atof(argv[9]);
      given_maxs[0] = atof(argv[10]);
    }
    else if (*num_given_bounds == 2) {
      given_mins[0] = atof(argv[9]);
      given_mins[1] = atof(argv[10]);
      given_maxs[0] = atof(argv[11]);
      given_maxs[1] = atof(argv[12]);
    }
    else if (*num_given_bounds == 3) {
      given_mins[0] = atof(argv[9]);
      given_mins[1] = atof(argv[10]);
      given_mins[2] = atof(argv[11]);
      given_maxs[0] = atof(argv[12]);
      given_maxs[1] = atof(argv[13]);
      given_maxs[2] = atof(argv[14]);
    }
  }
  if (strcmp(argv[6], "!")) {
    project = true;
    proj_plane[0] = atof(argv[6]);
    proj_plane[1] = atof(argv[7]);
    proj_plane[2] = atof(argv[8]);
    mass = atof(argv[9]);
    *num_given_bounds = atoi(argv[10]);
    if (*num_given_bounds == 1) {
      given_mins[0] = atof(argv[11]);
      given_maxs[0] = atof(argv[12]);
    }
    else if (*num_given_bounds == 2) {
      given_mins[0] = atof(argv[11]);
      given_mins[1] = atof(argv[12]);
      given_maxs[0] = atof(argv[13]);
      given_maxs[1] = atof(argv[14]);
    }
    else if (*num_given_bounds == 3) {
      given_mins[0] = atof(argv[11]);
      given_mins[1] = atof(argv[12]);
      given_mins[2] = atof(argv[13]);
      given_maxs[0] = atof(argv[14]);
      given_maxs[1] = atof(argv[15]);
      given_maxs[2] = atof(argv[16]);
    }
  }

}
//--------------------------------------------------------------------------
