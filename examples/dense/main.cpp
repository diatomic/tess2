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
#include "tess/tess.h"
#include "tess/tess.hpp"
#include "tess/dense.hpp"

using namespace std;

void ParseArgs(int argc,
               char ** argv,
               alg &alg_type,
               int *num_given_bounds,
	       float *given_mins,
               float *given_maxs,
               bool &project,
	       float *proj_plane,
               float &mass,
               int *glo_num_idx)
{
  assert(argc >= 10);
  if (atoi(argv[3]) == 0)
    alg_type = DENSE_TESS;
  else
    alg_type = DENSE_CIC;
  glo_num_idx[0] = atoi(argv[4]);
  glo_num_idx[1] = atoi(argv[5]);
  glo_num_idx[2] = atoi(argv[6]);
  if (!strcmp(argv[7], "!"))
  {
    project = false;
    mass = atof(argv[8]);
    *num_given_bounds = atoi(argv[9]);
    if (*num_given_bounds == 1) {
      given_mins[0] = atof(argv[10]);
      given_maxs[0] = atof(argv[11]);
    }
    else if (*num_given_bounds == 2) {
      given_mins[0] = atof(argv[10]);
      given_mins[1] = atof(argv[11]);
      given_maxs[0] = atof(argv[12]);
      given_maxs[1] = atof(argv[13]);
    }
    else if (*num_given_bounds == 3) {
      given_mins[0] = atof(argv[10]);
      given_mins[1] = atof(argv[11]);
      given_mins[2] = atof(argv[12]);
      given_maxs[0] = atof(argv[13]);
      given_maxs[1] = atof(argv[14]);
      given_maxs[2] = atof(argv[15]);
    }
  }
  if (strcmp(argv[7], "!"))
  {
    project = true;
    proj_plane[0] = atof(argv[7]);
    proj_plane[1] = atof(argv[8]);
    proj_plane[2] = atof(argv[9]);
    mass = atof(argv[10]);
    *num_given_bounds = atoi(argv[11]);
    if (*num_given_bounds == 1)
    {
      given_mins[0] = atof(argv[12]);
      given_maxs[0] = atof(argv[13]);
    }
    else if (*num_given_bounds == 2)
    {
      given_mins[0] = atof(argv[12]);
      given_mins[1] = atof(argv[13]);
      given_maxs[0] = atof(argv[14]);
      given_maxs[1] = atof(argv[15]);
    }
    else if (*num_given_bounds == 3)
    {
      given_mins[0] = atof(argv[12]);
      given_mins[1] = atof(argv[13]);
      given_mins[2] = atof(argv[14]);
      given_maxs[0] = atof(argv[15]);
      given_maxs[1] = atof(argv[16]);
      given_maxs[2] = atof(argv[17]);
    }
  }
}

int main(int argc, char** argv)
{
  int tot_blocks;                             // global number of blocks
  int nblocks;                                // my local number of blocks
  int maxblocks;                              // max blocks in any process
  int rank, groupsize;                        // MPI usual
  MPI_Comm comm = MPI_COMM_WORLD;             // MPI communicator
  dblock_t *dblocks;                          // delaunay local blocks
  float eps = 0.0001;                         // epsilon for floating point values to be equal
  float data_mins[3], data_maxs[3];           // data global bounds
  float mass;                                 // particle mass
  alg alg_type;                               // tess or cic

  // grid bounds
  int num_given_bounds;                       // number of given bounds
  float given_mins[3], given_maxs[3];         // the given bounds
  int glo_num_idx[3];                         // global grid number of points
  float grid_phys_mins[3], grid_phys_maxs[3]; // grid physical bounds
  float grid_step_size[3];                    // physical size of one grid space

  // 2D projection
  bool project;                               // whether to project to 2D
  float proj_plane[3];                        // normal to projection plane

  ParseArgs(argc, argv, alg_type, &num_given_bounds, given_mins,given_maxs, project, proj_plane,
            mass, glo_num_idx);

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
  double times[DENSE_MAX_TIMES];               // timing
  MPI_Barrier(comm);
  times[TOTAL_TIME] = MPI_Wtime();
  times[INPUT_TIME] = MPI_Wtime();

  MPI_Barrier(comm);
  times[INPUT_TIME] = MPI_Wtime() - times[INPUT_TIME];
  times[COMP_TIME] = MPI_Wtime();

  // init diy
  int num_threads = 1;
  int mem_blocks = -1;
  diy::mpi::communicator    world(comm);
  diy::FileStorage          storage("./DIY.XXXXXX");
  diy::Master               master(world,
                                   &create_block,
                                   &destroy_block,
                                   mem_blocks,
				   num_threads,
                                   &storage,
                                   &save_block,
                                   &load_block);
  diy::RoundRobinAssigner   assigner(world.size(), -1);  // tot_blocks set by read_blocks

  // read the tessellation
  diy::io::read_blocks(argv[1], world, assigner, master, &load_block_light);

#if 0

  // get global block quantities
  nblocks = master.size();                    // local number of blocks
  MPI_Allreduce(&nblocks, &maxblocks, 1, MPI_INT, MPI_MAX, comm);
  MPI_Allreduce(&nblocks, &maxblocks, 1, MPI_INT, MPI_SUM, comm);

  // compute the density
  dense(alg_type, num_given_bounds, given_mins, given_maxs, project, proj_plane,
        mass, data_mins, data_maxs, grid_phys_mins, grid_phys_maxs, grid_step_size, eps,
        glo_num_idx, master);

  MPI_Barrier(comm);
  times[COMP_TIME] = MPI_Wtime() - times[COMP_TIME];

  // write file
  // NB: all blocks need to be in memory; WriteGrid is not diy2'ed yet
  times[OUTPUT_TIME] = MPI_Wtime();
  WriteGrid(maxblocks, tot_blocks, argv[2], project, glo_num_idx, eps, data_mins, data_maxs,
            num_given_bounds, given_mins, given_maxs, master, assigner);
  MPI_Barrier(comm);
  times[OUTPUT_TIME] = MPI_Wtime() - times[OUTPUT_TIME];


  MPI_Barrier(comm);
  times[TOTAL_TIME] = MPI_Wtime() - times[TOTAL_TIME];

  dense_stats(times, comm, grid_step_size, grid_phys_mins, glo_num_idx);

#endif

  MPI_Finalize();

  return 0;
}
