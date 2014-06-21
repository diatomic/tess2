//
// driver for tess_test coupled directly to dense
//
// only for newer delaunay format, does not support old voronoi version
//
#include "mpi.h"
#include <assert.h>
#include "tess/tess.h"
#include "tess/dense.hpp"

void GetArgs(int argc, char **argv, int &tb, int *dsize, float *jitter,
	     float *minvol, float *maxvol, int *wrap, int *walls, 
	     char *outfile, int *num_given_bounds,
	     float *given_mins, float *given_maxs, bool &project, 
	     float *proj_plane, float &mass, int *glo_num_idx);

int main(int argc, char *argv[]) {

  int dim = 3;
  int tot_blocks; // total number of blocks in the domain
  int nblocks; // my local number of blocks
  int dsize[3]; // domain grid size
  float jitter; // max amount to randomly displace particles
  float minvol, maxvol; // volume range, -1.0 = unused
  int wrap; // wraparound neighbors flag
  int walls; // apply walls to simulation (wrap must be off)
  char outfile[256]; // output file name
  dblock_t *dblocks; // local blocks
  float eps = 0.0001; // epsilon for floating point values to be equal
  float data_mins[3], data_maxs[3]; // data global bounds
  int given[3] = {0, 0, 0}; // no constraints on decomposition in {x, y, z} 
  int ghost[6] = {0, 0, 0, 0, 0, 0}; // ghost in {-x, +x, -y, +y, -z, +z} 
  MPI_Comm comm = MPI_COMM_WORLD;

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

  MPI_Init(&argc, &argv);

  // timing
  double tess_times[TESS_MAX_TIMES]; // tessllation timing
  double dense_times[DENSE_MAX_TIMES]; // density timing
  double tess_time; // overall tess time
  double dense_time; // overall dense time
  double overall_time; // overall timing (tess + dense)
  for (int i = 0; i < TESS_MAX_TIMES; i++)
    tess_times[i] = 0.0;
  for (int i = 0; i < DENSE_MAX_TIMES; i++)
    dense_times[i] = 0.0;
  MPI_Barrier(comm);
  overall_time = MPI_Wtime();
  dense_times[TESS_TIME] = MPI_Wtime();

  GetArgs(argc, argv, tot_blocks, dsize, &jitter, &minvol, &maxvol, &wrap, 
	  &walls, outfile, &num_given_bounds, given_mins, given_maxs, project, 
	  proj_plane, mass, glo_num_idx);

  // have DIY do the decomposition 
  DIY_Init(dim, 1, comm);
  DIY_Decompose(ROUND_ROBIN_ORDER, dsize, tot_blocks, &nblocks, 1, 
		ghost, given, wrap);

  // tessellate
  MPI_Barrier(comm);
  tess_time = MPI_Wtime();
  dblocks = tess_test_diy_exist(nblocks, dsize, jitter, minvol, maxvol, wrap, 
				walls, tess_times, comm);

  MPI_Barrier(comm);
  tess_time = MPI_Wtime() - tess_time;
  dense_times[TESS_TIME] = MPI_Wtime() - dense_times[TESS_TIME];
  dense_time = MPI_Wtime();
  dense_times[TOTAL_TIME] = MPI_Wtime();
  dense_times[COMP_TIME] = MPI_Wtime();

  // compute the density
  float *density[nblocks];
  dense(density, nblocks, comm, num_given_bounds, given_mins, 
	given_maxs, project, proj_plane, mass, data_mins, data_maxs, dblocks,
	grid_phys_mins, grid_phys_maxs, grid_step_size, eps, glo_num_idx);

  MPI_Barrier(comm);
  dense_times[COMP_TIME] = MPI_Wtime() - dense_times[COMP_TIME];
  dense_times[OUTPUT_TIME] = MPI_Wtime();

  int maxblocks; // max blocks in any process
  MPI_Allreduce(&nblocks, &maxblocks, 1, MPI_INT, MPI_MAX, comm);

  // write file
  MPI_Barrier(comm);
  dense_time = MPI_Wtime() - dense_time;
  dense_times[OUTPUT_TIME] = MPI_Wtime();
  WriteGrid(density, comm, nblocks, maxblocks, outfile, project, glo_num_idx,
	    dblocks, eps, data_mins, data_maxs, num_given_bounds, given_mins,
	    given_maxs);
  MPI_Barrier(comm);
  dense_times[OUTPUT_TIME] = MPI_Wtime() - dense_times[OUTPUT_TIME];
  dense_times[TOTAL_TIME] = MPI_Wtime() - dense_times[TOTAL_TIME];
  overall_time = MPI_Wtime() - overall_time;

  // dense stats
  SummaryStats(dense_times, comm, grid_step_size, grid_phys_mins, glo_num_idx);

  // overall timing
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0)
    fprintf(stderr, "Overall time = %.3lf s = %.3lf s tess + %.3lf s dense\n",
	    overall_time, tess_time, dense_time);

  // cleanup
  for (int i = 0; i < nblocks; i++)
    delete[] density[i];
  destroy_blocks(nblocks, dblocks, NULL);
  DIY_Finalize();
  MPI_Finalize();

  return 0;

}
//----------------------------------------------------------------------------
//
// gets command line args
//
void GetArgs(int argc, char **argv, int &tb, int *dsize, float *jitter,
	     float *minvol, float *maxvol, int *wrap, int *walls, 
	     char *outfile, int *num_given_bounds,
	     float *given_mins, float *given_maxs, bool &project, 
	     float *proj_plane, float &mass, int *glo_num_idx) {

  assert(argc >= 17);

  tb = atoi(argv[1]);
  dsize[0] = atoi(argv[2]);
  dsize[1] = atoi(argv[3]);
  dsize[2] = atoi(argv[4]);
  *jitter = atof(argv[5]);
  *minvol = atof(argv[6]);
  *maxvol = atof(argv[7]);
  *wrap = atoi(argv[8]);
  *walls = atoi(argv[9]);
  if (argv[10][0] =='!')
    strcpy(outfile, "");
  else
    strcpy(outfile, argv[10]);

  glo_num_idx[0] = atoi(argv[11]);
  glo_num_idx[1] = atoi(argv[12]);
  glo_num_idx[2] = atoi(argv[13]);
  if (!strcmp(argv[14], "!")) {
    project = false;
    mass = atof(argv[15]);
    *num_given_bounds = atoi(argv[16]);
    if (*num_given_bounds == 1) {
      given_mins[0] = atof(argv[17]);
      given_maxs[0] = atof(argv[18]);
    }
    else if (*num_given_bounds == 2) {
      given_mins[0] = atof(argv[17]);
      given_mins[1] = atof(argv[18]);
      given_maxs[0] = atof(argv[19]);
      given_maxs[1] = atof(argv[20]);
    }
    else if (*num_given_bounds == 3) {
      given_mins[0] = atof(argv[17]);
      given_mins[1] = atof(argv[18]);
      given_mins[2] = atof(argv[19]);
      given_maxs[0] = atof(argv[20]);
      given_maxs[1] = atof(argv[21]);
      given_maxs[2] = atof(argv[22]);
    }
  }
  if (strcmp(argv[14], "!")) {
    project = true;
    proj_plane[0] = atof(argv[14]);
    proj_plane[1] = atof(argv[15]);
    proj_plane[2] = atof(argv[16]);
    mass = atof(argv[17]);
    *num_given_bounds = atoi(argv[18]);
    if (*num_given_bounds == 1) {
      given_mins[0] = atof(argv[19]);
      given_maxs[0] = atof(argv[20]);
    }
    else if (*num_given_bounds == 2) {
      given_mins[0] = atof(argv[19]);
      given_mins[1] = atof(argv[20]);
      given_maxs[0] = atof(argv[21]);
      given_maxs[1] = atof(argv[22]);
    }
    else if (*num_given_bounds == 3) {
      given_mins[0] = atof(argv[19]);
      given_mins[1] = atof(argv[20]);
      given_mins[2] = atof(argv[21]);
      given_maxs[0] = atof(argv[22]);
      given_maxs[1] = atof(argv[23]);
      given_maxs[2] = atof(argv[24]);
    }
  }

}
//----------------------------------------------------------------------------
