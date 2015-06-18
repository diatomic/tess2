//
// driver for tess_test coupled directly to dense
//
//
#include "mpi.h"
#include <assert.h>
#include <string.h>
#include "tess/tess.h"
#include "tess/tess.hpp"
#include "tess/dense.hpp"

void GetArgs(int argc,
             char **argv,
             alg& alg_type,
             int &tb,
             int *dsize,
             float *jitter,
	     float *minvol,
             float *maxvol,
             int *wrap,
             int *walls,
	     char *outfile,
             int *num_given_bounds,
	     float *given_mins,
             float *given_maxs,
             bool &project,
	     float *proj_plane,
             float &mass,
             int *glo_num_idx)
{
  assert(argc >= 18);

  if (atoi(argv[1]) == 0)
    alg_type = DENSE_TESS;
  else
    alg_type = DENSE_CIC;
  tb = atoi(argv[2]);
  dsize[0] = atoi(argv[3]);
  dsize[1] = atoi(argv[4]);
  dsize[2] = atoi(argv[5]);
  *jitter = atof(argv[6]);
  *minvol = atof(argv[7]);
  *maxvol = atof(argv[8]);
  *wrap = atoi(argv[9]);
  *walls = atoi(argv[10]);
  if (argv[11][0] =='!')
    strcpy(outfile, "");
  else
    strcpy(outfile, argv[11]);

  glo_num_idx[0] = atoi(argv[12]);
  glo_num_idx[1] = atoi(argv[13]);
  glo_num_idx[2] = atoi(argv[14]);
  if (!strcmp(argv[15], "!"))
  {
    project = false;
    mass = atof(argv[16]);
    *num_given_bounds = atoi(argv[17]);
    if (*num_given_bounds == 1)
    {
      given_mins[0] = atof(argv[18]);
      given_maxs[0] = atof(argv[19]);
    }
    else if (*num_given_bounds == 2)
    {
      given_mins[0] = atof(argv[18]);
      given_mins[1] = atof(argv[19]);
      given_maxs[0] = atof(argv[20]);
      given_maxs[1] = atof(argv[21]);
    }
    else if (*num_given_bounds == 3)
    {
      given_mins[0] = atof(argv[18]);
      given_mins[1] = atof(argv[19]);
      given_mins[2] = atof(argv[20]);
      given_maxs[0] = atof(argv[21]);
      given_maxs[1] = atof(argv[22]);
      given_maxs[2] = atof(argv[23]);
    }
  }
  if (strcmp(argv[15], "!"))
  {
    project = true;
    proj_plane[0] = atof(argv[15]);
    proj_plane[1] = atof(argv[16]);
    proj_plane[2] = atof(argv[17]);
    mass = atof(argv[18]);
    *num_given_bounds = atoi(argv[19]);
    if (*num_given_bounds == 1)
    {
      given_mins[0] = atof(argv[20]);
      given_maxs[0] = atof(argv[21]);
    }
    else if (*num_given_bounds == 2)
    {
      given_mins[0] = atof(argv[20]);
      given_mins[1] = atof(argv[21]);
      given_maxs[0] = atof(argv[22]);
      given_maxs[1] = atof(argv[23]);
    }
    else if (*num_given_bounds == 3)
    {
      given_mins[0] = atof(argv[20]);
      given_mins[1] = atof(argv[21]);
      given_mins[2] = atof(argv[22]);
      given_maxs[0] = atof(argv[23]);
      given_maxs[1] = atof(argv[24]);
      given_maxs[2] = atof(argv[25]);
    }
  }
}

int main(int argc, char *argv[])
{
  int tot_blocks;                             // total number of blocks in the domain
  int nblocks;                                // my local number of blocks
  int dsize[3];                               // domain grid size
  float jitter;                               // max amount to randomly displace particles
  float minvol, maxvol;                       // volume range, -1.0 = unused
  int wrap;                                   // wraparound neighbors flag
  int walls;                                  // apply walls to simulation (wrap must be off)
  char outfile[256];                          // output file name
  float eps = 0.0001;                         // epsilon for floating point values to be equal
  float data_mins[3], data_maxs[3];           // data global bounds
  MPI_Comm comm = MPI_COMM_WORLD;
  alg alg_type;                               // TESS or CIC

  // grid bounds
  int num_given_bounds;                       // number of given bounds
  float given_mins[3], given_maxs[3];         // the given bounds
  int glo_num_idx[3];                         // global grid number of points
  float grid_phys_mins[3], grid_phys_maxs[3]; // grid physical bounds
  float grid_step_size[3];                    // physical size of one grid space

  // 2D projection
  bool project;                               // whether to project to 2D
  float proj_plane[3];                        // normal to projection plane

  // particle mass
  float mass;

  MPI_Init(&argc, &argv);

  // timing
  double tess_times[TESS_MAX_TIMES];         // tessllation timing
  double dense_times[DENSE_MAX_TIMES];       // density timing
  double tess_time;                          // overall tess time
  double dense_time;                         // overall dense time
  double overall_time;                       // overall timing (tess + dense)
  for (int i = 0; i < TESS_MAX_TIMES; i++)
    tess_times[i] = 0.0;
  for (int i = 0; i < DENSE_MAX_TIMES; i++)
    dense_times[i] = 0.0;
  MPI_Barrier(comm);
  overall_time = MPI_Wtime();
  dense_times[TESS_TIME] = MPI_Wtime();

  GetArgs(argc, argv, alg_type, tot_blocks, dsize, &jitter, &minvol, &maxvol, &wrap,
	  &walls, outfile, &num_given_bounds, given_mins, given_maxs, project,
	  proj_plane, mass, glo_num_idx);

  // data extents
  typedef     diy::ContinuousBounds         Bounds;
  Bounds domain;
  for(int i = 0; i < 3; i++)
  {
    domain.min[i] = 0;
    domain.max[i] = dsize[i] - 1.0;
  }

  // init diy
  int num_threads = 1;
  int mem_blocks = -1;
  diy::mpi::communicator    world(comm);
  diy::FileStorage          storage("./DIY.XXXXXX");
  diy::Master               master(world,
				   num_threads,
                                   mem_blocks,
                                   &create_block,
                                   &destroy_block,
                                   &storage,
                                   &save_block,
                                   &load_block);
  diy::RoundRobinAssigner   assigner(world.size(), tot_blocks);
  AddAndGenerate            create(master, jitter);

  // decompose
  std::vector<int> my_gids;
  assigner.local_gids(world.rank(), my_gids);
  diy::RegularDecomposer<Bounds>::BoolVector          wraps;
  diy::RegularDecomposer<Bounds>::BoolVector          share_face;
  diy::RegularDecomposer<Bounds>::CoordinateVector    ghosts;
  if (wrap)
    wraps.assign(3, true);
  diy::decompose(3, world.rank(), domain, assigner, create, share_face, wraps, ghosts);
  nblocks = master.size();

  // tessellate
  MPI_Barrier(comm);
  tess_time = MPI_Wtime();
  quants_t quants;
  timing(tess_times, -1, -1);
  timing(tess_times, TOT_TIME, -1);
  tess(master, quants, tess_times);
  timing(tess_times, -1, TOT_TIME);
  tess_stats(master, quants, tess_times);

  MPI_Barrier(comm);
  tess_time = MPI_Wtime() - tess_time;
  dense_times[TESS_TIME] = MPI_Wtime() - dense_times[TESS_TIME];
  dense_time = MPI_Wtime();
  dense_times[TOTAL_TIME] = MPI_Wtime();
  dense_times[COMP_TIME] = MPI_Wtime();

  // compute the density
  dense(alg_type, num_given_bounds, given_mins, given_maxs, project, proj_plane, mass,
        data_mins, data_maxs, grid_phys_mins, grid_phys_maxs, grid_step_size, eps, glo_num_idx,
        master);
  MPI_Barrier(comm);
  dense_times[COMP_TIME] = MPI_Wtime() - dense_times[COMP_TIME];
  dense_times[OUTPUT_TIME] = MPI_Wtime();

  int maxblocks;                           // max blocks in any process
  MPI_Allreduce(&nblocks, &maxblocks, 1, MPI_INT, MPI_MAX, comm);

  // write file
  // NB: all blocks need to be in memory; WriteGrid is not diy2'ed yet
  MPI_Barrier(comm);
  dense_times[OUTPUT_TIME] = MPI_Wtime();
  WriteGrid(maxblocks, tot_blocks, outfile, project, glo_num_idx, eps, data_mins, data_maxs,
            num_given_bounds, given_mins, given_maxs, master, assigner);
  MPI_Barrier(comm);
  dense_times[OUTPUT_TIME] = MPI_Wtime() - dense_times[OUTPUT_TIME];
  dense_times[TOTAL_TIME] = MPI_Wtime() - dense_times[TOTAL_TIME];
  dense_time = MPI_Wtime() - dense_time;
  overall_time = MPI_Wtime() - overall_time;

  dense_stats(dense_times, master, grid_step_size, grid_phys_mins, glo_num_idx);

  // overall timing
  if (world.rank() == 0)
    fprintf(stderr, "Overall time = %.3lf s = %.3lf s tess + %.3lf s dense\n",
	    overall_time, tess_time, dense_time);

  MPI_Finalize();

  return 0;
}

