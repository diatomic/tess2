#include "mpi.h"
#include <assert.h>
#include <string.h>
#include "tess/tess.h"
#include "tess/tess.hpp"

void GetArgs(int argc,
             char **argv,
             int &tb,
             int &mb,
             int *dsize,
             float *jitter,
	     float *minvol,
             float *maxvol,
             int *wrap,
             int *walls,
             char *outfile)
{
  assert(argc >= 11);

  tb = atoi(argv[1]);
  mb = atoi(argv[2]);
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
}

int main(int argc, char *argv[])
{
  int tot_blocks;                           // total number of blocks in the domain
  int mem_blocks;                           // max blocks in memory
  int dsize[3];                             // domain grid size
  float jitter;                             // max amount to randomly displace particles
  float minvol, maxvol;                     // volume range, -1.0 = unused
  double times[TESS_MAX_TIMES];             // timing
  int wrap;                                 // wraparound neighbors flag
  int walls;                                // apply walls to simulation (wrap must be off)
  char outfile[256];                        // output file name
  int num_threads = 1;                      // threads diy can use

  // init MPI
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Init(&argc, &argv);

  GetArgs(argc, argv, tot_blocks, mem_blocks, dsize, &jitter, &minvol, &maxvol, &wrap, &walls,
          outfile);

  // data extents
  typedef     diy::ContinuousBounds         Bounds;
  Bounds domain { 3 };
  for(int i = 0; i < 3; i++)
  {
    domain.min[i] = 0;
    domain.max[i] = dsize[i] - 1.0;
  }

  // init diy
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

  // tessellate
  quants_t quants;
  timing(times, -1, -1, world);
  timing(times, TOT_TIME, -1, world);
  tess(master, quants, times);

  // output
  tess_save(master, outfile, times);
  timing(times, -1, TOT_TIME, world);
  tess_stats(master, quants, times);

  MPI_Finalize();

  return 0;
}

