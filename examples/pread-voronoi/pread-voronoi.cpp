#include "mpi.h"
#include <assert.h>
#include <vector>
#include <stdio.h>
#include <cmath>
#include <iostream>

#include "tess/tess.h"
#include "tess/tess.hpp"

#include "pread.h"

#include <diy/mpi.hpp>
#include <diy/master.hpp>
#include <diy/assigner.hpp>
#include <diy/decomposition.hpp>
#include <diy/reduce.hpp>
#include <diy/partners/swap.hpp>

#include "../opts.h"
#include "../memory.h"


using namespace std;

void verify_particles(void* b_, const diy::Master::ProxyWithLink& cp, void*);

struct AddAndRead: public AddBlock
{
	AddAndRead(diy::Master&			    m,
		   int				    nblocks_,
		   const char*			    infile_,
		   const std::vector<std::string>&  coordinates_):
	  AddBlock(m),
	  nblocks(nblocks_),
	  infile(infile_),
	  coordinates(coordinates_)		    {}

  void  operator()(int gid, const Bounds& core, const Bounds& bounds, const Bounds& domain,
                   const RCLink& link) const
  {
    dblock_t* b = AddBlock::operator()(gid, core, bounds, domain, link);

    // read points
    std::vector<float>	particles;
    read_particles(master.communicator(), infile, gid, nblocks, particles, coordinates);
    //printf("%d: Read %lu particles\n", gid, particles.size()/3);

    b->num_particles = particles.size()/3;
    b->num_orig_particles = b->num_particles;
    b->particles     = (float *)malloc(particles.size() * sizeof(float));
    for (size_t i = 0; i < particles.size(); ++i)
      b->particles[i] = particles[i];

    for (int i = 0; i < 3; ++i)
    {
      b->box.min[i] = domain.min[i];
      b->box.max[i] = domain.max[i];
    }
  }

  int					nblocks;
  const char*				infile;
  const std::vector<std::string>&	coordinates;
};

int main(int argc, char *argv[])
{
  int tot_blocks; // total number of blocks in the domain
  int num_threads; // number of threads diy can use
  int mem_blocks; // number of blocks to keep in memory
  string infile; // input file name
  string outfile; // output file name
  float mins[3], maxs[3]; // data global extents
  float minvol, maxvol; // volume range, -1.0 = unused
  float **particles; // particles[block_num][particle]
		     //  where each particle is 3 values, px, py, pz
  int *num_particles; // number of particles in each block
  int dim = 3; // 3d always
  int block_given[3] = {0, 0, 0}; // constraints on blocking (none)
  int wrap_; // whether wraparound neighbors are used
  int rank,size; // MPI usual
  vector <float> p; // temporary particles
  std::vector<std::string>  coordinates; // coordinates to read
  double times[TESS_MAX_TIMES]; // timing
  quants_t quants; // quantity stats

  diy::mpi::environment	    env(argc, argv);
  diy::mpi::communicator    world;

  rank = world.rank();
  size = world.size();

  typedef     diy::ContinuousBounds         Bounds;
  Bounds domain;

  using namespace opts;

  // defaults
  tot_blocks  = size;
  num_threads = 1;
  mem_blocks  = -1;
  string prefix = "./DIY.XXXXXX";
  minvol = 0;
  maxvol = 0;

  Options ops(argc, argv);
  ops
    >> Option('b', "blocks",    tot_blocks,   "Total number of blocks to use")
    >> Option('t', "threads",   num_threads,  "Number of threads to use")
    >> Option('m', "in-memory", mem_blocks,   "Number of blocks to keep in memory")
    >> Option('s', "storage",   prefix,       "Path for out-of-core storage")
    >> Option(     "minvol",    minvol,       "minvol cutoff")
    >> Option(     "maxvol",    maxvol,       "minvol cutoff")
  ;
  wrap_ = ops >> Present('w', "wrap", "Use periodic boundary conditions");
  bool single = ops >> Present('1', "single", "use single-phase version of the algorithm");
  bool kdtree = ops >> Present(     "kdtree", "use kdtree decomposition");

  coordinates.resize(3);
  if (  ops >> Present('h', "help", "show help") ||
      !(ops >> PosOption(infile) >> PosOption(outfile)
            >> PosOption(coordinates[0]) >> PosOption(coordinates[1]) >> PosOption(coordinates[2])
            >> PosOption(domain.min[0])  >> PosOption(domain.min[1])  >> PosOption(domain.min[2])
            >> PosOption(domain.max[0])  >> PosOption(domain.max[1])  >> PosOption(domain.max[2])
       )
     )
  {
    if (rank == 0)
    {
      std::cout << "Usage: " << argv[0] << " [OPTIONS] infile outfile coordinates mins maxs\n";
      std::cout << ops;
    }
    return 1;
  }

  if (kdtree)
  {
    if (!single)
    {
      if (rank == 0)
	std::cout << "kdtree can only be used with a single-phase version of the algorithm\n";
      return 1;
    }

    if (mem_blocks != -1)
    {
      if (rank == 0)
	std::cout << "kdtree doesn't yet support the out-of-core mode\n";
      return 1;
    }
  }

  if (outfile == "!")
    outfile = "";
  
  timing(times, -1, -1, world);
  timing(times, TOT_TIME, -1, world);

  // initialize DIY and decompose domain
  diy::FileStorage          storage(prefix);
  diy::Master               master(world,
				   num_threads,
                                   mem_blocks,
                                   &create_block,
                                   &destroy_block,
                                   &storage,
                                   &save_block,
                                   &load_block);
  //diy::RoundRobinAssigner   assigner(world.size(), tot_blocks);
  diy::ContiguousAssigner   assigner(world.size(), tot_blocks);

  AddAndRead		    create_and_read(master, tot_blocks, infile.c_str(), coordinates);

  // decompose
  std::vector<int> my_gids;
  assigner.local_gids(rank, my_gids);
  diy::RegularDecomposer<Bounds>::BoolVector          wrap;
  diy::RegularDecomposer<Bounds>::BoolVector          share_face;
  diy::RegularDecomposer<Bounds>::CoordinateVector    ghosts;
  if (wrap_)
    wrap.assign(3, true);
  diy::decompose(3, rank, domain, assigner, create_and_read, share_face, wrap, ghosts);

#if 0	    // debug
  for (unsigned i = 0; i < master.size(); ++i)
    printf("%d [%d]: %d\n", world.rank(), master.gid(i), master.block<dblock_t>(i)->num_particles);
#endif

  // sort and distribute particles to all blocks
  if (kdtree)
    tess_kdtree_exchange(master, assigner, times, wrap_);
  else
    tess_exchange(master, assigner, times);
  printf("%d: particles exchanged\n", rank);

#if 0	    // debug
  for (unsigned i = 0; i < master.size(); ++i)
  {
    printf("%d: %d\n", i, master.block<dblock_t>(i)->num_particles);
    printf("%d: %f %f %f\n", i,
		master.block<dblock_t>(i)->box.min[0],
		master.block<dblock_t>(i)->box.min[1],
		master.block<dblock_t>(i)->box.min[2]);
    printf("%d: %f %f %f\n", i,
		master.block<dblock_t>(i)->box.max[0],
		master.block<dblock_t>(i)->box.max[1],
		master.block<dblock_t>(i)->box.max[2]);
    printf("%d: %f %f %f\n", i,
		master.block<dblock_t>(i)->mins[0],
		master.block<dblock_t>(i)->mins[1],
		master.block<dblock_t>(i)->mins[2]);
    printf("%d: %f %f %f\n", i,
		master.block<dblock_t>(i)->maxs[0],
		master.block<dblock_t>(i)->maxs[1],
		master.block<dblock_t>(i)->maxs[2]);
  }
#endif

  // debug purposes only: checks if the particles got into the right blocks
  //master.foreach(&verify_particles);

  tess(master, quants, times, single);
  
  tess_save(master, outfile.c_str(), times);

  timing(times, -1, TOT_TIME, world);
  tess_stats(master, quants, times);

  // Storage + memory stats
  size_t max_storage = storage.max_size(),
	 sum_max_storage;
  diy::mpi::reduce(world, max_storage, sum_max_storage, 0, std::plus<size_t>());

  size_t hwm = proc_status_value("VmHWM"),
	 max_hwm;
  diy::mpi::reduce(world, hwm, max_hwm, 0, diy::mpi::maximum<size_t>());

  if (rank == 0)
  {
    fprintf(stderr, "Sum of max storage:  %lu\n", sum_max_storage);
    fprintf(stderr, "Max high water mark: %lu\n", max_hwm);
  }


  return 0;

}


// check if the particles fall inside the block bounds
void verify_particles(void* b_, const diy::Master::ProxyWithLink& cp, void*)
{
  dblock_t* b = static_cast<dblock_t*>(b_);

  for (size_t i = 0; i < b->num_particles; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      if (b->particles[3*i + j] < b->mins[j] || b->particles[3*i + j] > b->maxs[j])
      {
	fprintf(stderr, "Particle outside the block: %f %f %f\n",
		b->particles[3*i],
		b->particles[3*i + 1],
		b->particles[3*i + 2]);
	fprintf(stderr, "    block mins: %f %f %f\n",
		b->mins[0],
		b->mins[1],
		b->mins[2]);
	fprintf(stderr, "    block maxs: %f %f %f\n",
		b->maxs[0],
		b->maxs[1],
		b->maxs[2]);
	std::exit(1);
      }
    }
  }
}
