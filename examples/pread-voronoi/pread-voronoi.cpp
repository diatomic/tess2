#include "mpi.h"
#include <assert.h>
#include <vector>
#include <stdio.h>
#include <cmath>

#include "tess/tess.h"
#include "tess/tess.hpp"

#include "pread.h"

#include <diy/mpi.hpp>
#include <diy/master.hpp>
#include <diy/assigner.hpp>
#include <diy/decomposition.hpp>
#include <diy/global.hpp>


using namespace std;

void GetArgs(int argc, char **argv, int &tb, char *infile, char *outfile,
	     std::vector<std::string>& coordinates,
	     float* mins, float* maxs,
	     float *minvol, float *maxvol, int *wrap);

void ExchangeParticles(diy::Master& master, diy::Assigner& assigner, int tot_blocks);
void verify_particles(void* b_, const diy::Master::ProxyWithLink& cp, void*);


struct AddAndRead: public AddBlock
{
	AddAndRead(diy::Master&			    m,
		   int				    nblocks_,
		   char*			    infile_,
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
    read_particles(infile, gid, nblocks, particles, coordinates);
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
  char*					infile;
  const std::vector<std::string>&	coordinates;
};

int main(int argc, char *argv[])
{
  int tot_blocks; // total number of blocks in the domain
  int mem_blocks = -1; // number of blocks to keep in memory
  char infile[256]; // input file name
  char outfile[256]; // output file name
  float mins[3], maxs[3]; // data global extents
  float minvol, maxvol; // volume range, -1.0 = unused
  //double times[TESS_MAX_TIMES]; // timing
  float **particles; // particles[block_num][particle]
		     //  where each particle is 3 values, px, py, pz
  int *num_particles; // number of particles in each block
  int num_threads = 1; // number of threads diy can use
  int dim = 3; // 3d always
  int block_given[3] = {0, 0, 0}; // constraints on blocking (none)
  int wrap_; // whether wraparound neighbors are used
  int rank,size; // MPI usual
  vector <float> p; // temporary particles
  std::vector<std::string>  coordinates; // coordinates to read
  double times[MAX_TIMES]; // timing
  quants_t quants; // quantity stats

  diy::mpi::environment	    env(argc, argv);
  diy::mpi::communicator    world;

  rank = world.rank();
  size = world.size();

  typedef     diy::ContinuousBounds         Bounds;
  Bounds domain;

  GetArgs(argc, argv, tot_blocks, infile, outfile,
          coordinates,
	  domain.min, domain.max,
          &minvol, &maxvol,
	  &wrap_);

  // initialize DIY and decompose domain
  diy::FileStorage          storage("./DIY.XXXXXX");
  diy::Communicator         comm(world);
  diy::Master               master(comm,
                                   &create_block,
                                   &destroy_block,
                                   mem_blocks,
				   num_threads,
                                   &storage,
                                   &save_block,
                                   &load_block);
  //diy::RoundRobinAssigner   assigner(world.size(), tot_blocks);
  diy::ContiguousAssigner   assigner(world.size(), tot_blocks);

  AddAndRead		    create_and_read(master, tot_blocks, infile, coordinates);

  // decompose
  std::vector<int> my_gids;
  assigner.local_gids(comm.rank(), my_gids);
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
  ExchangeParticles(master, assigner, tot_blocks);

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
  master.foreach(&verify_particles);

  tess(master, quants, times);
  tess_save(master, outfile, quants, times);

  return 0;

}
//----------------------------------------------------------------------------
//
// gets command line args
//
void GetArgs(int argc, char **argv, int &tb, char *infile, char *outfile,
	     std::vector<std::string>& coordinates,
	     float* mins, float* maxs,
	     float *minvol, float *maxvol, int *wrap) {

  assert(argc >= 10);

  tb = atoi(argv[1]);
  strcpy(infile, argv[2]);

  if (argv[3][0] =='!')
    strcpy(outfile, "");
  else
    strcpy(outfile, argv[3]);

  coordinates.resize(3);
  coordinates[0] = argv[4];
  coordinates[1] = argv[5];
  coordinates[2] = argv[6];
  mins[0] = atof(argv[7]);
  mins[1] = atof(argv[8]);
  mins[2] = atof(argv[9]);
  maxs[0] = atof(argv[10]);
  maxs[1] = atof(argv[11]);
  maxs[2] = atof(argv[12]);
  *minvol = atof(argv[13]);
  *maxvol = atof(argv[14]);
  *wrap = atoi(argv[15]);

}

void redistribute(void* b_, const diy::SwapReduceProxy& srp, const diy::RegularPartners& partners)
{
    dblock_t*                   b        = static_cast<dblock_t*>(b_);
    unsigned                    round    = srp.round();

    //fprintf(stderr, "in_link.size():  %d\n", srp.in_link().count());
    //fprintf(stderr, "out_link.size(): %d\n", srp.out_link().count());

    // step 1: dequeue and merge
    // dequeue all the incoming points and add them to this block's vector
    // could use srp.incoming() instead
    for (unsigned i = 0; i < srp.in_link().count(); ++i)
    {
      int nbr_gid = srp.in_link().target(i).gid;
      if (nbr_gid == srp.gid())
          continue;

      std::vector<float>    in_points;
      srp.dequeue(nbr_gid, in_points);
      int npts = in_points.size() / 3;

      //fprintf(stderr, "[%d] Received %d points from [%d]\n", srp.gid(), npts, nbr_gid);
      b->particles = (float *)realloc(b->particles, (b->num_particles + npts) * 3 * sizeof(float));
      size_t o = b->num_particles * 3;
      for (size_t j = 0; j < in_points.size(); ++j)
	b->particles[o++] = in_points[j];
      b->num_particles += npts;
    }
    b->num_orig_particles = b->num_particles;

    // step 2: subset and enqueue
    //fprintf(stderr, "[%d] out_link().count(): %d\n", srp.gid(), srp.out_link().count());
    if (srp.out_link().count() == 0)        // final round; nothing needs to be sent
        return;

    std::vector< std::vector<float> > out_points(srp.out_link().count());
    int group_size = srp.out_link().count();
    int cur_dim    = partners.dim(round);
    for (size_t i = 0; i < b->num_particles; ++i)
    {
      int loc = floor((b->particles[3*i + cur_dim] - b->box.min[cur_dim]) / (b->box.max[cur_dim] - b->box.min[cur_dim]) * group_size);
      out_points[loc].push_back(b->particles[3*i]);
      out_points[loc].push_back(b->particles[3*i + 1]);
      out_points[loc].push_back(b->particles[3*i + 2]);
    }
    int pos = -1;
    for (int i = 0; i < group_size; ++i)
    {
      if (srp.out_link().target(i).gid == srp.gid())
      {
	b->particles	 = (float *)realloc(b->particles, out_points[i].size() * sizeof(float));
	for (size_t j = 0; j < out_points[i].size(); ++j)
	  b->particles[j] = out_points[i][j];
	b->num_particles = out_points[i].size() / 3;
	b->num_orig_particles = b->num_particles;
        pos = i;
      }
      else
      {
        srp.enqueue(srp.out_link().target(i), out_points[i]);
        //fprintf(stderr, "[%d] Sent %d points to [%d]\n", srp.gid(), (int) out_points[i].size() / 3, srp.out_link().target(i).gid);
      }
    }
    float new_min = b->box.min[cur_dim] + (b->box.max[cur_dim] - b->box.min[cur_dim])/group_size*pos;
    float new_max = b->box.min[cur_dim] + (b->box.max[cur_dim] - b->box.min[cur_dim])/group_size*(pos + 1);
    b->box.min[cur_dim] = new_min;
    b->box.max[cur_dim] = new_max;
}


//----------------------------------------------------------------------------
//
// sort and distribute particles to all blocks
//
void ExchangeParticles(diy::Master& master, diy::Assigner& assigner, int tot_blocks)
{
  int k = 2;
  diy::RegularPartners  partners(3, tot_blocks, k, false);
  diy::swap_reduce(master, assigner, partners, redistribute);
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
