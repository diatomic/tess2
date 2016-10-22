#include <diy/master.hpp>
#include "tess/tess.hpp"
#include "tess/tet-neighbors.h"

void print_block_info(void* b_, const diy::Master::ProxyWithLink& cp)
{
  dblock_t* b = static_cast<dblock_t*>(b_);

  printf("%d: num_orig_particles = %d, num_particles = %d, num_tets = %d\n",
         b->gid, b->num_orig_particles, b->num_particles, b->num_tets);
}

void sum_edges(void* b_, const diy::Master::ProxyWithLink& cp)
{
  dblock_t* b = static_cast<dblock_t*>(b_);

  std::vector< std::pair<int, int> > nbrs;
  size_t total_edges = 0;
  for (size_t v = 0; v < b->num_orig_particles; ++v)
  {
    neighbor_edges(nbrs, v, b->tets, b->vert_to_tet[v]);
    total_edges += nbrs.size();
    nbrs.clear();
  }

  cp.all_reduce(total_edges, std::plus<size_t>());
  size_t nparticles = b->num_orig_particles;
  cp.all_reduce(nparticles, std::plus<size_t>());
}


int main(int argc, char** argv) {

  if (argc < 2) {
    fprintf(stderr, "Usage: stats <filename>\n");
    exit(0);
  }

  diy::mpi::environment	    env(argc, argv);
  diy::mpi::communicator    world;

  diy::Master               master(world, -1, -1,
                                   &create_block,
                                   &destroy_block);

  diy::ContiguousAssigner   assigner(world.size(), -1);	    // number of blocks will be set by read_blocks()
  diy::io::read_blocks(argv[1], world, assigner, master, &load_block_light);

  printf("Blocks read: %d\n", master.size());

  master.foreach(print_block_info);

  master.foreach(sum_edges);
  master.exchange();	// process collectives

  if (world.rank() == 0)
  {
    size_t total_edges     = master.proxy(0).get<size_t>();
    size_t total_particles = master.proxy(0).get<size_t>();

    printf("Total edges:  %lu\n", total_edges);
    printf("Total points: %lu\n", total_particles);
  }
}
