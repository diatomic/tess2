#include <diy/master.hpp>
#include "tess/tess.hpp"

void print_block_info(void* b_, const diy::Master::ProxyWithLink& cp, void*)
{
  dblock_t* b = static_cast<dblock_t*>(b_);

  printf("%d: num_orig_particles = %d, num_particles = %d, num_tets = %d\n",
         b->gid, b->num_orig_particles, b->num_particles, b->num_tets);
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

  master.foreach(&print_block_info);
}
