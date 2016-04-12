#include "mpi.h"
#include <assert.h>
#include <vector>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cfloat>

#include "tess/tess.h"
#include "tess/tess.hpp"

#include <diy/mpi.hpp>
#include <diy/master.hpp>
#include <diy/assigner.hpp>
#include <diy/decomposition.hpp>
#include <diy/reduce.hpp>
#include <diy/reduce-operations.hpp>
#include <diy/partners/swap.hpp>
#include <diy/io/bov.hpp>

#include "../opts.h"
#include "../memory.h"

#include "common.h"

struct UpdateBlock
{
    UpdateBlock(diy::Master&			m) :
        master(m)                               {}

    void operator()(int gid, int lid, const Bounds& core, const Bounds& bounds, const Bounds& domain,
                     const RCLink& link) const
        {
            dblock_t* b = (dblock_t*)master.block(lid);

            for (int i = 0; i < 3; ++i)
            {
                b->mins[i] = core.min[i];
                b->maxs[i] = core.max[i];
                b->box.min[i] = domain.min[i];
                b->box.max[i] = domain.max[i];
                b->data_bounds.min[i] = domain.min[i];
                b->data_bounds.max[i] = domain.max[i];
            }
        }

    diy::Master&                             master;
};

// read mesh vertices
void
read_vertices(void *b_, const diy::Master::ProxyWithLink& cp, void *aux)
{
    dblock_t* b				= static_cast<dblock_t*>(b_);
    const std::vector<float>& values	= *(std::vector<float>*) aux;

    int size = cp.master()->size();
    int lid  = cp.master()->lid(cp.gid());
    size_t npoints = values.size() / 3;
    size_t from = lid * npoints / size * 3;
    size_t to   = lid == size - 1 ? (values.size() - 1) : ((lid + 1) * npoints / size * 3 - 1);

    //fprintf(stderr, "[%d - %d]: reading [%lu,%lu] out of %lu\n", cp.gid(), lid, from, to, values.size());

    size_t nlocal = (to - from + 1)/3;
    b->num_particles      = nlocal;
    b->num_orig_particles = b->num_particles;
    b->particles          = (float *)malloc(nlocal * 3 * sizeof(float));

    float min[3], max[3];
    for (size_t j = 0; j < 3; ++j)
      min[j] = max[j] = values[j];

    for (size_t i = 0; i < (to - from + 1)/3; ++i)
      for (size_t j = 0; j < 3; ++j)
      {
	float x = values[from + i*3 + j];
	if (x < min[j])
	  min[j] = x;
	if (x > max[j])
	  max[j] = x;
	b->particles[i*3 + j] = x;
      }

    cp.all_reduce(min[0], diy::mpi::minimum<float>());
    cp.all_reduce(min[1], diy::mpi::minimum<float>());
    cp.all_reduce(min[2], diy::mpi::minimum<float>());
    cp.all_reduce(max[0], diy::mpi::maximum<float>());
    cp.all_reduce(max[1], diy::mpi::maximum<float>());
    cp.all_reduce(max[2], diy::mpi::maximum<float>());
}

void
fill_bounds(void *b_, const diy::Master::ProxyWithLink& cp, void*)
{
    dblock_t* b = static_cast<dblock_t*>(b_);
    b->data_bounds.min[0] = cp.get<float>() - .0001;
    b->data_bounds.min[1] = cp.get<float>() - .0001;
    b->data_bounds.min[2] = cp.get<float>() - .0001;
    b->data_bounds.max[0] = cp.get<float>() + .0001;
    b->data_bounds.max[1] = cp.get<float>() + .0001;
    b->data_bounds.max[2] = cp.get<float>() + .0001;
}

void
bounds_neighbors(void *b_, const diy::Master::ProxyWithLink& cp, void*)
{
    dblock_t* b = static_cast<dblock_t*>(b_);
    RCLink* link = dynamic_cast<RCLink*>(cp.link());

    fprintf(stderr, "[%d]: %f %f %f - %f %f %f\n",
		    cp.gid(),
		    b->box.min[0], b->box.min[1], b->box.min[2],
		    b->box.max[0], b->box.max[1], b->box.max[2]);

    for (size_t i = 0; i < link->size(); ++i)
    {
      fprintf(stderr, "   %d: %f %f %f - %f %f %f (dir = %d %d %d, wrap = %d %d %d)\n",
		      link->target(i).gid,
		      link->bounds(i).min[0], link->bounds(i).min[1], link->bounds(i).min[2],
		      link->bounds(i).max[0], link->bounds(i).max[1], link->bounds(i).max[2],
		      link->direction(i)[0],  link->direction(i)[1],  link->direction(i)[2],
		      link->wrap(i)[0],	      link->wrap(i)[1],	      link->wrap(i)[2]);
    }
}

int main(int argc, char *argv[])
{
    int tot_blocks;                     // total number of blocks in the domain
    int num_threads;                    // number of threads diy can use
    int mem_blocks;                     // number of blocks to keep in memory
    string infile;                      // input file name
    int rank, size;                     // MPI usual
    double times[TESS_MAX_TIMES];       // timing
    quants_t quants;                    // quantity stats

    diy::mpi::environment     env(argc, argv);
    diy::mpi::communicator    world;

    rank = world.rank();
    size = world.size();

    typedef     diy::ContinuousBounds         Bounds;
    Bounds domain;                           // initialize to [0,1] at first, will get reset
    domain.min[0] = domain.min[1] = domain.min[2] = 0.0;
    domain.max[0] = domain.max[1] = domain.max[2] = 1.0;

    using namespace opts;

    // defaults
    tot_blocks    = size;
    num_threads   = 4;
    mem_blocks    = -1;
    string prefix = "./DIY.XXXXXX";
    int chunk     = 1;

    Options ops(argc, argv);

    ops
        >> Option('b', "blocks",    tot_blocks,   "Total number of blocks to use")
        >> Option('t', "threads",   num_threads,  "Number of threads to use")
        >> Option('m', "in-memory", mem_blocks,   "Number of blocks to keep in memory")
        >> Option('s', "storage",   prefix,       "Path for out-of-core storage")
        >> Option('c', "chunk",     chunk,        "chunk size for writing BOV (for debugging)")
        ;
    bool wrap_  = ops >> Present('w', "wrap",   "Use periodic boundary conditions");
    bool kdtree = ops >> Present(     "kdtree", "use kdtree decomposition");
    bool debug  = ops >> Present('d', "debug",  "print debugging info");
    bool swap   = ops >> Present('s', "swap",   "swap bytes when writing bov file (for debugging)");

    if ( ops >> Present('h', "help", "show help") ||
         !(ops >> PosOption(infile)) )
    {
        if (rank == 0)
        {
            fprintf(stderr, "Usage: %s [OPTIONS] infile outfile minvol bf sr\n", argv[0]);
            std::cout << ops;
        }
        return 1;
    }

    if (kdtree)
    {
        if (mem_blocks != -1)
        {
            if (rank == 0)
                fprintf(stderr, "kdtree doesn't yet support the out-of-core mode\n");
            return 1;
        }

	if (wrap_ && tot_blocks < 64 && rank == 0)
	    fprintf(stderr, "Warning: using k-d tree with wrap on and fewer than 64 blocks is likely to fail\n");
    }

    timing(times, -1, -1, world);
    timing(times, TOT_TIME, -1, world);

    // initialize DIY
    diy::FileStorage          storage(prefix);
    diy::Master               master(world,
                                     num_threads,
                                     mem_blocks,
                                     &create_block,
                                     &destroy_block,
                                     &storage,
                                     &save_block,
                                     &load_block);
    diy::ContiguousAssigner   assigner(world.size(), tot_blocks);

    // decomposing with an uninitialized domain in order to add blocks and links to the master
    // will decompose later with proper domain and block bounds after points have been read
    diy::RegularDecomposer<Bounds>::BoolVector          wrap;
    diy::RegularDecomposer<Bounds>::BoolVector          share_face;
    if (wrap_)
        wrap.assign(3, true);
    AddBlock                  create(master);
    diy::decompose(3, rank, domain, assigner, create);

    // read all the points for this rank
    diy::mpi::io::file in(world, infile.c_str(), diy::mpi::io::file::rdonly);
    size_t sz = in.size() / sizeof(float);
    if (sz % 3 != 0)
    {
      fprintf(stderr, "Something is wrong: number of floats in BOV file is not divisible by 3, "
              "%lu %% 3 != 0\n", sz);
      return 1;
    }
    if (rank == 0)
      fprintf(stderr, "Found %lu points\n", sz / 3);
    std::vector<size_t> shape(1, sz / chunk);
    diy::io::BOV reader(in, shape);
    std::vector<float> values;
    diy::DiscreteBounds box;
    size_t npoints = sz / 3;
    box.min[0] = rank * npoints / size * 3 / chunk;
    if (rank == size - 1)
        box.max[0] = sz / chunk - 1;
    else
        box.max[0] = (rank + 1) * npoints / size * 3 / chunk - 1;
    values.resize((box.max[0] - box.min[0] + 1) * chunk);
    reader.read(box, &values[0], true, chunk);
    if (swap)
        swap_bytes(&values[0], (box.max[0] - box.min[0] + 1) * chunk, sizeof(float));
    if (rank == 0)
        fprintf(stderr, "Values read\n");

    // split points into blocks
    master.foreach(&read_vertices, &values);
    if (rank == 0)
      fprintf(stderr, "Values distributed to blocks\n");
    std::vector<float>().swap(values);	    // empty values and free the memory
    master.exchange();			    // process collectives
    master.foreach(&fill_bounds);
    if (rank == 0)
      fprintf(stderr, "Bounds filled\n");

    // get the domain from any block
    dblock_t* b = (dblock_t*)(master.block(master.loaded_block()));
    domain.min[0] = b->data_bounds.min[0];
    domain.min[1] = b->data_bounds.min[1];
    domain.min[2] = b->data_bounds.min[2];
    domain.max[0] = b->data_bounds.max[0];
    domain.max[1] = b->data_bounds.max[1];
    domain.max[2] = b->data_bounds.max[2];

    // debug
    // fprintf(stderr, "min[%.3f %.3f %.3f] max[%.3f %.3f %.3f]\n",
    //         domain.min[0], domain.min[1], domain.min[2],
    //         domain.max[0], domain.max[1], domain.max[2]);

    // decompose
    UpdateBlock update(master);
    diy::decompose(3, rank, domain, assigner, master, update, share_face, wrap);

    // sort and distribute particles to all blocks
    if (kdtree)
        tess_kdtree_exchange(master, assigner, times, wrap_);
    else
        tess_exchange(master, assigner, times);
    if (rank == 0)
      fprintf(stderr, "particles exchanged\n");

    DuplicateCountMap count;
    master.foreach(&deduplicate, &count);

    if (debug)
    {
      // debug purposes only: checks if the particles got into the right blocks
      master.foreach(&verify_particles);

      // debug
      master.foreach(&bounds_neighbors);
    }

    size_t rounds = tess(master, quants, times);
    if (rank == 0)
      fprintf(stderr, "Done in %lu rounds\n", rounds);

    tess_save(master, "del.out", times);

    timing(times, -1, TOT_TIME, world);
    tess_stats(master, quants, times);
    
    printf("Enumerating cells\n");
    master.foreach(&enumerate_cells);
    master.exchange();

    size_t total_infinite = master.proxy(master.loaded_block()).read<size_t>();
    if (rank == 0)
      fprintf(stderr, "Total infinite cells: %lu\n", total_infinite);

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
