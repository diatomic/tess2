#include "mpi.h"
#include <assert.h>
#include <vector>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <algorithm>

#include "tess/tess.h"
#include "tess/tess.hpp"

#include "io/hdf5/pread.h"
#ifdef TESS_GADGET_IO
#include "io/gadget/particles.h"
#endif

#include <diy/mpi.hpp>
#include <diy/master.hpp>
#include <diy/assigner.hpp>
#include <diy/decomposition.hpp>
#include <diy/reduce.hpp>
#include <diy/partners/swap.hpp>

#include "../opts.h"
#include "../memory.h"

#include "common.h"


// This class is coupled with how reading works in io/{hdf5,gadget}
struct ParticleRange
{
            ParticleRange(int size_, size_t total_):
                size(size_), total(total_)      {}

    size_t  from(int rank) const                { if (rank >= size) return total; return total / size * rank; }
    size_t  to(int rank) const                  { if (rank >= size - 1) return total; return from(rank + 1); }
    size_t  count(int rank) const               { return to(rank) - from(rank); }
    int     rank(size_t p) const                { int r = p / (total / size); if (r >= size) r = size - 1; return r; }

    int     size;
    size_t  total;
};

int main(int argc, char *argv[])
{
    string infile; // input file name
    int rank,size; // MPI usual
    std::vector<std::string>  coordinates; // coordinates to read
    double times[TESS_MAX_TIMES]; // timing
    quants_t quants; // quantity stats

    diy::mpi::environment     env(argc, argv);
    diy::mpi::communicator    world;

    rank = world.rank();
    size = world.size();

    typedef     diy::ContinuousBounds         Bounds;
    Bounds domain;

    using namespace opts;

    // defaults
    int min_blocks = size,
        max_blocks = size;
    string prefix = "./DIY.XXXXXX";

    Options ops(argc, argv);

    ops
        >> Option(     "min-blocks",    min_blocks,   "Minimum number of blocks to use")
        >> Option(     "max-blocks",    max_blocks,   "Maximum number of blocks to use")
        ;

    coordinates.resize(3);
    if (  ops >> Present('h', "help", "show help") ||
          !(ops >> PosOption(infile)
            >> PosOption(coordinates[0]) >> PosOption(coordinates[1]) >> PosOption(coordinates[2])
            >> PosOption(domain.min[0])  >> PosOption(domain.min[1])  >> PosOption(domain.min[2])
            >> PosOption(domain.max[0])  >> PosOption(domain.max[1])  >> PosOption(domain.max[2])
              )
        )
    {
        if (rank == 0)
        {
            fprintf(stderr, "Usage: %s [OPTIONS] infile coordinates mins maxs\n", argv[0]);
            std::cout << ops;
        }
        return 1;
    }

    // read or generate points
    std::vector<float>  particles;
        // read points
#if defined TESS_GADGET_IO
    std::string infn(infile);
    if (infn.size() > 7 && infn.substr(0,7) == "gadget:")
    {
        std::string infilename = std::string(infile).substr(7);
        io::gadget::read_particles(world,
                                   infilename.c_str(),
                                   rank,
                                   size,
                                   particles,
                                   coordinates);
    } else      // assume HDF5
#endif
        io::hdf5::read_particles(world,
                                 infile.c_str(),
                                 rank,
                                 size,
                                 particles,
                                 coordinates);

    size_t total_particles;
    diy::mpi::all_reduce(world, particles.size()/3, total_particles, std::plus<size_t>());

    std::vector<float> source_particles;
    source_particles.swap(particles);       // keep a copy of the original for possible redistribution

    if (rank == 0)
        std::cout << "Particles read: " << total_particles << std::endl;

    for (int nblocks = min_blocks; nblocks <= max_blocks; nblocks *= 2)
    {
        world.barrier();
        diy::mpi::communicator working_comm = world;
        if (nblocks < size)
        {
            // redistribute source particles into particles
            std::list<diy::mpi::request> inflight;

            int tag = 0;
            ParticleRange source(size,    total_particles);
            ParticleRange target(nblocks, total_particles);
            particles.resize(target.count(world.rank())*3);

            // post receive requests
            diy::mpi::request r;
            size_t from  = target.from(world.rank());
            size_t i     = from;
            size_t to    = target.to(world.rank());
            particles.resize((to - from)*3);
            //std::cout << "[" << world.rank() << "]: " << "source particles size = " << source_particles.size() << "; target particles size = " << particles.size() << std::endl;
            while (i < to)
            {
                int rk     = source.rank(i);
                int count  = std::min(source.count(rk), to - i);
                //std::cout << "[" << world.rank() << "]: " << "Posting receive at " << (i - from)*3 << " for " << count*3 << " from " << rk << std::endl;
                MPI_Irecv(&particles[(i - from)*3], count*3, MPI_FLOAT, rk, tag, world, &r.r);
                inflight.push_back(r);
                i += count;
            }

            // post send requests
            from = source.from(world.rank());
            i    = from;
            to   = source.to(world.rank());
            while (i < to)
            {
                int rk     = target.rank(i);
                int count  = std::min(target.count(rk), to - i);
                //std::cout << "[" << world.rank() << "]: " << "Posting send at " << (i - from)*3 << " for " << count*3 << " to " << rk << std::endl;
                MPI_Isend(&source_particles[(i - from)*3], count*3, MPI_FLOAT, rk, tag, world, &r.r);
                inflight.push_back(r);
                i += count;
            }

            // kick requests until done
            while(!inflight.empty())
                for (auto it = inflight.begin(); it != inflight.end(); ++it)
                    if (it->test())
                        inflight.erase(it--);

            // restrict the working_comm
            int color = world.rank() < nblocks ? 0 : 1;
            MPI_Comm newcomm;
            MPI_Comm_split(world, color, world.rank(), &newcomm);
            working_comm = newcomm;

            if (world.rank() >= nblocks)
                continue;
        } else
        {
            particles = source_particles;   // just make a copy
        }
        rank = working_comm.rank();

        size_t average = total_particles / nblocks;
        if (rank == 0)
            std::cout << "----\n"
                      << "nblocks=" << nblocks << "; average=" << average << std::endl;

        // initialize DIY and decompose domain
        diy::FileStorage          storage(prefix);
        diy::Master               master(working_comm, 1, -1,
                                         &create_block,
                                         &destroy_block,
                                         &storage,
                                         &save_block,
                                         &load_block);

        diy::ContiguousAssigner   assigner(working_comm.size(), nblocks);

        // decompose
        AddBlock add(master);
        std::vector<int> local_gids;
        assigner.local_gids(rank, local_gids);
        size_t ngids = local_gids.size();
        std::map<int,int> lids;
        for (size_t i = 0; i < local_gids.size(); ++i)
            lids[local_gids[i]] = i;
        auto fill_block = [&add,&particles,&lids](int gid, const Bounds& core, const Bounds& bounds, const Bounds& domain, const RCLink& link)
                          {
                            DBlock* b = add(gid, core, bounds, domain, link);

                            int lid = lids[gid];
                            size_t nparticles = particles.size() / 3;
                            size_t width = nparticles / lids.size();
                            size_t from = lid * width * 3;
                            size_t to   = (lid != lids.size() - 1 ? (lid + 1) * width * 3 - 1 : particles.size() - 1);
                            b->num_particles      = (to - from + 1) / 3;
                            b->num_orig_particles = b->num_particles;
                            b->particles     = (float *)realloc(b->particles, (to - from + 1) * sizeof(float));
                            for (size_t i = from; i <= to; ++i)
                                b->particles[i - from] = particles[i];

                            for (int i = 0; i < 3; ++i)
                            {
                                b->box.min[i] = domain.min[i];
                                b->box.max[i] = domain.max[i];
                            }
                          };

        // regular decomposition
        diy::decompose(3, rank, domain, assigner, fill_block);

        tess_exchange(master, assigner, times);

        // figure out the maxs
        master.foreach([](DBlock* b, const diy::Master::ProxyWithLink& cp)
                       {
                           cp.collectives()->clear();
                           cp.all_reduce(b->num_particles, diy::mpi::maximum<int>());
                       });
        master.exchange();

        int all_max_regular;
        if (rank == 0)
        {
            all_max_regular = master.proxy(0).get<int>();
            std::cout << "Regular: " << all_max_regular << ' '
                                     << float(all_max_regular)/average << ' '
                                     << times[EXCH_TIME]
                                     << std::endl;
        }

        // k-d tree histogram
        master.clear();
        diy::decompose(3, rank, domain, assigner, fill_block);
        tess_kdtree_exchange(master, assigner, times, false);

        // figure out the maxs
        master.foreach([](DBlock* b, const diy::Master::ProxyWithLink& cp)
                       {
                           cp.collectives()->clear();
                           cp.all_reduce(b->num_particles, diy::mpi::maximum<int>());
                       });
        master.exchange();

        int all_max_kdtree_hist;
        if (rank == 0)
        {
            all_max_kdtree_hist = master.proxy(0).get<int>();
            std::cout << "K-d tree (histogram): "
                      << all_max_kdtree_hist << ' '
                      << float(all_max_kdtree_hist)/average << ' '
                      << times[EXCH_TIME]
                      << std::endl;
        }

        // k-d tree sampling
        master.clear();
        diy::decompose(3, rank, domain, assigner, fill_block);
        tess_kdtree_exchange(master, assigner, times, false, true);

        // figure out the maxs
        master.foreach([](DBlock* b, const diy::Master::ProxyWithLink& cp)
                       {
                           cp.collectives()->clear();
                           cp.all_reduce(b->num_particles, diy::mpi::maximum<int>());
                       });
        master.exchange();

        int all_max_kdtree_sample;
        if (rank == 0)
        {
            all_max_kdtree_sample = master.proxy(0).get<int>();
            std::cout << "K-d tree (sampling): "
                      << all_max_kdtree_sample << ' '
                      << float(all_max_kdtree_sample)/average << ' '
                      << times[EXCH_TIME]
                      << std::endl;
        }
    }

    return 0;
}
