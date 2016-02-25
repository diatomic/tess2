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

#include "io/moab/particles.h"

#include <diy/mpi.hpp>
#include <diy/master.hpp>
#include <diy/assigner.hpp>
#include <diy/decomposition.hpp>
#include <diy/reduce.hpp>
#include <diy/reduce-operations.hpp>
#include <diy/partners/swap.hpp>

#include "../opts.h"
#include "../memory.h"

typedef     std::map<size_t, int>       DuplicateCountMap;

using namespace std;

void verify_particles(void* b_, const diy::Master::ProxyWithLink& cp, void*);

struct UpdateBlock
{
    UpdateBlock(diy::Master&			m) :
        master(m)                               {}

    void operator()(int gid, int lid, const Bounds& core, const Bounds& bounds, const Bounds& domain,
                     const RCLink& link) const
        {
            dblock_t* b = (dblock_t*)master.block(lid);

            // fprintf(stderr, "gid %d lid %d num_particles %d num_orig_particles %d "
            //         "core min [%.3f %.3f %.3f] max[%.3f %.3f %.3f] "
            //         "bounds min[%.3f %.3f %.3f] max[%.3f %.3f %.3f] "
            //         "domain min[%.3f %.3f %.3f] max[%.3f %.3f %.3f]\n",
            //         gid, lid, b->num_particles, b->num_orig_particles,
            //         core.min[0], core.min[1], core.min[2],
            //         core.max[0], core.max[1], core.max[2],
            //         bounds.min[0], bounds.min[1], bounds.min[2],
            //         bounds.max[0], bounds.max[1], bounds.max[2],
            //         domain.min[0], domain.min[1], domain.min[2],
            //         domain.max[0], domain.max[1], domain.max[2]);

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

//
// callback function for global reduction of domain bounds
// when using all-to-all, write the callback as if it is only called once at the beginning
// round and once at the end; diy will take care of the intermediate rounds for you
//
void minmax(void* b_,                                  // local block
            const diy::ReduceProxy& rp)                // communication proxy
{
    dblock_t* b = static_cast<dblock_t*>(b_);

    // step 0: initialize global bounds
    if (!rp.in_link().size() && b->num_particles)      // this block read points
    {
        b->data_bounds.min[0] = b->mins[0];
        b->data_bounds.min[1] = b->mins[1];
        b->data_bounds.min[2] = b->mins[2];
        b->data_bounds.max[0] = b->maxs[0];
        b->data_bounds.max[1] = b->maxs[1];
        b->data_bounds.max[2] = b->maxs[2];
    }
    if (!rp.in_link().size() && !b->num_particles)     // this block did not read points
    {
        b->data_bounds.min[0] = FLT_MAX;
        b->data_bounds.min[1] = FLT_MAX;
        b->data_bounds.min[2] = FLT_MAX;
        b->data_bounds.max[0] = -FLT_MAX;
        b->data_bounds.max[1] = -FLT_MAX;
        b->data_bounds.max[2] = -FLT_MAX;
    }

    // step 1: enqueue
    for (int i = 0; i < rp.out_link().size(); ++i)
        rp.enqueue(rp.out_link().target(i), b->data_bounds);

    // step 2: dequeue
    for (unsigned i = 0; i < rp.in_link().size(); ++i)
    {
        bb_c_t data_bounds;
        rp.dequeue(rp.in_link().target(i).gid, data_bounds);
        if (data_bounds.min[0] < b->data_bounds.min[0])
            b->data_bounds.min[0] = data_bounds.min[0];
        if (data_bounds.min[1] < b->data_bounds.min[1])
            b->data_bounds.min[1] = data_bounds.min[1];
        if (data_bounds.min[2] < b->data_bounds.min[2])
            b->data_bounds.min[2] = data_bounds.min[2];
        if (data_bounds.max[0] > b->data_bounds.max[0])
            b->data_bounds.max[0] = data_bounds.max[0];
        if (data_bounds.max[1] > b->data_bounds.max[1])
            b->data_bounds.max[1] = data_bounds.max[1];
        if (data_bounds.max[2] > b->data_bounds.max[2])
            b->data_bounds.max[2] = data_bounds.max[2];

        // debug
        // fprintf(stderr, "min[%.3f %.3f %.3f] max[%.3f %.3f %.3f]\n",
        //         b->data_bounds.min[0], b->data_bounds.min[1], b->data_bounds.min[2],
        //         b->data_bounds.max[0], b->data_bounds.max[1], b->data_bounds.max[2]);
    }
}

// Structure to remove duplicate points, since downstream code can't handle them.
struct DedupPoint
{
    float data[3];
    bool operator<(const DedupPoint& other) const
        {
            return std::lexicographical_compare(data, data + 3, other.data, other.data + 3);
        }
    bool operator==(const DedupPoint& other) const
        {
            return std::equal(data, data + 3, other.data);
        }
};

// Removes duplicate points
void deduplicate(void* b_, const diy::Master::ProxyWithLink& cp, void* aux)
{
    dblock_t* b = static_cast<dblock_t*>(b_);

    if (!b->num_particles)
        return;

    // simple static_assert to ensure sizeof(Point) == sizeof(float[3]);
    // necessary to make this hack work
    typedef int static_assert_Point_size[sizeof(DedupPoint) == sizeof(float[3]) ? 1 : -1];
    DedupPoint* bg  = (DedupPoint*) &b->particles[0];
    DedupPoint* end = (DedupPoint*) &b->particles[3*b->num_particles];
    std::sort(bg,end);

    DuplicateCountMap* count = (DuplicateCountMap*) aux;
    DedupPoint* out = bg + 1;
    for (DedupPoint* it = bg + 1; it != end; ++it)
    {
        if (*it == *(it - 1))
            (*count)[out - bg - 1]++;
        else
        {
            *out = *it;
            ++out;
        }
    }
    b->num_orig_particles = b->num_particles = out - bg;

    if (!count->empty())
    {
        size_t total = 0;
        for (DuplicateCountMap::const_iterator it = count->begin(); it != count->end(); ++it)
            total += it->second;
        fprintf(stderr, "%d: Found %ld particles that appear more than once, with %ld "
                "total extra copies\n", b->gid, count->size(), total);
    }
}

// check if the particles fall inside the block bounds
void verify_particles(void* b_, const diy::Master::ProxyWithLink& cp, void*)
{
    dblock_t* b = static_cast<dblock_t*>(b_);

    fprintf(stderr, "gid %d num_particles %d num_orig_particles %d "
            "bounds min[%.3f %.3f %.3f] max[%.3f %.3f %.3f] "
            "domain min[%.3f %.3f %.3f] max[%.3f %.3f %.3f]\n",
            b->gid, b->num_particles, b->num_orig_particles,
            b->mins[0], b->mins[1], b->mins[2],
            b->maxs[0], b->maxs[1], b->maxs[2],
            b->data_bounds.min[0], b->data_bounds.min[1], b->data_bounds.min[2],
            b->data_bounds.max[0], b->data_bounds.max[1], b->data_bounds.max[2]);

    for (size_t i = 0; i < b->num_particles; ++i)
    {
        // fprintf(stderr, "Particle: [%f %f %f]\n",
        //         b->particles[3*i],
        //         b->particles[3*i + 1],
        //         b->particles[3*i + 2]);
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

struct Aux
{
    string* infile;
    diy::Assigner* assigner;
};

// read mesh vertices
void
read_vertices(void *b_, const diy::Master::ProxyWithLink& cp, void *aux)
{
    dblock_t* b             = static_cast<dblock_t*>(b_);
    string* infile          = ((Aux*)aux)->infile;
    diy::Assigner* assigner = ((Aux*)aux)->assigner;
    ErrorCode rval; // moab return value

    std::vector<int> my_gids;                // my local gids
    assigner->local_gids(cp.master()->communicator().rank(), my_gids);

    // only read a new moab block once for each mpi rank
    if (b->gid == my_gids[0])
    {
        // load mesh
        Interface *mb       = new Core();
        const char* options = ";;PARALLEL=READ_PART;PARALLEL_RESOLVE_SHARED_ENTS;"
            "PARTITION=MATERIAL_SET;PARTITION_DISTRIBUTE";
        EntityHandle file_set;
        rval = mb->create_meshset(MESHSET_SET, file_set); ERR;
        rval = mb->load_file(infile->c_str(), &file_set, options); ERR;

        // get vertices (0-dimensional entities)
        Range pts;
        rval = mb->get_entities_by_dimension(file_set, 0, pts); ERR;

        b->num_particles      = pts.size();
        b->num_orig_particles = b->num_particles;
        b->particles          = (float *)malloc(pts.size() * 3 * sizeof(float));

        // the point coordinates could be be extracted in one line, except that we
        // need to convert double to float for each coordinate; hence need to loop over the vertices
        // and copy each coordinate separately
        double pt[3];
        size_t i = 0;
        for (Range::iterator it = pts.begin(); it != pts.end(); it++)
        {
            // copy point
            rval = mb->get_coords(&(*it), 1, pt); ERR;
            b->particles[i++] = pt[0];
            b->particles[i++] = pt[1];
            b->particles[i++] = pt[2];

            // extrema
            // eventually the block bounds will get overwritten when we have a proper decomposition
            // for now using them to store local point extrema for the purpose of reducing to
            // global domain bounds
            if (it == pts.begin())
            {
                b->mins[0] = b->maxs[0] = pt[0];
                b->mins[1] = b->maxs[1] = pt[1];
                b->mins[2] = b->maxs[2] = pt[2];
            }
            else
            {
                if (pt[0] < b->mins[0]) b->mins[0] = pt[0];
                if (pt[1] < b->mins[1]) b->mins[1] = pt[1];
                if (pt[2] < b->mins[2]) b->mins[2] = pt[2];
                if (pt[0] > b->maxs[0]) b->maxs[0] = pt[0];
                if (pt[1] > b->maxs[1]) b->maxs[1] = pt[1];
                if (pt[2] > b->maxs[2]) b->maxs[2] = pt[2];
            }
        }

        // debug
        // fprintf(stderr, "min[%.3f %.3f %.3f] max[%.3f %.3f %.3f]\n",
        //         b->mins[0], b->mins[1], b->mins[2], b->maxs[0], b->maxs[1], b->maxs[2]);

        // cleanup
        delete mb;
    }
    else
    {
        b->num_particles = b->num_orig_particles = 0;
        b->particles = NULL;
    }
}

// debug: print the block
void debug(void* b_, const diy::Master::ProxyWithLink& cp, void*)
{
    dblock_t* b = static_cast<dblock_t*>(b_);

    fprintf(stderr, "gid %d num_particles %d num_orig_particles %d num_tets %d complete %d\n"
            "bounds min[%.3f %.3f %.3f] max[%.3f %.3f %.3f]\n"
            "domain min[%.3f %.3f %.3f] max[%.3f %.3f %.3f]\n"
            "box min[%.3f %.3f %.3f] max[%.3f %.3f %.3f]\n",
            b->gid, b->num_particles, b->num_orig_particles, b->num_tets, b->complete,
            b->mins[0], b->mins[1], b->mins[2],
            b->maxs[0], b->maxs[1], b->maxs[2],
            b->data_bounds.min[0], b->data_bounds.min[1], b->data_bounds.min[2],
            b->data_bounds.max[0], b->data_bounds.max[1], b->data_bounds.max[2],
            b->box.min[0], b->box.min[1], b->box.min[2],
            b->box.max[0], b->box.max[1], b->box.max[2]);
}

int main(int argc, char *argv[])
{
    int tot_blocks;                     // total number of blocks in the domain
    int num_threads;                    // number of threads diy can use
    int mem_blocks;                     // number of blocks to keep in memory
    string infile;                      // input file name
    string outfile;                     // output file name
    float minvol, maxvol;               // volume range, -1.0 = unused
    int rank, size;                     // MPI usual
    double times[TESS_MAX_TIMES];       // timing
    quants_t quants;                    // quantity stats

    diy::mpi::environment     env(argc, argv);
    diy::mpi::communicator    world;

    rank = world.rank();
    size = world.size();

    typedef     diy::ContinuousBounds         Bounds;
    Bounds domain;

    using namespace opts;

    // defaults
    tot_blocks    = size;
    num_threads   = 4;
    mem_blocks    = -1;
    string prefix = "./DIY.XXXXXX";
    minvol        = 0;
    maxvol        = 0;

    Options ops(argc, argv);

    ops
        >> Option('b', "blocks",    tot_blocks,   "Total number of blocks to use")
        >> Option('t', "threads",   num_threads,  "Number of threads to use")
        >> Option('m', "in-memory", mem_blocks,   "Number of blocks to keep in memory")
        >> Option('s', "storage",   prefix,       "Path for out-of-core storage")
        >> Option(     "minvol",    minvol,       "minvol cutoff")
        >> Option(     "maxvol",    maxvol,       "minvol cutoff")
        ;
    bool single = ops >> Present('1', "single", "use single-phase version of the algorithm");
    bool kdtree = ops >> Present(     "kdtree", "use kdtree decomposition");

    if ( ops >> Present('h', "help", "show help") ||
         !(ops >> PosOption(infile) >> PosOption(outfile)) )
    {
        if (rank == 0)
        {
            fprintf(stderr, "Usage: %s [OPTIONS] infile outfile minvol bf sr\n", argv[0]);
            std::cout << ops;
        }
        return 1;
    }

    // debug
    // if (rank == 0)
    //     fprintf(stderr, "infile %s outfile %s minv %.1f maxv %.1f "
    //             "th %d mb %d opts %d %d tb %d\n",
    //             infile.c_str(), outfile.c_str(), minvol, maxvol,
    //             num_threads, mem_blocks, single, kdtree, tot_blocks);

    if (kdtree)
    {
        if (!single)
        {
            if (rank == 0)
                fprintf(stderr, "kdtree can only be used with a single-phase version "
                        "of the algorithm\n");
            return 1;
        }

        if (mem_blocks != -1)
        {
            if (rank == 0)
                fprintf(stderr, "kdtree doesn't yet support the out-of-core mode\n");
            return 1;
        }
    }

    if (outfile == "!")
        outfile = "";

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

    // add blocks to the master manually, without a decomposition
    // because I first need to read data into the blocks, reduce it, to get global domain bounds
    // and then domain bounds are needed for decomposition
    std::vector<int> my_gids;                       // my local gids
    assigner.local_gids(rank, my_gids);
    for (unsigned i = 0; i < my_gids.size(); ++i)   // for the local blocks in this processor
    {
        diy::Link* link = new diy::Link;            // link is this block's neighborhood
        dblock_t*  b    = (dblock_t*)create_block();// create a new block
        master.add(my_gids[i], b, link);            // add the current local block to the master
        // init block fields
        b->gid = my_gids[i];
        b->num_orig_particles = 0;
        b->num_particles = 0;
        b->particles = NULL;
        b->num_tets = 0;
        b->tets = NULL;
        b->rem_gids = NULL;
        b->rem_lids = NULL;
        b->vert_to_tet = NULL;
        b->num_grid_pts = 0;
        b->density = NULL;
    }

    // read points
    Aux aux;
    aux.infile   = &infile;
    aux.assigner = &assigner;
    master.foreach(&read_vertices, &aux);

    // reduce global domain bounds
    diy::all_to_all(master, assigner, &minmax);

    // get the domain from any block
    dblock_t*b = (dblock_t*)(master.block(0));
    domain.min[0] = b->data_bounds.min[0];
    domain.min[1] = b->data_bounds.min[1];
    domain.min[2] = b->data_bounds.min[2];
    domain.max[0] = b->data_bounds.max[0];
    domain.max[1] = b->data_bounds.max[1];
    domain.max[2] = b->data_bounds.max[2];

    // debug
    fprintf(stderr, "min[%.3f %.3f %.3f] max[%.3f %.3f %.3f]\n",
            domain.min[0], domain.min[1], domain.min[2],
            domain.max[0], domain.max[1], domain.max[2]);

    // decompose
    UpdateBlock update(master);
    diy::redecompose(3, rank, domain, assigner, master, update);

    // sort and distribute particles to all blocks
    if (kdtree)
        tess_kdtree_exchange(master, assigner, times);
    else
        tess_exchange(master, assigner, times);
    if (rank == 0)
        printf("particles exchanged\n");

    DuplicateCountMap count;
    master.foreach(&deduplicate, &count);

    // debug purposes only: checks if the particles got into the right blocks
    master.foreach(&verify_particles);

    // debug
    // master.foreach(&debug);

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
