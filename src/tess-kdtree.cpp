#include <vector>
#include <cstdio>

#include <diy/algorithms.hpp>

#include "tess/tess.h"
#include "tess/delaunay.h"

struct KDTreeBlock
{
    struct Point
    {
        float&          operator[](unsigned i)              { return data[i]; }
        const float&    operator[](unsigned i) const        { return data[i]; }
        float           data[3];
    };
    std::vector<Point>                points;
};

struct WrapMaster
{
    diy::Master* master;
    bool         wrap;
};

void populate_kdtree_block(dblock_t*                         d,
                           const diy::Master::ProxyWithLink& cp,
                           diy::Master&                      kdtree_master,
                           bool                              wrap)
{
    diy::ContinuousBounds domain = d->data_bounds;

    KDTreeBlock* b = new KDTreeBlock;
    diy::RegularContinuousLink* l = new diy::RegularContinuousLink(3, domain, domain);
    kdtree_master.add(cp.gid(), b, l);

    // copy the particles over
    b->points.resize(d->num_orig_particles);
    for (size_t i = 0; i < d->num_orig_particles; ++i)
    {
        b->points[i][0] = d->particles[3*i + 0];
        b->points[i][1] = d->particles[3*i + 1];
        b->points[i][2] = d->particles[3*i + 2];
    }
}

void extract_kdtree_block(KDTreeBlock*                      b,
                          const diy::Master::ProxyWithLink& cp,
                          diy::Master&                      tess_master)
{
    int tess_lid = tess_master.lid(cp.gid());
    dblock_t* d  = (dblock_t*) tess_master.block(tess_lid); // assumes all blocks in memory

    // copy out the particles
    d->num_particles = d->num_orig_particles = b->points.size();
    d->particles = (float *)realloc(d->particles, b->points.size() * 3 * sizeof(float));
    for (size_t i = 0; i < d->num_orig_particles; ++i)
    {
        d->particles[3*i + 0] = b->points[i][0];
        d->particles[3*i + 1] = b->points[i][1];
        d->particles[3*i + 2] = b->points[i][2];
    }

    //fprintf(stderr, "[%d]: %d particles copied out\n", cp.gid(), d->num_orig_particles);

    // steal the link
    diy::RegularContinuousLink* tess_link   =
        static_cast<diy::RegularContinuousLink*>(tess_master.link(tess_lid));
    diy::RegularContinuousLink* kdtree_link =
        static_cast<diy::RegularContinuousLink*>(cp.link());
    tess_link->swap(*kdtree_link);

    d->box  = tess_link->bounds();
    for (int i = 0; i < 3; ++i)
    {
        d->mins[i] = tess_link->bounds().min[i];
        d->maxs[i] = tess_link->bounds().max[i];
    }

    delete b; // safe to do since kdtree_master doesn't own the blocks (no create/destroy supplied)
}

void tess_kdtree_exchange(diy::Master& master,
                          const diy::Assigner& assigner,
                          double* times,
                          bool wrap,
                          bool sampling)
{
    timing(times, EXCH_TIME, -1, master.communicator());

    diy::Master kdtree_master(master.communicator(),  master.threads(), -1);
    master.foreach([&](dblock_t* b, const diy::Master::ProxyWithLink& cp)
                   { populate_kdtree_block(b, cp, kdtree_master, wrap); });

    int bins = 1024;      // histogram bins; TODO: make a function argument
    diy::ContinuousBounds domain = master.block<dblock_t>(master.loaded_block())->data_bounds;
    if (sampling)
        diy::kdtree_sampling(kdtree_master, assigner, 3, domain, &KDTreeBlock::points, bins, wrap);
    else
        diy::kdtree(kdtree_master, assigner, 3, domain, &KDTreeBlock::points, bins, wrap);

    kdtree_master.foreach([&](KDTreeBlock* b, const diy::Master::ProxyWithLink& cp)
                          { extract_kdtree_block(b, cp, master); });
    master.set_expected(kdtree_master.expected());

    timing(times, -1, EXCH_TIME, master.communicator());
}

void tess_kdtree_exchange(diy::Master& master,
                          const diy::Assigner& assigner,
                          bool wrap,
                          bool sampling)
{
    double times[TESS_MAX_TIMES];
    tess_kdtree_exchange(master, assigner, times, wrap, sampling);
}
