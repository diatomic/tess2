// ---------------------------------------------------------------------------
//
//   tess2 header
//
//   Tom Peterka
//   Argonne National Laboratory
//   9700 S. Cass Ave.
//   Argonne, IL 60439
//   tpeterka@mcs.anl.gov
//
// --------------------------------------------------------------------------
#ifndef _TESS_HPP
#define _TESS_HPP

#ifdef BGQ
#include <string.h>
#else
#include <string>
#endif

#include <vector>
#include <set>
#include <limits>

#include <diy/mpi.hpp>
#include <diy/master.hpp>
#include <diy/assigner.hpp>
#include <diy/serialization.hpp>
#include <diy/decomposition.hpp>
#include <diy/pick.hpp>
#include <diy/io/block.hpp>

#ifdef TESS_USE_CGAL
#include "tess-cgal.h"
#endif

#include "tess.h"
#include "delaunay.hpp"

using namespace std;

// quantity stats per process
struct quants_t
{
    int min_quants[MAX_QUANTS];       // min of quantities
    int max_quants[MAX_QUANTS];       // max of quantities
    int sum_quants[MAX_QUANTS];       // sum of quantities
};

typedef diy::RegularContinuousLink  RCLink;
typedef vector<RCLink>              LinkVector;
typedef vector<size_t>              LastNeighbors;

size_t tess(diy::Master& master);
size_t tess(diy::Master& master,
            quants_t& quants,
            double* times);
void tess_exchange(diy::Master& master,
                   const diy::Assigner& assigner);
void tess_exchange(diy::Master& master,
                   const diy::Assigner& assigner,
                   double* times);
void tess_kdtree_exchange(diy::Master& master,
                          const diy::Assigner& assigner,
                          bool wrap,
                          bool sampling = false);
void tess_kdtree_exchange(diy::Master& master,
                          const diy::Assigner& assigner,
                          double* times,
                          bool wrap,
                          bool sampling = false);
void tess_save(diy::Master& master,
               const char* outfile,
               const diy::MemoryBuffer& extra = diy::MemoryBuffer());
void tess_save(diy::Master& master,
               const char* outfile,
               double* times,
               const diy::MemoryBuffer& extra = diy::MemoryBuffer());
void tess_load(diy::Master& master,
               diy::StaticAssigner& assigner,
               const char* infile);
void tess_load(diy::Master& master,
               diy::StaticAssigner& assigner,
               const char* infile,
               diy::MemoryBuffer& extra);
void tess_stats(diy::Master& master,
                quants_t& quants,
                double* times);
void* create_block();
void destroy_block(void* b);
void save_block(const void* b,
                diy::BinaryBuffer& bb);
void load_block(void* b,
                diy::BinaryBuffer& bb);
void save_block_light(const void* b,
                      diy::BinaryBuffer& bb);
void load_block_light(void* b,
                      diy::BinaryBuffer& bb);
void create(int gid,
            const diy::ContinuousBounds& core,
            const diy::ContinuousBounds& bounds,
            const diy::Link& link);
int gen_particles(DBlock* b,
                  float jitter);
void delaunay(DBlock*                           b,
              const diy::Master::ProxyWithLink& cp,
              const LinkVector&                 links,
              LastNeighbors&                    neighbors,
              bool                              first);
void finalize(DBlock*                           b,
              const diy::Master::ProxyWithLink& cp,
              quants_t&                         quants);
void neighbor_particles(DBlock* b,
                        const diy::Master::ProxyWithLink& cp);
size_t incomplete_cells(struct DBlock *dblock,
                        const diy::Master::ProxyWithLink& cp,
                        size_t last_neighbor);
void reset_block(struct DBlock* &dblock);
void fill_vert_to_tet(DBlock* dblock);
void fill_vert_to_tet(dblock_t* dblock);
void wall_particles(struct DBlock *dblock);
void sample_particles(float *particles,
                      int &num_particles,
                      int sample_rate);
void wrap_pt(point_t& rp,
             diy::Direction wrap_dir,
             diy::ContinuousBounds& domain);
int compare(const void *a,
            const void *b);

// add block to a master
// user should not instantiate AddBlock; use AddAndGenerafet or AddEmpty (see below)
struct AddBlock
{
    AddBlock(diy::Master& master_):
        master(master_)           {}

    DBlock* operator()(int gid,
                       const diy::ContinuousBounds& core,
                       const diy::ContinuousBounds& bounds,
                       const diy::ContinuousBounds& domain,
                       const RCLink& link) const
        {
            DBlock*      b = static_cast<DBlock*>(create_block());
            RCLink*        l = new RCLink(link);
            diy::Master&   m = const_cast<diy::Master&>(master);

            m.add(gid, b, l);

            // init block fields
            b->gid = gid;
            b->bounds = core;
            b->data_bounds = domain;
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

            return b;
        }

    diy::Master&  master;
};

// add block to master and generate test particles
struct AddAndGenerate: public AddBlock
{
    AddAndGenerate(diy::Master& master_,
                   float        jitter_):
        AddBlock(master_),
        jitter(jitter_)           {}

    void  operator()(int gid,
                     const diy::ContinuousBounds& core,
                     const diy::ContinuousBounds& bounds,
                     const diy::ContinuousBounds& domain,
                     const RCLink& link) const
        {
            DBlock* b = AddBlock::operator()(gid, core, bounds, domain, link);
            b->num_particles = gen_particles(b, jitter);
            b->num_orig_particles = b->num_particles;
        }

    float jitter;
};

// add empty block to master
struct AddEmpty: public AddBlock
{
    AddEmpty(diy::Master& master_) :
        AddBlock(master_)         {}

    void  operator()(int gid,
                     const diy::ContinuousBounds& core,
                     const diy::ContinuousBounds& bounds,
                     const diy::ContinuousBounds& domain,
                     const RCLink& link) const
        {
            DBlock* b = AddBlock::operator()(gid, core, bounds, domain, link);
            b->num_particles      = 0;
            b->num_orig_particles = 0;
        }
};

// serialize a block
namespace diy
{
    template<>
    struct Serialization<DBlock>
    {
        static void save(BinaryBuffer& bb, const DBlock& d)
            {
                // debug
                //       fprintf(stderr, "Saving block gid %d\n", d.gid);
                diy::save(bb, d.gid);
                diy::save(bb, d.bounds);
                diy::save(bb, d.box);
                diy::save(bb, d.data_bounds);
                diy::save(bb, d.num_orig_particles);
                diy::save(bb, d.num_particles);
                diy::save(bb, d.particles, 3 * d.num_particles);
                diy::save(bb, d.rem_gids, d.num_particles - d.num_orig_particles);
                diy::save(bb, d.rem_lids, d.num_particles - d.num_orig_particles);
                diy::save(bb, d.num_grid_pts);
                diy::save(bb, d.density, d.num_grid_pts);
                // NB tets and vert_to_tet get recreated in each phase; not saved and reloaded

                diy::save(bb, d.complete);

#ifdef TESS_USE_CGAL
                if (!d.complete)
                {
#if 0
                    const Delaunay3D* Dt = static_cast<Delaunay3D*>(d.Dt);
                    diy::save(bb, *Dt);
#endif
                }
                //fprintf(stderr, "Delaunay saved with %lu vertices\n", Dt->number_of_vertices());
#endif

                if (d.complete)
                {
                    diy::save(bb, d.num_tets);
                    diy::save(bb, d.tets, d.num_tets);
                    diy::save(bb, d.vert_to_tet, d.num_particles);
                }

                // debug
                //       fprintf(stderr, "Done saving block gid %d\n", d.gid);
            }

        static void load(BinaryBuffer& bb, DBlock& d)
            {
                diy::load(bb, d.gid);
                // debug
                //       fprintf(stderr, "Loading block gid %d\n", d.gid);
                diy::load(bb, d.bounds);
                diy::load(bb, d.box);
                diy::load(bb, d.data_bounds);
                diy::load(bb, d.num_orig_particles);
                diy::load(bb, d.num_particles);
                d.particles = NULL;
                if (d.num_particles)
                    d.particles = (float*)malloc(d.num_particles * 3 * sizeof(float));
                diy::load(bb, d.particles, 3 * d.num_particles);
                d.rem_gids = NULL;
                d.rem_lids = NULL;
                if (d.num_particles - d.num_orig_particles)
                {
                    d.rem_gids =
                        (int*)malloc((d.num_particles - d.num_orig_particles) * sizeof(int));
                    d.rem_lids =
                        (int*)malloc((d.num_particles - d.num_orig_particles) * sizeof(int));
                }
                diy::load(bb, d.rem_gids, d.num_particles - d.num_orig_particles);
                diy::load(bb, d.rem_lids, d.num_particles - d.num_orig_particles);
                diy::load(bb, d.num_grid_pts);
                d.density = (float*)malloc(d.num_grid_pts * sizeof(float));
                diy::load(bb, d.density, d.num_grid_pts);
                // NB tets and vert_to_tet get recreated in each phase; not saved and reloaded
                d.num_tets = 0;
                d.tets = NULL;
                d.vert_to_tet = NULL;
                if (d.num_particles)
                    d.vert_to_tet = (int*)malloc(d.num_particles * sizeof(int));

                diy::load(bb, d.complete);

#ifdef TESS_USE_CGAL
                if (!d.complete)
                {
#if 0
                    Delaunay3D* Dt = static_cast<Delaunay3D*>(d.Dt);
                    diy::load(bb, *Dt);
                    //fprintf(stderr, "Delaunay loaded with %lu vertices\n", Dt->number_of_vertices());
#endif
                }
#endif

                if (d.complete)
                {
                    diy::load(bb, d.num_tets);
                    d.tets = (tet_t*)malloc(d.num_tets * sizeof(tet_t));
                    diy::load(bb, d.tets, d.num_tets);
                    d.vert_to_tet = NULL;
                    if (d.num_particles)
                        d.vert_to_tet = (int*)malloc(d.num_particles * sizeof(int));
                    diy::load(bb, d.vert_to_tet, d.num_particles);
                }

                // debug
                //       fprintf(stderr, "Done loading block gid %d\n", d.gid);
            }
    };
}

#endif
