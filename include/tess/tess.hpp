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

#include <vector>
#include <set>

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

// quantity stats per process
struct quants_t {
int min_quants[MAX_QUANTS];
int max_quants[MAX_QUANTS];
};

typedef  diy::ContinuousBounds       Bounds;
typedef  diy::RegularContinuousLink  RCLink;

using namespace std;

void tess(diy::Master& master, quants_t& quants, double* times);
void tess_save(diy::Master& master, const char* outfile, double* times);
void tess_stats(diy::Master& master, quants_t& quants, double* times);
void* create_block();
void destroy_block(void* b);
void save_block(const void* b, diy::BinaryBuffer& bb);
void load_block(void* b, diy::BinaryBuffer& bb);
void save_block_light(const void* b, diy::BinaryBuffer& bb);
void load_block_light(void* b, diy::BinaryBuffer& bb);
void create(int gid, const Bounds& core, const Bounds& bounds, const diy::Link& link);
int gen_particles(dblock_t* b, float jitter);
void delaunay1(void* b_, const diy::Master::ProxyWithLink& cp, void* misc_args);
void delaunay2(void* b_, const diy::Master::ProxyWithLink& cp, void*);
void delaunay3(void* b_, const diy::Master::ProxyWithLink& cp, void* misc_args);
void neighbor_particles(void* b_, const diy::Master::ProxyWithLink& cp);
void incomplete_cells_initial(struct dblock_t *dblock, const diy::Master::ProxyWithLink& cp);
void incomplete_cells_final(struct dblock_t *dblock, const diy::Master::ProxyWithLink& cp);
void reset_block(struct dblock_t* &dblock);
void fill_vert_to_tet(dblock_t* dblock);
void wall_particles(struct dblock_t *dblock);
void sample_particles(float *particles, int &num_particles, int sample_rate);
diy::Direction nearest_neighbor(float* p, float* mins, float* maxs);
void wrap_pt(point_t& rp, int wrap_dir, Bounds& domain);
int compare(const void *a, const void *b);

// add block to a master
struct AddBlock
{
  AddBlock(diy::Master& master_):
    master(master_)           {}

  dblock_t*	operator()(int gid, const Bounds& core, const Bounds& bounds, const Bounds& domain,
                   const RCLink& link) const
  {
    dblock_t*      b = static_cast<dblock_t*>(create_block());
    RCLink*        l = new RCLink(link);
    diy::Master&   m = const_cast<diy::Master&>(master);

    int lid = m.add(gid, b, l);

    // init block fields
    b->gid = gid;
    b->mins[0] = core.min[0]; b->mins[1] = core.min[1]; b->mins[2] = core.min[2];
    b->maxs[0] = core.max[0]; b->maxs[1] = core.max[1]; b->maxs[2] = core.max[2];
    b->data_bounds = domain;
    b->num_orig_particles = 0;
    b->num_particles = 0;
    b->particles = NULL;
    b->num_tets = 0;
    b->tets = NULL;
    b->rem_gids = NULL;
    b->vert_to_tet = NULL;
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

  void  operator()(int gid, const Bounds& core, const Bounds& bounds, const Bounds& domain,
                   const RCLink& link) const
  {
    dblock_t* b = AddBlock::operator()(gid, core, bounds, domain, link);
    b->num_particles = gen_particles(b, jitter);
    b->num_orig_particles = b->num_particles;
  }

  float jitter;
};

// serialize a block
namespace diy
{
  template<>
  struct Serialization<dblock_t>
  {
    static void save(BinaryBuffer& bb, const dblock_t& d)
    {
      // debug
//       fprintf(stderr, "Saving block gid %d\n", d.gid);
      diy::save(bb, d.gid);
      diy::save(bb, d.mins);
      diy::save(bb, d.maxs);
      diy::save(bb, d.box);
      diy::save(bb, d.data_bounds);
      diy::save(bb, d.num_orig_particles);
      diy::save(bb, d.num_particles);
      diy::save(bb, d.particles, 3 * d.num_particles);
      diy::save(bb, d.rem_gids, d.num_particles - d.num_orig_particles);
      diy::save(bb, d.num_grid_pts);
      diy::save(bb, d.density, d.num_grid_pts);
      // NB tets and vert_to_tet get recreated in each phase; not saved and reloaded
      vector <int> *convex_hull_particles =
        static_cast<vector <int>*>(d.convex_hull_particles);
      diy::save(bb, *convex_hull_particles);
      vector <set <int> > *sent_particles =
        static_cast<vector <set <int> >*>(d.sent_particles);
      diy::save(bb, *sent_particles);

#ifdef TESS_USE_CGAL
      const Delaunay3D* Dt = static_cast<Delaunay3D*>(d.Dt);
      diy::save(bb, *Dt);
      //fprintf(stderr, "Delaunay saved with %lu vertices\n", Dt->number_of_vertices());
#endif

      // debug
//       fprintf(stderr, "Done saving block gid %d\n", d.gid);
    }

    static void load(BinaryBuffer& bb, dblock_t& d)
    {
      diy::load(bb, d.gid);
      // debug
//       fprintf(stderr, "Loading block gid %d\n", d.gid);
      diy::load(bb, d.mins);
      diy::load(bb, d.maxs);
      diy::load(bb, d.box);
      diy::load(bb, d.data_bounds);
      diy::load(bb, d.num_orig_particles);
      diy::load(bb, d.num_particles);
      d.particles = NULL;
      if (d.num_particles)
        d.particles = (float*)malloc(d.num_particles * 3 * sizeof(float));
      diy::load(bb, d.particles, 3 * d.num_particles);
      d.rem_gids = NULL;
      if (d.num_particles - d.num_orig_particles)
        d.rem_gids = (int*)malloc((d.num_particles - d.num_orig_particles) * sizeof(int));
      diy::load(bb, d.rem_gids, d.num_particles - d.num_orig_particles);
      diy::load(bb, d.num_grid_pts);
      diy::load(bb, d.density, d.num_grid_pts);
      // NB tets and vert_to_tet get recreated in each phase; not saved and reloaded
      d.num_tets = 0;
      d.tets = NULL;
      d.vert_to_tet = NULL;
      if (d.num_particles)
        d.vert_to_tet = (int*)malloc(d.num_particles * sizeof(int));
      diy::load(bb, *(static_cast<vector <int>*>(d.convex_hull_particles)));
      diy::load(bb, *(static_cast<vector <set <int> >*>(d.sent_particles)));

#ifdef TESS_USE_CGAL
      Delaunay3D* Dt = static_cast<Delaunay3D*>(d.Dt);
      diy::load(bb, *Dt);
      //fprintf(stderr, "Delaunay loaded with %lu vertices\n", Dt->number_of_vertices());
#endif

      // debug
//       fprintf(stderr, "Done loading block gid %d\n", d.gid);
    }
  };
}

#endif
