// ---------------------------------------------------------------------------
//  
//   functions that have C++ arguments that C source files
//   should not see, hence they are in a separate header
//  
//   Tom Peterka
//   Argonne National Laboratory
//   9700 S. Cass Ave.
//   Argonne, IL 60439
//   tpeterka@mcs.anl.gov
//  
//   (C) 2013 by Argonne National Laboratory.
//   See COPYRIGHT in top-level directory.
//  
// --------------------------------------------------------------------------

#include <vector>
#include <set>

#include <diy/serialization.hpp>
#include <diy/master.hpp>

typedef  diy::ContinuousBounds       Bounds;
typedef  diy::RegularContinuousLink  Link;

using namespace std;

void fill_vert_to_tet(dblock_t* dblock);
void wall_particles(struct dblock_t *dblock);
void neighbor_is_complete(int nblocks, struct dblock_t *dblocks,
			  struct remote_ic_t **rics,
			  vector <struct sent_t> *sent_particles);
void sample_particles(float *particles, int &num_particles, int sample_rate);
diy::Direction nearest_neighbor(float* p, float* mins, float* maxs);

// callbacks for new diy version
void* create_block();
void destroy_block(void* b);
void save_block(const void* b, diy::BinaryBuffer& bb);
void load_block(void* b, diy::BinaryBuffer& bb);
void create(int gid, const Bounds& core, const Bounds& bounds, const diy::Link& link);
void gen_particles(void* b_, const diy::Master::ProxyWithLink& cp, void*);
void delaunay1(void* b_, const diy::Master::ProxyWithLink& cp, void* ps);
void delaunay2(void* b_, const diy::Master::ProxyWithLink& cp, void* ps);
void delaunay3(void* b_, const diy::Master::ProxyWithLink& cp, void* ps);
void neighbor_particles(void* b_, const diy::Master::ProxyWithLink& cp, void*);
void incomplete_cells_initial(struct dblock_t *dblock, vector< set<int> > &destinations,
                              vector <int> &convex_hull_particles,
                              const diy::Master::ProxyWithLink& cp);
void incomplete_cells_final(struct dblock_t *dblock, vector< set<int> > &destinations,
                            vector <int> &convex_hull_particles,
                            const diy::Master::ProxyWithLink& cp);
void reset_block(struct dblock_t* &dblock);

// add blocks to a master
struct AddBlock
{
  AddBlock(diy::Master& master_):
    master(master_)           {}

  void  operator()(int gid, const Bounds& core, const Bounds& bounds, const Link& link) const
  {
    dblock_t*      b = new dblock_t;
    Link*          l = new Link(link);
    diy::Master&   m = const_cast<diy::Master&>(master);

    int lid = m.add(gid, b, l);

    // init block fields
    b->gid = gid;
    b->mins[0] = core.min[0]; b->mins[1] = core.min[1]; b->mins[2] = core.min[2];
    b->maxs[0] = core.max[0]; b->maxs[1] = core.max[1]; b->maxs[2] = core.max[2];
    b->num_orig_particles = 0;
    b->num_particles = 0;
    b->num_tets = 0;
    b->tets = NULL;
    b->vert_to_tet = NULL;

    // debug
//     fprintf(stderr, "creating gid %d mins[%.1f %.1f %.1f] maxs[%.1f %.1f %.1f]\n", b->gid,
//             b->mins[0], b->mins[1], b->mins[2], b->maxs[0], b->maxs[1], b->maxs[2]);

  }

  diy::Master&  master;
};

// serialize a block
// TODO: not serializing Dt for now, recomputing in load() instead
namespace diy
{
  template<>
  struct Serialization<dblock_t>
  {
    static void save(BinaryBuffer& bb, const dblock_t& d)
    {
      diy::save(bb, d.gid);
      diy::save(bb, d.mins);
      diy::save(bb, d.maxs);
      diy::save(bb, d.num_orig_particles);
      diy::save(bb, d.num_particles);
      diy::save(bb, d.particles, d.num_particles);
      diy::save(bb, d.num_tets);
      diy::save(bb, d.tets, d.num_tets);
      diy::save(bb, d.vert_to_tet, d.num_particles);
    }

    static void load(BinaryBuffer& bb, dblock_t& d)
    {
      diy::load(bb, d.gid);
      diy::load(bb, d.mins);
      diy::load(bb, d.maxs);
      diy::load(bb, d.num_orig_particles);
      diy::load(bb, d.num_particles);
      d.particles = (float*)malloc(d.num_particles * 3 * sizeof(float));
      diy::load(bb, d.particles, d.num_particles);
      diy::load(bb, d.num_tets);
      d.tets = (struct tet_t*)malloc(d.num_tets * sizeof(struct tet_t));
      diy::load(bb, d.tets, d.num_tets);
      // DEPRECATED
      //       diy::load(bb, d.num_rem_tet_verts);
      //       d.rem_tet_verts = (struct remote_vert_t*)malloc(d.num_rem_tet_verts * 
      //                                                        sizeof(struct remote_vert_t));
      //       diy::load(bb, d.rem_tet_verts, d.num_rem_tet_verts);
      d.vert_to_tet = (int*)malloc(d.num_particles * sizeof(int));
      diy::load(bb, d.vert_to_tet, d.num_particles);
      // TODO: re-initializing Dt instead of saving/loading it for now
      init_delaunay_data_structure(&d);
    }
  };
}
