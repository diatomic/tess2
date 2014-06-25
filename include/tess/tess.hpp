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

typedef     diy::ContinuousBounds       Bounds;

using namespace std;

void create_blocks(int num_blocks, struct dblock_t* &dblocks, int** &hdrs,
		   float **particles, int *num_particles);
void reset_blocks(int num_blocks, struct dblock_t* &dblocks);
void fill_vert_to_tet(dblock_t* dblock);
void incomplete_cells_initial(struct dblock_t *tblock, int lid,
			      vector <set <gb_t> > &sent_particles,
			      vector <int> &convex_hull_particles);
void incomplete_cells_final(struct dblock_t *dblock, int lid,
			    vector <set <gb_t> > &sent_particles,
			    vector <int> &convex_hull_particles);
void wall_particles(struct dblock_t *dblock);
void neighbor_is_complete(int nblocks, struct dblock_t *dblocks,
			  struct remote_ic_t **rics,
			  vector <struct sent_t> *sent_particles);
void sample_particles(float *particles, int &num_particles, int sample_rate);

// callbacks for new diy version
void* create_block();
void destroy_block(void* b);
void save_block(const void* b, diy::BinaryBuffer& bb);
void load_block(void* b, diy::BinaryBuffer& bb);
void create(int gid, const Bounds& core, const Bounds& bounds, const diy::Link& link);
void gen_particles(void* b_, const diy::Master::ProxyWithLink& cp, void*);
void d_delaunay(void* b_, const diy::Master::ProxyWithLink& cp, void*);
void d_incomplete_cells_initial(struct dblock_t *dblock, vector< set<gb_t> > &destinations,
                                vector <int> &convex_hull_particles,
                                const diy::Master::ProxyWithLink& cp);

// add blocks to a master
struct AddBlock
{
  AddBlock(diy::Master& master_):
    master(master_)           {}

  void  operator()(int gid, const Bounds& core, const Bounds& bounds, const diy::Link& link) const
  {
    dblock_t*      b = new dblock_t;
    diy::Link*     l = new diy::Link(link);
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
    b->num_rem_tet_verts = 0;
    b->rem_tet_verts = NULL;
    b->vert_to_tet = NULL;

    // debug
    fprintf(stderr, "creating gid %d mins[%.1f %.1f %.1f] maxs[%.1f %.1f %.1f]\n", b->gid,
            b->mins[0], b->mins[1], b->mins[2], b->maxs[0], b->maxs[1], b->maxs[2]);

  }

  diy::Master&  master;
};

// serialize a remote point
namespace diy
{
  template<>
  struct Serialization<RemotePoint>
  {
    static void save(BinaryBuffer& bb, const RemotePoint& p)
    {
      fprintf(stderr, "2:\n");
      diy::save(bb, p.x);
      diy::save(bb, p.y);
      diy::save(bb, p.z);
      diy::save(bb, p.gid);
      diy::save(bb, p.nid);
      diy::save(bb, p.dir);
    }

    static void load(BinaryBuffer& bb, RemotePoint& p)
    {
      diy::save(bb, p.x);
      diy::save(bb, p.y);
      diy::save(bb, p.z);
      diy::save(bb, p.gid);
      diy::save(bb, p.nid);
      diy::save(bb, p.dir);
    }
  };
}
