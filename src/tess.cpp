// ---------------------------------------------------------------------------
//  
//   parallel voronoi and delaunay tesselation
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

// MEMORY PROFILING 
// #define MEMORY 

#include "mpi.h"

#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <sys/resource.h>
#include <assert.h>
#include <vector>
#include <set>
#include <algorithm>
#include <cstring>

#include "tess/tess.h"
#include "tess/tess.hpp"
#include "tess/io.h"
#include "tess/tet.h"
#include "tess/tet-neighbors.h"

#include <diy/mpi.hpp>
#include <diy/communicator.hpp>
#include <diy/master.hpp>
#include <diy/assigner.hpp>
#include <diy/serialization.hpp>
#include <diy/decomposition.hpp>
#include <diy/pick.hpp>

#ifdef BGQ
#include <spi/include/kernel/memory.h>
#include "builtins.h"
#define __builtin_ctzll(x) __cnttz8(x)
#define __builtin_clzll(x) __cntlz8(x);
#endif

using namespace std;

static int dim = 3; // everything 3D 
static float data_mins[3], data_maxs[3]; // extents of overall domain 
static float min_vol, max_vol; // cell volume range 
static int nblocks; // number of blocks per process 
static double times[TESS_MAX_TIMES]; // timing info 
static int wrap_neighbors; // whether wraparound neighbors are used 
// CLP - if wrap_neighbors is 0 then check this condition. 
static int walls_on;
static int rank; // my MPI rank
static MPI_Comm comm = MPI_COMM_WORLD; // MPI communicator TODO: get from diy?
static float jitter; // amount to randomly jitter synthetic particles off grid positions

#if 0

// ------------------------------------------------------------------------
//
//   initialize parallel voronoi and delaunay tessellation including
//   initializing DIY with given info on local blocks and their neighbors
//
//   num_blocks: local number of blocks in my process
//   gids: global ids of my local blocks
//   bounds: block bounds (extents) of my local blocks
//   neighbors: neighbor lists for each of my local blocks, in lid order
//   neighbor bounds need not be known, will be discovered automatically
//   num_neighbors: number of neighbors for each of my local blocks, 
//     in lid order
//   global_mins, global_maxs: overall data extents
//   wrap: whether wraparound neighbors are used
//   twalls_on: whether walls boundaries are used
//   minvol, maxvol: filter range for which cells to keep
//   pass -1.0 to skip either or both bounds
//   mpi_comm: MPI communicator
//   all_times: times for particle exchange, voronoi cells, convex hulls, 
//     and output
//
void tess_init(int num_blocks, int *gids, 
	       struct bb_t *bounds, struct gb_t **neighbors, 
	       int *num_neighbors, float *global_mins, float *global_maxs, 
	       int wrap, int twalls_on, float minvol, float maxvol, 
	       MPI_Comm mpi_comm, double *all_times) {

  int i;

  // save globals 
  comm = mpi_comm;
  nblocks = num_blocks;
  min_vol = minvol;
  max_vol = maxvol;
  times = all_times;
  wrap_neighbors = wrap;
  walls_on = twalls_on;

  // data extents 
  for(i = 0; i < 3; i++) {
    data_mins[i] = global_mins[i];
    data_maxs[i] = global_maxs[i];
  }

  // init times 
  for (i = 0; i < TESS_MAX_TIMES; i++)
    times[i] = 0.0;

  // init DIY 
  DIY_Init(dim, 1, comm);
  DIY_Decomposed(num_blocks, gids, bounds, NULL, NULL, NULL, NULL, neighbors, 
		 num_neighbors, wrap);

}
// ------------------------------------------------------------------------
//
//   initialize parallel voronoi and delaunay tessellation with an existing
//   diy domain, assumes DIY_Init and DIY_Decompose done already
//
//   num_blocks: local number of blocks in my process
//   global_mins, global_maxs: overall data extents
//   wrap: whether wraparound neighbors are used
//   twalls_on: whether walls boundaries are used
//   minvol, maxvol: filter range for which cells to keep
//   pass -1.0 to skip either or both bounds
//   mpi_comm: MPI communicator
//   all_times: times for particle exchange, voronoi cells, 
//    convex hulls, and output
void tess_init_diy_exist(int num_blocks, float *global_mins, 
			 float *global_maxs, int wrap, int twalls_on, 
			 float minvol, float maxvol, MPI_Comm mpi_comm, 
			 double *all_times) {

  int i;

  // save globals 
  comm = mpi_comm;
  nblocks = num_blocks;
  min_vol = minvol;
  max_vol = maxvol;
  times = all_times;
  wrap_neighbors = wrap;
  walls_on = twalls_on;

  // data extents 
  for(i = 0; i < 3; i++) {
    data_mins[i] = global_mins[i];
    data_maxs[i] = global_maxs[i];
  }

  // init times 
  for (i = 0; i < TESS_MAX_TIMES; i++)
    times[i] = 0.0;

}
// ------------------------------------------------------------------------
//
// finalize tesselation
//
void tess_finalize() {

  DIY_Finalize();

}
// ------------------------------------------------------------------------
//
//   parallel tessellation
//
//   particles: particles[block_num][particle] 
//   where each particle is 3 values, px, py, pz
//   num_particles; number of particles in each block
//   out_file: output file name
void tess(float **particles, int *num_particles, char *out_file) {

  delaunay(nblocks, particles, num_particles, times, out_file);

}
// --------------------------------------------------------------------------
//
//   test of parallel tesselation with an existing
//   diy domain, assumes DIY_Init and DIY_Decompose done already
//
//   only for newer delaunay format, does not support old voronoi version
//
//   nblocks: number of local blocks
//   data_size: domain grid size (x, y, z)
//   jitter: maximum amount to randomly move each particle
//   minvol, maxvol: filter range for which cells to keep
//   pass -1.0 to skip either or both bounds
//   wrap: whether wraparound neighbors are used
//   twalls_on: whether walls boundaries are used
//   times: times for particle exchange, voronoi cells, convex hulls, and output
//   mpi_comm: MPI communicator
//
//   returns: array of local delaunay blocks
//
struct dblock_t *tess_test_diy_exist(int nblocks, int *data_size, float jitter, 
				     float minvol, float maxvol, int wrap,
				     int twalls_on, double *times,
				     MPI_Comm mpi_comm) {

  float **particles; // particles[block_num][particle] 
		     // where each particle is 3 values, px, py, pz 
  int *num_particles; // number of particles in each block 
  struct dblock_t *ret_blocks; // returned delaunay blocks

  comm = mpi_comm;
  min_vol = minvol;
  max_vol = maxvol;
  wrap_neighbors = wrap;
  walls_on = twalls_on;
  
  // data extents 
  for(int i = 0; i < 3; i++) {
    data_mins[i] = 0.0;
    data_maxs[i] = data_size[i] - 1.0;
  }

  // generate test points in each block 
  particles = (float **)malloc(nblocks * sizeof(float *));
  num_particles = (int *)malloc(nblocks * sizeof(int));
  for (int i = 0; i < nblocks; i++)
    num_particles[i] = gen_particles(i, &particles[i], jitter);

  // compute tessellations 
  ret_blocks = delaunay(nblocks, particles, num_particles, times, (char *)"");

  // cleanup 
  for (int i = 0; i < nblocks; i++)
    free(particles[i]);
  free(particles);
  free(num_particles);

  return ret_blocks;

}
// ------------------------------------------------------------------------

#endif

// ------------------------------------------------------------------------
//
//   test of parallel tesselation
//
//   tot_blocks: total number of blocks in the domain
//   data_size: domain grid size (x, y, z)
//   jitter: maximum amount to randomly move each particle
//   minvol, maxvol: filter range for which cells to keep
//   pass -1.0 to skip either or both bounds
//   wrap_: whether wraparound neighbors are used
//   twalls_on: whether walls boundaries are used
//   times: times for particle exchange, voronoi cells, convex hulls, and output
//   outfile: output file name
//   mpi_comm: MPI communicator
//
void tess_test(int tot_blocks, int *data_size, float jitter, 
	       float minvol, float maxvol, int wrap_, int twalls_on, 
	       double *times, char *outfile, MPI_Comm mpi_comm)
{
  // globals
  ::min_vol = minvol;
  ::max_vol = maxvol;
  ::wrap_neighbors = wrap_;
  ::walls_on = twalls_on;
  ::jitter = jitter;
  
  // data extents 
  typedef     diy::ContinuousBounds         Bounds;
  Bounds domain;
  for(int i = 0; i < 3; i++) {
    // TDOD: remove the global data_mins, data_maxs?
    data_mins[i] = 0.0;
    data_maxs[i] = data_size[i] - 1.0;
    domain.min[i] = 0;
    domain.max[i] = data_size[i] - 1.0;
  }

  // max number of blocks in memory
  // pretend all fit for now; TODO, experiment with fewer
  int max_blocks_mem = tot_blocks;

  // init diy
  diy::mpi::communicator    world(mpi_comm);
  diy::FileStorage          storage("./DIY.XXXXXX");
  diy::Communicator         comm(world);
  diy::Master               master(comm,
                                   &create_block,
                                   &destroy_block,
                                   max_blocks_mem,
                                   &storage,
                                   &save_block,
                                   &load_block);
  diy::RoundRobinAssigner   assigner(world.size(), tot_blocks);
  AddBlock create(master);
  ::rank = world.rank();

  // decompose
  std::vector<int> my_gids;
  assigner.local_gids(comm.rank(), my_gids);
  ::nblocks = my_gids.size();
  diy::RegularDecomposer<Bounds>::BoolVector          wrap;
  diy::RegularDecomposer<Bounds>::BoolVector          share_face;
  diy::RegularDecomposer<Bounds>::CoordinateVector    ghosts;
  diy::decompose(3, ::rank, domain, assigner, create, share_face, wrap, ghosts);

  // particles on the convex hull of the local points and
  // information about particles sent to neighbors
  vector <int> convex_hull_particles;
  vector <set <int> > sent_particles; // sent_particles[particle][i] = ith neighbor (edge)
  void *ps[2]; // pointers to convexh_hull_particles and sent_particles;
  ps[0] = (void*)&convex_hull_particles;
  ps[1] = (void*)&sent_particles;

  // generate particles
  master.foreach(&gen_particles);

  // compute first stage tessellation
  master.foreach(&delaunay1, ps);

  // exchange particles
  master.exchange();

  // parse received particles
  master.foreach(neighbor_particles);

  // compute second stage tessellation
  master.foreach(&delaunay2, ps);

  // exchange particles
  master.exchange();

  // parse received particles
  master.foreach(neighbor_particles);

  // compute third stage tessellation
  master.foreach(&delaunay3, ps);

  // write output 
  // TODO: how to get all my blocks now?
//   if (outfile[0])
//   {
//     char out_ncfile[256];
//     strncpy(out_ncfile, outfile, sizeof(out_ncfile));
//     strncat(out_ncfile, ".nc", sizeof(out_ncfile));
//     pnetcdf_write(nblocks, dblocks, out_ncfile, comm);
//     destroy_block(b);
//   }

  // profile
//   timing(times, -1, OUT_TIME);
//   timing(times, -1, TOT_TIME);
//   get_mem(9, dwell);
 
  // TODO: collect stats 
//   collect_stats(nblocks, dblocks, times);

  // TODO: original version had the option of returning blocks instead of writing to file
  // (for coupling to dense and other tools)
}
//
// diy::Master callback functions
//
void* create_block()
{
  return new dblock_t;
}

void destroy_block(void* b)
{
  delete static_cast<dblock_t*>(b);
}

void save_block(const void* b, diy::BinaryBuffer& bb)
{
  diy::save(bb, *static_cast<const dblock_t*>(b));
}

void load_block(void* b, diy::BinaryBuffer& bb)
{
  diy::load(bb, *static_cast<dblock_t*>(b));
}
//
// foreach block functions
//
void gen_particles(void* b_, const diy::Master::ProxyWithLink& cp, void*)
{
  int sizes[3]; // number of grid points 
  int i, j, k;
  int n = 0;
  int num_particles; // theoretical num particles with duplicates at 
		     // block boundaries 
  float jit; // random jitter amount, 0 - MAX_JITTER 

  dblock_t* b = (dblock_t*)b_;

  // allocate particles 
  sizes[0] = (int)(b->maxs[0] - b->mins[0] + 1);
  sizes[1] = (int)(b->maxs[1] - b->mins[1] + 1);
  sizes[2] = (int)(b->maxs[2] - b->mins[2] + 1);

  num_particles = sizes[0] * sizes[1] * sizes[2];

  b->particles = (float *)malloc(num_particles * 3 * sizeof(float));
  float *p = b->particles;

  // assign particles 
  n = 0;
  for (i = 0; i < sizes[0]; i++) {
    if (b->mins[0] > 0 && i == 0) // dedup block doundary points 
      continue;
    for (j = 0; j < sizes[1]; j++) {
      if (b->mins[1] > 0 && j == 0) // dedup block doundary points 
	continue;
      for (k = 0; k < sizes[2]; k++) {
	if (b->mins[2] > 0 && k == 0) // dedup block doundary points 
	  continue;

	// start with particles on a grid 
	p[3 * n] = b->mins[0] + i;
	p[3 * n + 1] = b->mins[1] + j;
	p[3 * n + 2] = b->mins[2] + k;

	// and now jitter them 
	jit = rand() / (float)RAND_MAX * 2 * jitter - jitter;
	if (p[3 * n] - jit >= b->mins[0] &&
	    p[3 * n] - jit <= b->maxs[0])
	  p[3 * n] -= jit;
	else if (p[3 * n] + jit >= b->mins[0] &&
		 p[3 * n] + jit <= b->maxs[0])
	  p[3 * n] += jit;

	jit = rand() / (float)RAND_MAX * 2 * jitter - jitter;
	if (p[3 * n + 1] - jit >= b->mins[1] &&
	    p[3 * n + 1] - jit <= b->maxs[1])
	  p[3 * n + 1] -= jit;
	else if (p[3 * n + 1] + jit >= b->mins[1] &&
		 p[3 * n + 1] + jit <= b->maxs[1])
	  p[3 * n + 1] += jit;

	jit = rand() / (float)RAND_MAX * 2 * jitter - jitter;
	if (p[3 * n + 2] - jit >= b->mins[2] &&
	    p[3 * n + 2] - jit <= b->maxs[2])
	  p[3 * n + 2] -= jit;
	else if (p[3 * n + 2] + jit >= b->mins[2] &&
		 p[3 * n + 2] + jit <= b->maxs[2])
	  p[3 * n + 2] += jit;
  
	n++;

      }

    }

  }
  b->num_particles = n; // final count <= amount originally allocated
  b->num_orig_particles = n;

  // debug
  fprintf(stderr, "generating %d (actual %d) particles in gid %d\n", num_particles, b->num_particles, b->gid);
}

void delaunay1(void* b_, const diy::Master::ProxyWithLink& cp, void* ps)
{
  dblock_t* b = (dblock_t*)b_;

  init_delaunay_data_structure(b);

  // init timing 
  for (int i = 0; i < TESS_MAX_TIMES; i++)
    times[i] = 0.0;
  timing(times, TOT_TIME, -1);

  // profile
  int dwell = 10;
  get_mem(1, dwell);
  timing(times, LOC1_TIME, -1);

  // create local delaunay cells
  local_cells(b);
  
  // profile
  get_mem(2, dwell);
  timing(times, INC1_TIME, LOC1_TIME);

  // particles on the convex hull of the local points and
  // information about particles sent to neighbors
  // sent_particles[particle][i] = ith neighbor (edge)
  vector <int>* convex_hull_particles = static_cast<vector <int>*>( ((void**)ps)[0] );
  vector <set <int> >* sent_particles = static_cast<vector <set <int> >*>( ((void**)ps)[1] ); 

  // determine which cells are incomplete or too close to neighbor 
  incomplete_cells_initial(b, *sent_particles, *convex_hull_particles, cp);

  // profile
  get_mem(3, dwell);
  timing(times, NEIGH1_TIME, INC1_TIME);

  // cleanup block
  reset_block(b);

}

void delaunay2(void* b_, const diy::Master::ProxyWithLink& cp, void* ps)
{
  dblock_t* b = (dblock_t*)b_;

  // particles on the convex hull of the local points and
  // information about particles sent to neighbors
  // sent_particles[particle][i] = ith neighbor (edge)
  vector <int>* convex_hull_particles = static_cast<vector <int>*>( ((void**)ps)[0] );
  vector <set <int> >* sent_particles = static_cast<vector <set <int> >*>( ((void**)ps)[1] ); 

  // profile
  int dwell = 10;
  get_mem(4, dwell);
  timing(times, LOC2_TIME, NEIGH1_TIME);
  
#ifdef DEBUG
  int max_particles;
  MPI_Reduce(&b->num_particles, &max_particles, 1, MPI_INT, MPI_MAX, 0, comm);
  if (rank == 0)
    fprintf(stderr, "phase 1: max_particles = %d\n", max_particles);
#endif

  // recompute local cells
  local_cells(b);

  // profile
  get_mem(5, dwell);
  timing(times, INC2_TIME, LOC2_TIME);

  for (int i = 0; i < nblocks; i++)
    incomplete_cells_final(b, *sent_particles, *convex_hull_particles, cp);

  // cleanup sent particles
  sent_particles->clear();
  
  // profile
  get_mem(6, dwell);
  timing(times, NEIGH2_TIME, INC2_TIME);
  
  // TODO, turn walls back on
  // generate particles to create wall
//   if (!wrap_neighbors && walls_on)
//     for (int i = 0; i < nblocks; i++)
//         wall_particles(&dblocks[i]);

  // cleanup block
  reset_block(b);
}

void delaunay3(void* b_, const diy::Master::ProxyWithLink& cp, void* ps)
{
  dblock_t* b = (dblock_t*)b_;
  // particles on the convex hull of the local points
  vector <int>* convex_hull_particles = static_cast<vector <int>*>( ((void**)ps)[0] );

  // profile
  int dwell = 10;
  get_mem(7, dwell);
  timing(times, LOC3_TIME, NEIGH2_TIME);
  
  // cleanup convex hull particles
  convex_hull_particles->clear();

  // create all final cells 
  local_cells(b);
  
  // profile
  get_mem(8, dwell);
  timing(times, OUT_TIME, LOC3_TIME);

  // cleanup delaunay data structure and sent particles
  clean_delaunay_data_structure(b);
}
//
// incomplete cells functions
//
void incomplete_cells_initial(struct dblock_t *dblock, vector< set<int> > &destinations,
                              vector <int> &convex_hull_particles,
                              const diy::Master::ProxyWithLink& cp)
{
  struct RemotePoint rp; // particle being sent or received 
  destinations.resize(dblock->num_orig_particles);

  // link
  Link* l = dynamic_cast<Link*>(cp.link());

  // identify and enqueue convex hull particles
  for (int p = 0; p < dblock->num_orig_particles; ++p)
  {
    if (dblock->vert_to_tet[p] == -1)
    {
      fprintf(stderr, "Particle %d is not in the triangulation. "
              "Perhaps it's a duplicate? Aborting.\n", p);
      assert(false);
    }

    // on convex hull = less than 4 neighbors
    if (dblock->num_tets == 0 || 
    	!complete(p, dblock->tets, dblock->num_tets, dblock->vert_to_tet[p]))
    {
      // add to list of convex hull particles
      convex_hull_particles.push_back(p);

      // incomplete cell goes to the closest neighbor 
      diy::Direction nearest_dir = 
    	nearest_neighbor(&(dblock->particles[3 * p]), dblock->mins, dblock->maxs);
      // TODO: helper functions will be moved to standalone
      if (l->direction(nearest_dir) != -1)
	destinations[p].insert(l->direction(nearest_dir));
    }

  }

  // for all tets
  for (int t = 0; t < dblock->num_tets; t++)
  {
    // cirumcenter of tet and radius from circumcenter to any vertex
    float center[3]; // circumcenter
    circumcenter(center, &dblock->tets[t], dblock->particles);
    int p = dblock->tets[t].verts[0];
    float rad = distance(center, &dblock->particles[3 * p]);

    // find nearby blocks within radius of circumcenter
    set<int> dests; // destination neighbor edges for this point
    for (unsigned i = 0; i < l->count(); ++i)
    {
      near(*l, center, rad, std::inserter(dests, dests.end()));
    }

    // all 4 verts go these dests
    for (int v = 0; v < 4; v++)
    {
      int p = dblock->tets[t].verts[v];
      // destinations is a set; neighbors will be unique
      for (set<int>::iterator it = dests.begin(); it != dests.end(); it++)
      {
        destinations[p].insert(*it);
      }
    }
  }

  // enqueue the particles
  for (int p = 0; p < dblock->num_orig_particles; p++)
  {
    rp.x   = dblock->particles[3 * p];
    rp.y   = dblock->particles[3 * p + 1];
    rp.z   = dblock->particles[3 * p + 2];
    rp.gid = dblock->gid;
    rp.nid = p;
    rp.dir = 0x00;
    for (set<int>::iterator it = destinations[p].begin(); it != destinations[p].end(); it++)
    {
      // debug
//       fprintf(stderr, "gid %d sending %.1f %.1f %.1f to gid %d\n", 
//               dblock->gid, rp.x, rp.y, rp.z, cp.link()->target(*it).gid);
      cp.enqueue(cp.link()->target(*it), rp);
    }
  }                     

}

void incomplete_cells_final(struct dblock_t *dblock, vector< set<int> > &destinations,
                            vector <int> &convex_hull_particles,
                            const diy::Master::ProxyWithLink& cp)
{
  struct RemotePoint rp; // particle being sent or received 
  diy::BoundsLink<Bounds>* l = dynamic_cast<diy::BoundsLink<Bounds>*>(cp.link()); // link
  set<int> new_dests; // new destination neighbor edges for sending this point

  // for all convex hull particles
  for (int j = 0; j < (int)convex_hull_particles.size(); ++j)
  {

    int p = convex_hull_particles[j];

    if (dblock->vert_to_tet[p] == -1)
    {
      fprintf(stderr, "Particle %d is not in the triangulation. "
	      "Perhaps it's a duplicate? Aborting.\n", p);
      assert(false);
    }

    std::vector<int> nbrs;
    bool complete = neighbor_tets(nbrs, p, dblock->tets, 
                                  dblock->num_tets,
    				  dblock->vert_to_tet[p]);

    if (!complete)
    {
      // local point still on the convex hull goes to everybody it hasn't gone to yet 
      for (int n = 0; n < l->count(); n++) // all neighbors
      {
      	if (destinations[p].find(n) == destinations[p].end())
        {
          new_dests.insert(n);
      	}
      }    
    } // !complete

    else // complete
    {
      // point not on convex hull anymore goes to all neighbbors neear enough provided it hasn't
      // gone to them already
      for (int j = 0; j < (int)nbrs.size(); ++j)
      {
    	int t = nbrs[j];
    	float center[3];
    	circumcenter(center, &dblock->tets[t], dblock->particles);

    	// radius is distance from circumcenter to any tet vertex
    	int p0 = dblock->tets[t].verts[0];
    	float rad = distance(center, &dblock->particles[3 * p0]);

        // find nearby blocks within radius of circumcenter
        set<int> near_candts; // candidate destination neighbor edges for this point
        for (unsigned i = 0; i < l->count(); ++i)
        {
          near(*l, center, rad, std::inserter(near_candts, near_candts.end()));
        }

    	// remove the nearby neighbors we've already sent to
        for (set<int>::iterator it = near_candts.begin(); it != near_candts.end(); it++)
        {
    	  if (destinations[p].find(*it) == destinations[p].end())
          {
    	    new_dests.insert(*it);
    	  }
    	}
      } // nbrs
    } // complete 

    // enquue the particle to the new destinations
    if (new_dests.size())
    {
      rp.x = dblock->particles[3 * p];
      rp.y = dblock->particles[3 * p + 1];
      rp.z = dblock->particles[3 * p + 2];
      rp.gid = dblock->gid;
      rp.nid = p;
      rp.dir = 0x00;
      for (set<int>::iterator it = new_dests.begin(); it != new_dests.end(); it++)
      {
        // debug
//         fprintf(stderr, "gid %d sending %.1f %.1f %.1f to gid %d\n", 
//                 dblock->gid, rp.x, rp.y, rp.z, cp.link()->target(*it).gid);
        cp.enqueue(cp.link()->target(*it), rp);
      }
    }

  } // for convex hull particles

}
//
// parse received particles
//
void neighbor_particles(void* b_, const diy::Master::ProxyWithLink& cp, void*)
{
  dblock_t*  b = (dblock_t*)b_;
  diy::Link* l = cp.link();
  std::vector<int> in;
  cp.incoming(in);

  // count total number of incoming points
  int numpts = 0;
  for (int i = 0; i < (int)in.size(); i++)
  {
    numpts += cp.incoming(in[i]).buffer.size() / sizeof(RemotePoint);
  }

  // grow space for remote tet verts
  int n = (b->num_particles - b->num_orig_particles);
  int new_remote_particles = numpts + n;
  b->num_rem_tet_verts = new_remote_particles;
  b->rem_tet_verts = 
    (struct remote_vert_t *)realloc(b->rem_tet_verts, 
                                    new_remote_particles * 
                                    sizeof(struct remote_vert_t));

  // grow space for particles
  if (numpts)
  {
    b->particles = (float *)realloc(b->particles, (b->num_particles + numpts) * 3 * sizeof(float));
  }

  // copy received particles 
  for (int i = 0; i < (int)in.size(); i++)
  {
    numpts = cp.incoming(in[i]).buffer.size() / sizeof(RemotePoint);
    // debug
//     fprintf(stderr, "gid %d received %d points from gid %d\n", b->gid, numpts, in[i]);
    vector<RemotePoint> pts;
    pts.reserve(numpts);
    // NB ProxyWithLink::dequeue is for only one object, must use diy::load instead
    diy::load(cp.incoming(in[i]), &pts[0], numpts);

    for (int j = 0; j < numpts; j++)
    {
      // debug
//       fprintf(stderr, "gid %d received %.1f %.1f %.1f from gid %d\n", 
//               b->gid, pts[j].x, pts[j].y, pts[j].z, in[i]);
      b->particles[3 * b->num_particles]     = pts[j].x;
      b->particles[3 * b->num_particles + 1] = pts[j].y;
      b->particles[3 * b->num_particles + 2] = pts[j].z;
      b->rem_tet_verts[n].gid                = pts[j].gid;
      b->rem_tet_verts[n].nid                = pts[j].nid;
      b->rem_tet_verts[n].dir                = pts[j].dir;

      b->num_particles++;
      n++;
    }
  }

}
//
// block cleanup functions
//
void destroy_block(struct dblock_t *dblock)
{
  if (dblock->particles)
    free(dblock->particles);
  if (dblock->tets)
    free(dblock->tets);
  if (dblock->rem_tet_verts)
    free(dblock->rem_tet_verts);
  if (dblock->vert_to_tet)
    free(dblock->vert_to_tet);

  delete dblock;
}

void reset_block(struct dblock_t* &dblock)
{
  // free old data
  if (dblock->tets)
    free(dblock->tets);
  if (dblock->vert_to_tet)
    free(dblock->vert_to_tet);

  // initialize new data
  dblock->num_tets = 0;
  dblock->tets = NULL;
  dblock->vert_to_tet = NULL;
}
//
//   finds the direction of the nearest block to the given point
//
//   p: coordinates of the point
//   mins, maxs: block bounds
// 
diy::Direction nearest_neighbor(float* p, float* mins, float* maxs)
{
  // TODO: possibly find the 3 closest neighbors, and look at the ratio of
  //   the distances to deal with the corners  

  int               i;
  float             dists[6];
  diy::Direction    dirs[6] = { DIY_X0, DIY_X1, DIY_Y0, DIY_Y1, DIY_Z0, DIY_Z1 };

  for (i = 0; i < 3; ++i)
  {
    dists[2*i]     = p[i] - mins[i];
    dists[2*i + 1] = maxs[i] - p[i];
  }

  int   smallest = 0;
  for (i = 1; i < 6; ++i)
  {
    if (dists[i] < dists[smallest])
      smallest = i;
  }

  return dirs[smallest];
}
// --------------------------------------------------------------------------
//
// for each vertex saves a tet that contains it
//
void fill_vert_to_tet(dblock_t* dblock) {

  dblock->vert_to_tet = 
    (int*)realloc(dblock->vert_to_tet, sizeof(int) * dblock->num_particles);

  for (int p = 0; p < dblock->num_particles; ++p)
      dblock->vert_to_tet[p] = -1;

  for (int t = 0; t < dblock->num_tets; ++t) {
    for (int v = 0; v < 4; ++v) {
      int p = dblock->tets[t].verts[v];
      dblock->vert_to_tet[p] = t;	// the last one wins
    }
  }

}
// --------------------------------------------------------------------------
//
//   CLP Add wall particles
//
//   nblocks: local number of blocks
//   dblocks: local blocks
//
void wall_particles(struct dblock_t *dblock) {
 
  //   using data_mins and data_maxs
  //   Currently assuimg walls on all sides, but format can easily be 
  //   modified to be ANY set of walls 
  struct wall_t *walls = NULL;
  int num_walls = 0;

  create_walls(&num_walls,&walls);
  
  int* wall_cut = (int *)malloc(num_walls * sizeof(int));
 
  std::vector<double> new_points;

  // Find all particles that need to be mirrored.
  for (int p = 0; p < dblock->num_orig_particles; ++p) {
  
    //  zero generate-wall-point array (length of number of walls)
    for (int wi = 0; wi < num_walls; wi++) wall_cut[wi] = 0;

    // Determine if the cell is complete
    vector< pair<int, int> > nbrs;
    bool finite = neighbor_edges(nbrs, p, dblock->tets, dblock->vert_to_tet[p]);

    if (!finite) {
      //  set the mirror-generate array to all ones
      // (extra calculations but simpler to assume!) 
      for (int wi = 0; wi < num_walls; wi++)
	wall_cut[wi] = 1;
    }
    else {
      // loop throug the list of all the Voronoi cell vertices of the point.  
      // See if any are outside a wall.
        
      // neighbor edges a vector of (vertex u, tet of vertex u) pairs
      // that neighbor vertex v
        
      // the following loop is the equivalent of
      // for all faces in a voronoi cell
      for (int i = 0; i < (int)nbrs.size(); ++i) {

	// get edge link
	int u  = nbrs[i].first;
	int ut = nbrs[i].second;
	std::vector<int> edge_link;
	fill_edge_link(edge_link, p, u, ut, dblock->tets);

	// following is equivalent of all vertices in a face
	for (int j = 0; j < (int)edge_link.size(); ++j) {
	  float pt[3];
	  circumcenter(pt,&(dblock->tets[edge_link[j]]), dblock->particles);
	  for (int wi = 0; wi < num_walls; wi++)
	    if (!wall_cut[wi])
	      wall_cut[wi] = test_outside(pt,&walls[wi]);
                    
	}
      }
    }
      
    // Make the mirrored particles
    //   For each mirror-generate index that is 1,
    //   generate the mirror point given site rp and the wall
    //   Create list of points
    for (int wi =0; wi < num_walls; wi++) {

      if (wall_cut[wi]) {
        float rpt[3];
        float spt[3];
        spt[0] = dblock->particles[3 * p];
        spt[1] = dblock->particles[3 * p + 1];
        spt[2] = dblock->particles[3 * p + 2];
        generate_mirror(rpt,spt,&walls[wi]);
        new_points.push_back(rpt[0]);
        new_points.push_back(rpt[1]);
        new_points.push_back(rpt[2]);
      }

    }

  }
  
  // Add all the new points to the dblock.

  if (new_points.size()) {
    int n = (dblock->num_particles - dblock->num_orig_particles);
    int new_remote_particles = new_points.size()/3 + n;
          
    dblock->num_rem_tet_verts = new_remote_particles;
    dblock->rem_tet_verts =  
      (struct remote_vert_t *)realloc(dblock->rem_tet_verts,
				      new_remote_particles * 
				      sizeof(struct remote_vert_t));
 
    // grow space 
    dblock->particles = 
      (float *)realloc(dblock->particles,
		       (dblock->num_particles*3 + 
			new_points.size())*sizeof(float));

    // copy new particles
    for (int j = 0; j < (int)new_points.size(); j=j+3) {
      dblock->particles[3 * dblock->num_particles    ] = new_points[j    ];
      dblock->particles[3 * dblock->num_particles + 1] = new_points[j + 1];
      dblock->particles[3 * dblock->num_particles + 2] = new_points[j + 2];
      dblock->rem_tet_verts[n].gid = -1;
      dblock->rem_tet_verts[n].nid = -1;
      dblock->rem_tet_verts[n].dir = 0x00;
      dblock->num_particles++;
      n++;
    }
  }
  
  // CLP cleanup 
  free(wall_cut);
  destroy_walls(num_walls, walls);

}
// --------------------------------------------------------------------------
//
//   collects statistics
//
//   nblocks: number of blocks
//   dblocks: local delaunay blocks
//   times: timing info
// 
void collect_stats(int nblocks, struct dblock_t *dblocks, double *times) {

  nblocks = nblocks; // quite compiler warning
  int rank;

    double max_times[TESS_MAX_TIMES];
  MPI_Reduce( times, max_times, TESS_MAX_TIMES, MPI_DOUBLE, MPI_MAX, 0, comm);

  // TODO: need to first compute over all blocks
  const int MAX_QUANTS = 10;
  int quants[MAX_QUANTS], min_quants[MAX_QUANTS], max_quants[MAX_QUANTS];
  quants[0] = dblocks[0].num_orig_particles;
  quants[1] = dblocks[0].num_particles;
  quants[2] = dblocks[0].num_tets;
  quants[3] = dblocks[0].num_rem_tet_verts;
  MPI_Reduce( quants, min_quants, MAX_QUANTS, MPI_INT, MPI_MIN, 0, comm);
  MPI_Reduce( quants, max_quants, MAX_QUANTS, MPI_INT, MPI_MAX, 0, comm);

  // --- print output --- 

  // global stats 
  if (rank == 0) {
    fprintf(stderr, "----------------- global stats ------------------\n");
    fprintf(stderr, "first local delaunay time     = %.3lf s\n",
	    times[LOC1_TIME]);
    fprintf(stderr, "first incomplete cell time    = %.3lf s\n",
	    times[INC1_TIME]);
    fprintf(stderr, "first particle exchange time  = %.3lf s\n", 
	    times[NEIGH1_TIME]);
    fprintf(stderr, "second local delaunay time    = %.3lf s\n",
	    times[LOC2_TIME]);
    fprintf(stderr, "second incomplete cell time   = %.3lf s\n",
	    times[INC2_TIME]);
    fprintf(stderr, "second particle exchange time = %.3lf s\n", 
	    times[NEIGH2_TIME]);
    fprintf(stderr, "third local delaunay time     = %.3lf s\n",
	    times[LOC3_TIME]);
    fprintf(stderr, "output time                   = %.3lf s\n", 
	    times[OUT_TIME]);
    fprintf(stderr, "total time                    = %.3lf s\n", 
	    times[TOT_TIME]);
    fprintf(stderr, "All times printed in one row:\n");
    fprintf(stderr, "%.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf\n",
	    times[LOC1_TIME], times[INC1_TIME], times[NEIGH1_TIME],
	    times[LOC2_TIME], times[INC2_TIME], times[NEIGH2_TIME],
	    times[LOC3_TIME], times[OUT_TIME], times[TOT_TIME]);
    fprintf(stderr, "-------------------------------------------------\n");
    fprintf(stderr, "original particles = [%d, %d]\n", min_quants[0], max_quants[0]);
    fprintf(stderr, "with ghosts        = [%d, %d]\n", min_quants[1], max_quants[1]);
    fprintf(stderr, "tets               = [%d, %d]\n", min_quants[2], max_quants[2]);
    fprintf(stderr, "remote tet verts   = [%d, %d]\n", min_quants[3], max_quants[3]);
    fprintf(stderr, "-------------------------------------------------\n");
  }

}
// --------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//
//   randomly samples a set of partidles
//
//   particles: pointer to particles (input and output)
//   num_particles: number of particles (input and output)
//   sample_rate: 1 out of every sample_rate particles will be kept
// 
//   overwrites the old particles with the new but does not shrink memory
//
void sample_particles(float *particles, int &num_particles, int sample_rate) {

  int old_num_particles = num_particles;
  num_particles /= sample_rate;
  float *new_particles = new float[3 * num_particles];

  // sample particles
  for (int i = 0; i < num_particles; i++) {
    int rand_i = rand() / (float)RAND_MAX * old_num_particles;
    new_particles[3 * i] = particles[3 * rand_i];
    new_particles[3 * i + 1] = particles[3 * rand_i + 1];
    new_particles[3 * i + 2] = particles[3 * rand_i + 2];
  }

  // copy samples back to original
  for (int i = 0; i < num_particles; i++) {
    particles[3 * i] = new_particles[3 * i];
    particles[3 * i + 1] = new_particles[3 * i + 1];
    particles[3 * i + 2] = new_particles[3 * i + 2];
  }

  delete[] new_particles;

}
// --------------------------------------------------------------------------
//
//   prints a block
//
//   dblock: current delaunay block
//   gid: global block id
// 
void print_block(struct dblock_t *dblock, int gid) {

  fprintf(stderr, "block gid = %d has %d tets:\n", 
	  gid, dblock->num_tets);
  for (int i = 0; i < dblock->num_tets; i++) {
    int sort_verts[4], sort_tets[4]; // sorted verts and tets
    for (int j = 0; j < 4; j++) {
      sort_verts[j] = dblock->tets[i].verts[j];
      sort_tets[j] = dblock->tets[i].tets[j];
    }
    qsort(sort_verts, 4, sizeof(int), &compare);
    qsort(sort_tets, 4, sizeof(int), &compare);
    fprintf(stderr, "tet %d verts [%d %d %d %d] neigh_tets [%d %d %d %d]\n", 
	    i, sort_verts[0], sort_verts[1], sort_verts[2], sort_verts[3],
	    sort_tets[0], sort_tets[1], sort_tets[2], sort_tets[3]);
  }

}
// --------------------------------------------------------------------------
//
//   prints particles
//
//   prticles: particle array
//   num_particles: number of particles
//   gid: block global id
// 
void print_particles(float *particles, int num_particles, int gid) {

  int n;

  for (n = 0; n < num_particles; n++)
    fprintf(stderr, "block = %d particle[%d] = [%.1lf %.1lf %.1lf]\n",
	    gid, n, particles[3 * n], particles[3 * n + 1],
	    particles[3 * n + 2]);

}
// --------------------------------------------------------------------------
//
//   transforms particles for enqueueing to wraparound neighbors
//   p: pointer to particle
//   wrap_dir: wrapping direcion
// 
void transform_particle(char *p, unsigned char wrap_dir) {

#if 0

  // debug 
  float particle[3]; // original particle 
  particle[0] = ((struct remote_particle_t*)p)->x;
  particle[1] = ((struct remote_particle_t*)p)->y;
  particle[2] = ((struct remote_particle_t*)p)->z;

  // wrapping toward the left transforms to the right 
  if ((wrap_dir & DIY_X0) == DIY_X0) {
    ((struct remote_particle_t*)p)->x += (data_maxs[0] - data_mins[0]);
    ((struct remote_particle_t*)p)->dir |= DIY_X0;
  }

  // and vice versa 
  if ((wrap_dir & DIY_X1) == DIY_X1) {
    ((struct remote_particle_t*)p)->x -= (data_maxs[0] - data_mins[0]);
    ((struct remote_particle_t*)p)->dir |= DIY_X1;
  }

  // similar for y, z 
  if ((wrap_dir & DIY_Y0) == DIY_Y0) {
    ((struct remote_particle_t*)p)->y += (data_maxs[1] - data_mins[1]);
    ((struct remote_particle_t*)p)->dir |= DIY_Y0;
  }

  if ((wrap_dir & DIY_Y1) == DIY_Y1) {
    ((struct remote_particle_t*)p)->y -= (data_maxs[1] - data_mins[1]);
    ((struct remote_particle_t*)p)->dir |= DIY_Y1;
  }

  if ((wrap_dir & DIY_Z0) == DIY_Z0) {
    ((struct remote_particle_t*)p)->z += (data_maxs[2] - data_mins[2]);
    ((struct remote_particle_t*)p)->dir |= DIY_Z0;
  }

  if ((wrap_dir & DIY_Z1) == DIY_Z1) {
    ((struct remote_particle_t*)p)->z -= (data_maxs[2] - data_mins[2]);
    ((struct remote_particle_t*)p)->dir |= DIY_Z1;
  }

#endif

}
// --------------------------------------------------------------------------
//
//    comparison function for qsort
// 
int compare(const void *a, const void *b) {

  if (*((int*)a) < *((int*)b))
    return -1;
  if (*((int*)a) == *((int*)b))
    return 0;
  return 1;

}
// --------------------------------------------------------------------------
//
//  writes particles to a file in interleaved x,y,z order
//  no other block information is retained, all particles get sqaushed together
//
//  nblocks: locaal number of blocks
//  particles: particles in each local block
//  num_particles: number of particles in each local block
//  outfile: output file name
// 
void write_particles(int nblocks, float **particles, int *num_particles, 
		     char *outfile) {

  int i;
  float *all_particles; // particles from all local blocks merged 
  MPI_Offset my_size; // size of my particles 
  MPI_Offset ofst; // offset in file where I write my particles 
  MPI_File fd;
  MPI_Status status;

  // merge all blocks together into one 
  int tot_particles = 0;
  for (i = 0; i < nblocks; i++)
    tot_particles += num_particles[i];
  all_particles = (float *)malloc(tot_particles * 3 * sizeof(float));
  int n = 0;
  for (i = 0; i < nblocks; i++) {
    memcpy(&all_particles[n], &particles[i][0], 
	   num_particles[i] * 3 * sizeof(float));
    n += num_particles[i] * 3;
  }

  // compute my offset 
  my_size = tot_particles * 3 * sizeof(float);
  // in MPI 2.2, should use MPI_OFFSET datatype, but
  //    using MPI_LONG_LONG for backwards portability 
  MPI_Exscan(&my_size, &ofst, 1, MPI_LONG_LONG, MPI_SUM, comm);
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0)
    ofst = 0;

  // open 
  int retval = MPI_File_open(comm, (char *)outfile,
			     MPI_MODE_WRONLY | MPI_MODE_CREATE,
			     MPI_INFO_NULL, &fd);
  assert(retval == MPI_SUCCESS);
  MPI_File_set_size(fd, 0); // start with an empty file every time

  // write 
  retval = MPI_File_write_at_all(fd, ofst, all_particles, tot_particles * 3, 
				 MPI_FLOAT, &status);
  if (retval != MPI_SUCCESS)
    handle_error(retval, comm, (char *)"MPI_File_write_at_all");

  // cleanup 
  free(all_particles);
  MPI_File_close(&fd);

}
// --------------------------------------------------------------------------
//
//   MPI error handler
//   decodes and prints MPI error messages
// 
void handle_error(int errcode, MPI_Comm comm, char *str) {

  char msg[MPI_MAX_ERROR_STRING];
  int resultlen;
  MPI_Error_string(errcode, msg, &resultlen);
  fprintf(stderr, "%s: %s\n", str, msg);
  MPI_Abort(comm, 1);

}
// --------------------------------------------------------------------------
// CLP
//   creates and initializes walls
//
//   walls: pointer to array of walls
//   
//   Important!  [a,b,c] must be a unit length vector!
//
//    allocate blocks and headers
//
// 
void create_walls(int *num_walls, struct wall_t **walls) {

  (*num_walls) = 6;
  *walls = (struct wall_t*)malloc(sizeof(struct wall_t) * (*num_walls));
  
  // bottom xy wall
  (*walls)[0].a = 0;
  (*walls)[0].b = 0;
  (*walls)[0].c = 1;
  (*walls)[0].d = -data_mins[2];
  
  // forward xz wall
  (*walls)[1].a = 0;
  (*walls)[1].b = 1;
  (*walls)[1].c = 0;
  (*walls)[1].d = -data_mins[1];
  
  // left yz wall
  (*walls)[2].a = 1;
  (*walls)[2].b = 0;
  (*walls)[2].c = 0;
  (*walls)[2].d = -data_mins[0];
  
  // top xy wall
  (*walls)[3].a = 0;
  (*walls)[3].b = 0;
  (*walls)[3].c = -1;
  (*walls)[3].d = data_maxs[2];
  
  // back xz wall
  (*walls)[4].a = 0;
  (*walls)[4].b = -1;
  (*walls)[4].c = 0;
  (*walls)[4].d = data_maxs[1];
  
  // right yz wall
  (*walls)[5].a = -1;
  (*walls)[5].b = 0;
  (*walls)[5].c = 0;
  (*walls)[5].d = data_maxs[0];
  
}
// ---------------------------------------------------------------------------
// CLP
//   frees walls
//
//   num_walls: number of walls
//   walls: array of walls
// 
void destroy_walls(int num_walls, struct wall_t *walls) {

  if (num_walls)
    free(walls);

}
// ---------------------------------------------------------------------------
// CLP
//   determines if point is inside or outside wall
//   http://mathworld.wolfram.com/Plane.html Equation (24) 
//   Signed point-plane distance, ignoring positive denominator
//   pt: point
//   wall: wall
// 
int test_outside(const float *pt,const struct wall_t *wall) {

  float D = pt[0]*wall->a + pt[1]*wall->b + pt[2]* wall->c + wall->d;
  if (D > 0)
    return 0;
  else 
    return 1;

}
// ---------------------------------------------------------------------------
// CLP
//   returns point reflected across wall
//   http://mathworld.wolfram.com/Plane.html Equation (24) 
//   Signed point-plane distance, including denominator
//   rpt
//   pt: site-point
//   wall: wall
//
void generate_mirror(float *rpt, const float *pt, const struct wall_t *wall) {

  // signed distance from wall to site point 
  float D = (pt[0]*wall->a + pt[1]*wall->b + pt[2]* wall->c + 
	     wall->d)/sqrt(wall->a*wall->a + wall->b*wall->b + 
			   wall->c*wall->c);
    
  rpt[0] = pt[0] - 2*D*wall->a;
  rpt[1] = pt[1] - 2*D*wall->b;
  rpt[2] = pt[2] - 2*D*wall->c;

}
// ---------------------------------------------------------------------------
// CLP
//  Adds mirror particles to list
// 
void add_mirror_particles(int nblocks, float **mirror_particles, 
			  int *num_mirror_particles, float **particles,
			  int *num_particles, int *num_orig_particles,
			  int **gids, int **nids, unsigned char **dirs) {

  int i,j;
    
  // copy mirror particles to particles 
  for (i = 0; i < nblocks; i++) {

    int n = (num_particles[i] - num_orig_particles[i]);
    int new_remote_particles = num_mirror_particles[i] + n;

    gids[i] = (int *)realloc(gids[i], new_remote_particles * sizeof(int));
    nids[i] = (int *)realloc(nids[i], new_remote_particles * sizeof(int));
    dirs[i] = (unsigned char *)realloc(dirs[i], new_remote_particles);
    if (num_mirror_particles[i]) {

      // grow space 
      particles[i] = 
	(float *)realloc(particles[i], 
			 (num_particles[i] + num_mirror_particles[i]) *
			 3 * sizeof(float));

      // copy mirror particles 
      for (j = 0; j < num_mirror_particles[i]; j++) {

            particles[i][3 * num_particles[i]] = mirror_particles[i][3*j];
            particles[i][3 * num_particles[i] + 1] = mirror_particles[i][3*j+1];
            particles[i][3 * num_particles[i] + 2] = mirror_particles[i][3*j+2];
            gids[i][n] = -1;
            nids[i][n] = -1;
            dirs[i][n] = 0x00;

            num_particles[i]++;
            n++;

      } // copy mirror particles 

    } // if num_mirror_particles 

  }

}
// ---------------------------------------------------------------------------
//
// memory profile, prints max reseident usage and sleeps so that user
// can read current usage some other way, eg. an OS tool or dashboard
//
// breakpoint: breakpoint number
// dwell: sleep time in seconds
//
void get_mem(int breakpoint, int dwell) {

  // quite compiler warnings in case MEMORY is not defined
  breakpoint = breakpoint;
  dwell = dwell;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef MEMORY

#ifdef BGQ

  uint64_t shared, persist, heapavail, stackavail,
    stack, heap, heapmax, guard, mmap;
	
  // we're only interested in max heap size 
  // (same as max resident size, high water mark)
  Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPMAX, &heapmax);

  // some examples of other memory usage we could get if we wanted it
  // note that stack and heap both count the total of both, use one or the other
  Kernel_GetMemorySize(KERNEL_MEMSIZE_SHARED, &shared);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_PERSIST, &persist);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPAVAIL, &heapavail);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_STACKAVAIL, &stackavail);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_STACK, &stack);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAP, &heap);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_GUARD, &guard);
  Kernel_GetMemorySize(KERNEL_MEMSIZE_MMAP, &mmap);
	
  int to_mb = 1024 * 1024;
  double heap_mem = double(heapmax) / to_mb;
  double max_heap_mem;
  MPI_Reduce(&heap_mem, &max_heap_mem, 1, MPI_DOUBLE, MPI_MAX, 0, 
	     MPI_COMM_WORLD);
  if (rank == 0)
    fprintf(stderr, "%d: BGQ max memory = %.0lf MB\n", 
	    breakpoint, max_heap_mem);

#else

  struct rusage r_usage;
  getrusage(RUSAGE_SELF, &r_usage);

#ifdef __APPLE__
  const int to_mb = 1048576;
#else
  const int to_mb = 1024;
#endif

  float res = r_usage.ru_maxrss;
  float mem = res / (float)to_mb;
  float max_mem;
  MPI_Reduce(&mem, &max_mem, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0)
    fprintf(stderr, "%d: max memory = %0.1f MB\n", breakpoint, max_mem);
//   sleep(dwell);
//   fprintf(stderr, "%d: done\n", breakpoint);

#endif // BGQ

#endif // MEMORY

}
// ---------------------------------------------------------------------------
//
// starts / stops timing
// (does a barrier)
//
// times: array of times
// start: index of timer to start (-1 if not used)
// stop: index of timer to stop (-1 if not used)
//
void timing(double *times, int start, int stop) {

#ifdef TIMING

  MPI_Barrier(comm);
  if (start >= 0)
    times[start] = MPI_Wtime();
  if (stop >= 0)
    times[stop] = MPI_Wtime() - times[stop];

#endif

}
// ---------------------------------------------------------------------------
