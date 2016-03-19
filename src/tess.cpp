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
#include "tess/tet.h"
#include "tess/tet-neighbors.h"

#ifdef BGQ
#include <spi/include/kernel/memory.h>
// #include "builtins.h"
// #define __builtin_ctzll(x) __cnttz8(x)
// #define __builtin_clzll(x) __cntlz8(x);
#endif

using namespace std;

typedef std::vector<RCLink> LinkVector;
typedef std::vector<size_t> LastNeighbors;

struct Aux
{
  LinkVector*       links;
  LastNeighbors*    neighbors;
  bool              first;
};

size_t tess(diy::Master& master)
{
  double times[TESS_MAX_TIMES]; // timing
  quants_t quants; // quantity stats
  return tess(master, quants, times);
}

size_t tess(diy::Master& master,
            quants_t& quants,
            double* times)
{
#ifdef TIMING
  // if (master.threads() != 1)
  //   fprintf(stderr, "Warning: timing() calls MPI directly; "
  //           "it's not compatible with using multiple threads\n");
#endif

  timing(times, DEL_TIME, -1, master.communicator());

  // save the original link for every block in master
  LinkVector   original_links;
  for (size_t i = 0; i < master.size(); ++i)
  {
    original_links.push_back(*dynamic_cast<RCLink*>(master.link(i)));
    //fprintf(stderr, "Link of %d\n", master.gid(i));
    //for (size_t j = 0; j < original_links.back().size(); ++j)
    //  fprintf(stderr, "  %d\n", original_links.back().target(j).gid); 
  }
  
  // array of the previous link sizes
  LastNeighbors last_neighbors(master.size(), 0);

  Aux aux;
  aux.links = &original_links;
  aux.neighbors = &last_neighbors;
  aux.first = true;

  int done = false;
  size_t rounds = 0;
  while(!done)
  {
    rounds++;
 
    master.foreach(&delaunay, &aux);
    master.exchange();

    aux.first = false;
    done = master.proxy(master.loaded_block()).read<int>();

#ifdef MEMORY
    get_mem(rounds, master.communicator());
#endif
  }

  // this is not ideal, but need to do this to collect statistics and mark
  // blocks as complete; TODO: this of how to get rid of this
  quants.sum_quants[NUM_ORIG_PTS] = 0;
  quants.sum_quants[NUM_FINAL_PTS] = 0;
  quants.sum_quants[NUM_TETS] = 0;
  quants.sum_quants[NUM_LOC_BLOCKS] = master.size();
  master.foreach(&finalize, &quants);
    
  // restore the original links
  for (size_t i = 0; i < master.size(); ++i)
    master.replace_link(i, new RCLink(original_links[i]));
  
  timing(times, -1, DEL_TIME, master.communicator());

  return rounds;
}

void tess_save(diy::Master& master, const char* outfile, const diy::MemoryBuffer& extra)
{
  double times[TESS_MAX_TIMES]; // timing
  tess_save(master, outfile, times);
}

void tess_save(diy::Master& master, const char* outfile, double* times, const diy::MemoryBuffer& extra)
{
  // write output
  timing(times, OUT_TIME, -1, master.communicator());
  if (outfile[0])
    diy::io::write_blocks(outfile, master.communicator(), master, extra, &save_block_light);

  timing(times, -1, OUT_TIME, master.communicator());
}

void tess_load(diy::Master& master, diy::Assigner& assigner, const char* infile)
{
  diy::io::read_blocks(infile, master.communicator(), assigner, master, &load_block_light);
}

void tess_load(diy::Master& master, diy::Assigner& assigner, const char* infile, diy::MemoryBuffer& extra)
{
  diy::io::read_blocks(infile, master.communicator(), assigner, master, extra, &load_block_light);
}

//
// diy::Master callback functions
//
void* create_block()
{
    dblock_t* b = new dblock_t;
    b->complete = 0;
    init_delaunay_data_structure(b);
    return b;
}

void destroy_block(void* b_)
{
  dblock_t* b = static_cast<dblock_t*>(b_);

  // particles and tets
  if (b->particles)     free(b->particles);
  if (b->tets)          free(b->tets);
  if (b->rem_gids)      free(b->rem_gids);
  if (b->rem_lids)      free(b->rem_lids);
  if (b->vert_to_tet)   free(b->vert_to_tet);

  // density
  if (b->density)
    delete[] b->density;   // allocated with new, freed with delete

  if (b->Dt)
      clean_delaunay_data_structure(b);

  delete b;
}

void save_block(const void* b, diy::BinaryBuffer& bb)
{
  diy::save(bb, *static_cast<const dblock_t*>(b));
}

void load_block(void* b, diy::BinaryBuffer& bb)
{
  diy::load(bb, *static_cast<dblock_t*>(b));
}

void save_block_light(const void* b_, diy::BinaryBuffer& bb)
{
  const dblock_t& d = *static_cast<const dblock_t*>(b_);

  diy::save(bb, d.gid);
  diy::save(bb, d.mins);
  diy::save(bb, d.maxs);
  diy::save(bb, d.box);
  diy::save(bb, d.data_bounds);
  diy::save(bb, d.num_orig_particles);
  diy::save(bb, d.num_particles);
  diy::save(bb, d.particles, 3 * d.num_particles);
  diy::save(bb, d.rem_gids, d.num_particles - d.num_orig_particles);
  diy::save(bb, d.rem_lids, d.num_particles - d.num_orig_particles);
  diy::save(bb, d.num_grid_pts);
  diy::save(bb, d.density, d.num_grid_pts);

  diy::save(bb, d.complete);
  diy::save(bb, d.num_tets);
  diy::save(bb, d.tets, d.num_tets);
  diy::save(bb, d.vert_to_tet, d.num_particles);
}

void load_block_light(void* b_, diy::BinaryBuffer& bb)
{
  dblock_t& d = *static_cast<dblock_t*>(b_);

  diy::load(bb, d.gid);
  // debug
  // fprintf(stderr, "Loading block gid %d\n", d.gid);
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
  d.rem_lids = NULL;
  if (d.num_particles - d.num_orig_particles)
  {
    d.rem_gids = (int*)malloc((d.num_particles - d.num_orig_particles) * sizeof(int));
    d.rem_lids = (int*)malloc((d.num_particles - d.num_orig_particles) * sizeof(int));
  }
  diy::load(bb, d.rem_gids, d.num_particles - d.num_orig_particles);
  diy::load(bb, d.rem_lids, d.num_particles - d.num_orig_particles);
  diy::load(bb, d.num_grid_pts);
  d.density = new float[d.num_grid_pts];
  diy::load(bb, d.density, d.num_grid_pts);

  diy::load(bb, d.complete);
  diy::load(bb, d.num_tets);
  d.tets = (tet_t*)malloc(d.num_tets * sizeof(tet_t));
  diy::load(bb, d.tets, d.num_tets);
  d.vert_to_tet = NULL;
  if (d.num_particles)
    d.vert_to_tet = (int*)malloc(d.num_particles * sizeof(int));
  diy::load(bb, d.vert_to_tet, d.num_particles);
}

//
// generate particles, return final number of particles generated
//
int gen_particles(dblock_t* b, float jitter)
{
    // to test 0-blocks, make 0 particles in block gid 0
    // if (!b->gid)
    //     return 0;

  int sizes[3]; // number of grid points
  int n = 0;
  int num_particles; // theoretical num particles with duplicates at block boundaries
  float jit; // random jitter amount, 0 - MAX_JITTER

  // allocate particles
  sizes[0] = (int)(b->maxs[0] - b->mins[0] + 1);
  sizes[1] = (int)(b->maxs[1] - b->mins[1] + 1);
  sizes[2] = (int)(b->maxs[2] - b->mins[2] + 1);
  num_particles = sizes[0] * sizes[1] * sizes[2];
  b->particles = (float *)malloc(num_particles * 3 * sizeof(float));
  float *p = b->particles;

  // assign particles
  srand(b->gid);

#if 1 // generate points uniformly at random in the block

  for (unsigned i = 0; i < num_particles; ++i)
  {
    for (unsigned j = 0; j < 3; ++j)
    {
      float t = (float) rand() / RAND_MAX;
      p[3 * i + j] = t * (b->maxs[j] - b->mins[j]) + b->mins[j];
    }
    ++n;
  }

#else  // randomly jitter points on a grid

  n = 0;
  for (unsigned i = 0; i < sizes[0]; i++)
  {
    if (b->mins[0] > 0 && i == 0) // dedup block doundary points
      continue;
    for (unsigned j = 0; j < sizes[1]; j++)
    {
      if (b->mins[1] > 0 && j == 0) // dedup block doundary points
        continue;
      for (unsigned k = 0; k < sizes[2]; k++)
      {
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

#endif

  return n;
}

void delaunay(void* b_, const diy::Master::ProxyWithLink& cp, void* aux_)
{
  dblock_t*         b     = (dblock_t*) b_;

  Aux* aux = static_cast<Aux*>(aux_);

  int               lid           = cp.master()->lid(cp.gid());
  const RCLink&     original_link = (*aux->links)[lid];
  size_t&           last_neighbor = (*aux->neighbors)[lid];
  bool              first         = aux->first;

  RCLink*           link          = dynamic_cast<RCLink*>(cp.link());

  // cleanup block
  reset_block(b);

  // clear collectives
  cp.collectives()->clear();
  
  LinkVector in_links;
  if (!first)       // we don't receive on the first round
  {
    // dequeue links and work out the duplicates
    for (size_t i = last_neighbor; i < link->size(); ++i)
    {
      diy::MemoryBuffer& in = cp.incoming(link->target(i).gid);

      RCLink* l = dynamic_cast<RCLink*>(diy::LinkFactory::load(in));
      in_links.push_back(*l);
      delete l;
    }
    size_t last_last_neighbor = last_neighbor;
    last_neighbor = link->size();       // update last_neighbor

    // parse received particles
    neighbor_particles(b, cp);

    // update the links, taking care of duplicates

    std::set< std::pair<diy::BlockID, diy::Direction> > neighbor_blocks;
    for (size_t i = 0; i < link->size(); ++i)
      neighbor_blocks.insert(std::make_pair(link->target(i), link->wrap(i)));
    size_t original_size_unique = link->size_unique();

    // NB: this does not take care of the wrap properly
    //     (in fact, even duplicate pruning is based on BlockID)
    //     So this wouldn't work with wrap on (past a single round).
    for (size_t i = 0; i < in_links.size(); ++i)
    {
      size_t ii = i + last_last_neighbor;
      for (size_t j = 0; j < in_links[i].size(); ++j)
      {
        if (in_links[i].target(j).gid == cp.gid()) continue;        // skip self

	// add wrap of the neighbor we got the link from to the link's wrap
	diy::Direction wrap = link->wrap(ii);
	//fprintf(stderr, "[%d] -> %d: wrap = (%d,%d,%d); -> %d = (%d,%d,%d)\n",
	//                cp.gid(), link->target(ii).gid,
	//                wrap[0], wrap[1], wrap[2],
	//                in_links[i].target(j).gid,
	//                in_links[i].wrap(j)[0], in_links[i].wrap(j)[1], in_links[i].wrap(j)[2]);
				
	for (int k = 0; k < 3; ++k)
	{
	  wrap[k] += in_links[i].wrap(j)[k];
	  if (wrap[k] < -1 || wrap[k] > 1)
	  {
	    fprintf(stderr, "Warning: something is odd with the wrap, the it exceeds a single wrap-around\n");
	  }
	}
	//fprintf(stderr, "   -> wrap = (%d,%d,%d)\n",
	//                wrap[0], wrap[1], wrap[2]);

        bool inserted = neighbor_blocks.insert(std::make_pair(in_links[i].target(j),wrap)).second;
        if (!inserted) continue;

        link->add_neighbor(in_links[i].target(j));
        link->add_direction(in_links[i].direction(j));
        link->add_bounds(in_links[i].bounds(j));
	link->add_wrap(wrap);
      }
    }
    cp.master()->add_expected(link->size_unique() - original_size_unique);
  }
  //fprintf(stderr, "Links updated; last_neighbor = %lu\n", last_neighbor);

  // compute (or update) the local tessellation
  if (b->num_particles)                      // NB, crashes when this test is b->num_orig_particles
      local_cells(b);

  // enqueue the original link to the new neighbors
  for (size_t i = last_neighbor; i < link->size(); ++i)
  {
    diy::MemoryBuffer& out = cp.outgoing(link->target(i));
    diy::LinkFactory::save(out, &original_link);
  }

  // enqueue points to neighbors
  int done = 1;
  if (b->num_orig_particles)
  {
      size_t num = incomplete_cells(b, cp, last_neighbor);
      done = (num == 0);
  }
  cp.all_reduce(done, std::logical_and<int>());
}

void finalize(void* b_, const diy::Master::ProxyWithLink& cp, void* aux)
{
  dblock_t* b = (dblock_t*)b_;
  quants_t* quants = (quants_t*)aux;

  static bool first = true;

  b->complete = 1;

  // collect quantities
  if (first || b->num_orig_particles < quants->min_quants[NUM_ORIG_PTS])
    quants->min_quants[NUM_ORIG_PTS] = b->num_orig_particles;
  if (first || b->num_orig_particles > quants->max_quants[NUM_ORIG_PTS])
    quants->max_quants[NUM_ORIG_PTS] = b->num_orig_particles;
  quants->sum_quants[NUM_ORIG_PTS] += b->num_orig_particles;

  if (first || b->num_particles < quants->min_quants[NUM_FINAL_PTS])
    quants->min_quants[NUM_FINAL_PTS] = b->num_particles;
  if (first || b->num_particles > quants->max_quants[NUM_FINAL_PTS])
    quants->max_quants[NUM_FINAL_PTS] = b->num_particles;
  quants->sum_quants[NUM_FINAL_PTS] += b->num_particles;

  if (first || b->num_tets < quants->min_quants[NUM_TETS])
    quants->min_quants[NUM_TETS] = b->num_tets;
  if (first || b->num_tets > quants->max_quants[NUM_TETS])
    quants->max_quants[NUM_TETS] = b->num_tets;
  quants->sum_quants[NUM_TETS] += b->num_tets;

  first = false;

  // debug
//   fprintf(stderr, "phase 3 gid %d num_tets %d num_particles %d \n",
//           b->gid, b->num_tets, b->num_particles);
}

size_t incomplete_cells(struct dblock_t *dblock,
                        const diy::Master::ProxyWithLink& cp,
                        size_t last_neighbor)
{
  RCLink* l = dynamic_cast<RCLink*>(cp.link());
  std::vector< std::set<int> > to_send(dblock->num_orig_particles);

  // for all tets
  for (int t = 0; t < dblock->num_tets; t++)
  {
    int j;
    for (j = 0; j < 4; ++j)
      if (dblock->tets[t].verts[j] < dblock->num_orig_particles)
	break;
    if (j == 4)	    // no local particles in the tet, so we don't care
      continue;

    // cirumcenter of tet and radius from circumcenter to any vertex
    float center[3]; // circumcenter
    circumcenter(center, &dblock->tets[t], dblock->particles);
    int p = dblock->tets[t].verts[0];
    float rad = distance(center, &dblock->particles[3 * p]);

    // check for a convex hull facet
    for (j = 0; j < 4; ++j)
    {
      if (dblock->tets[t].tets[j] != -1) continue;	// facet opposite of vertex j is not on the convex hull

      int k;
      for (k = 0; k < 4; ++k)
      {
	if (k == j) continue;
	if (dblock->tets[t].verts[k] < dblock->num_orig_particles) break;
      }
      if (k == 4) continue;  // there is not a local particle on the convex hull

      for (int i = last_neighbor; i < l->size(); ++i)
      {
	diy::ContinuousBounds neigh_bounds = l->bounds(i);
	diy::wrap_bounds(neigh_bounds, l->wrap(i), dblock->data_bounds, l->dimension());

	// test whether there is a point in neigh_bounds that lies on the opposite side of the convex hull than j
	if (!side_of_plane(neigh_bounds.min, neigh_bounds.max, &dblock->tets[t], dblock->particles, j)) continue;

	// all 4 verts, but j go to these dests, if they are among the original particles
	for (int v = 0; v < 4; v++)
	{
	  if (v == j) continue;
	  int p = dblock->tets[t].verts[v];
	  if (p >= dblock->num_orig_particles)
	    continue;

	  to_send[p].insert(i);
	}
      }
    }

    // check if the circumsphere is too deep inside the block to be able to stick out
    for (j = 0; j < 3; ++j)
    {
      if (center[j] - l->bounds().min[j] <= rad) break;
      if (l->bounds().max[j] - center[j] <= rad) break;
    }
    if (j == 3)	// the circumsphere is too deep inside the block
      continue;

    // find nearby blocks within radius of circumcenter
    for (int i = last_neighbor; i < l->size(); ++i)
    {
      diy::ContinuousBounds neigh_bounds = l->bounds(i);
      diy::wrap_bounds(neigh_bounds, l->wrap(i), dblock->data_bounds, l->dimension());

      if (diy::distance(3, neigh_bounds, center) <= rad)
      {
        // all 4 verts go these dests, if they are among the original particles
        for (int v = 0; v < 4; v++)
        {
          int p = dblock->tets[t].verts[v];
          if (p >= dblock->num_orig_particles)
            continue;

          to_send[p].insert(i);
        }
      }
    }
  }

  // handle the weird corner case, where there are less than four points in a block;
  // in this case the previous for loop didn't execute at all
  if (dblock->num_tets == 0)
  {
    // send everybody everywhere
    for (int p = 0; p < dblock->num_orig_particles; ++p)
    {
      if (dblock->vert_to_tet[p] == -1)
      {
	fprintf(stderr, "Particle %d is not in the triangulation. "
		"Perhaps it's a duplicate? Aborting.\n", p);
	assert(false);
      }

      for (int i = last_neighbor; i < l->size(); ++i)
	to_send[p].insert(i);
    }
  }

  // enqueue the particles
  size_t enqueued = 0;
  //size_t convex_hull = 0;
  point_t rp; // particle being sent
  for (int p = 0; p < dblock->num_orig_particles; p++)
  {
    for (set<int>::iterator it  = to_send[p].begin();
                            it != to_send[p].end();
                            it++)
    {
      rp.x   = dblock->particles[3 * p];
      rp.y   = dblock->particles[3 * p + 1];
      rp.z   = dblock->particles[3 * p + 2];
      rp.gid = dblock->gid;
      rp.lid = p;
      wrap_pt(rp, l->wrap(*it), dblock->data_bounds);
      cp.enqueue(l->target(*it), rp);
      ++enqueued;

      //if (!complete(p, dblock->tets, dblock->num_tets, dblock->vert_to_tet[p]))
      //  ++convex_hull;
    }
  }
  //fprintf(stderr, "[%d]: %lu convex hull particles; %lu total\n", cp.gid(), convex_hull, enqueued);

  return enqueued;
}

//
// parse received particles
//
void neighbor_particles(dblock_t* b, const diy::Master::ProxyWithLink& cp)
{
  diy::Link* l = cp.link();
  std::vector<int> in; // gids of sources
  cp.incoming(in);

  // count total number of incoming points
  int numpts = 0;
  for (int i = 0; i < (int)in.size(); i++)
  {
    diy::MemoryBuffer& in_queue = cp.incoming(in[i]);
    numpts += (in_queue.size() - in_queue.position) / sizeof(point_t);
  }

  // grow space for remote particles
  int n = (b->num_particles - b->num_orig_particles);
  if (numpts)
  {
    b->particles = (float *)realloc(b->particles, (b->num_particles + numpts) * 3 * sizeof(float));
    b->rem_gids  = (int*)realloc(b->rem_gids, (n + numpts) * sizeof(int));
    b->rem_lids  = (int*)realloc(b->rem_lids, (n + numpts) * sizeof(int));
  }

  // copy received particles
  for (int i = 0; i < (int)in.size(); i++)
  {
    diy::MemoryBuffer& in_queue = cp.incoming(in[i]);
    numpts = (in_queue.size() - in_queue.position) / sizeof(point_t);
    vector<point_t> pts;
    pts.resize(numpts);
    diy::load(in_queue, &pts[0], numpts);

    for (int j = 0; j < numpts; j++)
    {
      b->particles[3 * b->num_particles    ] = pts[j].x;
      b->particles[3 * b->num_particles + 1] = pts[j].y;
      b->particles[3 * b->num_particles + 2] = pts[j].z;
      b->rem_gids[n] = pts[j].gid;
      b->rem_lids[n] = pts[j].lid;

      b->num_particles++;
      n++;
    }
  }
}
//
// cleans a block in between phases
// (deletes tets but keeps delauany data structure and convex hull particles, sent particles)
//
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
// wraps point coordinates
//
// wrap dir:wrapping direction from original block to wrapped neighbor block
// domain: overall domain bounds
//
void wrap_pt(point_t& rp, diy::Direction wrap_dir, Bounds& domain)
{
  rp.x -= wrap_dir[0] * (domain.max[0] - domain.min[0]);
  rp.y -= wrap_dir[1] * (domain.max[1] - domain.min[1]);
  rp.z -= wrap_dir[2] * (domain.max[2] - domain.min[2]);
}

//   collects statistics
void tess_stats(diy::Master& master,
                quants_t& quants,double* times)
{
    int global_min_quants[MAX_QUANTS],
        global_max_quants[MAX_QUANTS],
        global_sum_quants[MAX_QUANTS];

    MPI_Reduce(quants.min_quants, global_min_quants, MAX_QUANTS, MPI_INT, MPI_MIN, 0,
               master.communicator());
    MPI_Reduce(quants.max_quants, global_max_quants, MAX_QUANTS, MPI_INT, MPI_MAX, 0,
               master.communicator());
    MPI_Reduce(quants.sum_quants, global_sum_quants, MAX_QUANTS, MPI_INT, MPI_SUM, 0,
               master.communicator());

  if (master.communicator().rank() == 0)
  {
    fprintf(stderr, "----------------- global stats ------------------\n");
    fprintf(stderr, "particle exchange time        = %.3lf s\n", times[EXCH_TIME]);
    fprintf(stderr, "delaunay computation time     = %.3lf s\n", times[DEL_TIME]);
    fprintf(stderr, "output time                   = %.3lf s\n", times[OUT_TIME]);
    fprintf(stderr, "total time                    = %.3lf s\n", times[TOT_TIME]);
    fprintf(stderr, "All times printed in one row:\n");
    fprintf(stderr, "%.3lf %.3lf %.3lf %.3lf\n",
            times[EXCH_TIME], times[DEL_TIME],
            times[OUT_TIME],  times[TOT_TIME]);
    fprintf(stderr, "-------------------------------------------------\n");
    fprintf(stderr, "                     [min, avg, max]:\n");
    fprintf(stderr, "original particles = [%d, %d, %d]\n",
            global_min_quants[NUM_ORIG_PTS],
            global_sum_quants[NUM_ORIG_PTS] / global_sum_quants[NUM_LOC_BLOCKS],
            global_max_quants[NUM_ORIG_PTS]);
    fprintf(stderr, "with ghosts        = [%d, %d, %d]\n",
            global_min_quants[NUM_FINAL_PTS],
            global_sum_quants[NUM_FINAL_PTS] / global_sum_quants[NUM_LOC_BLOCKS],
            global_max_quants[NUM_FINAL_PTS]);
    fprintf(stderr, "tets               = [%d, %d, %d]\n",
            global_min_quants[NUM_TETS],
            global_sum_quants[NUM_TETS] / global_sum_quants[NUM_LOC_BLOCKS],
            global_max_quants[NUM_TETS]);
    fprintf(stderr, "-------------------------------------------------\n");
  }
}
//
// for each vertex saves a tet that contains it
//
void fill_vert_to_tet(dblock_t* dblock)
{
  //fprintf(stderr, "fill_vert_to_tet(): %d %d\n", dblock->num_particles, dblock->num_tets);

  dblock->vert_to_tet =
    (int*)realloc(dblock->vert_to_tet, sizeof(int) * dblock->num_particles);

  for (int p = 0; p < dblock->num_particles; ++p)
      dblock->vert_to_tet[p] = -1;

  for (int t = 0; t < dblock->num_tets; ++t)
  {
    for (int v = 0; v < 4; ++v)
    {
      int p = dblock->tets[t].verts[v];
      //if (p >= dblock->num_particles || p < 0)
      //  fprintf(stderr, "Warning: %d is out of bounds!\n", p);
      dblock->vert_to_tet[p] = t;       // the last one wins
    }
  }
}
//
// starts / stops timing
// (does a barrier on comm)
//
// times: timing data
// start: index of timer to start (-1 if not used)
// stop: index of timer to stop (-1 if not used)
//
void timing(double* times, int start, int stop, MPI_Comm comm)
{
  if (start < 0 && stop < 0)
  {
    for (int i = 0; i < TESS_MAX_TIMES; i++)
      times[i] = 0.0;
  }

#ifdef TIMING

  MPI_Barrier(comm);
  if (start >= 0)
    times[start] = MPI_Wtime();
  if (stop >= 0)
    times[stop] = MPI_Wtime() - times[stop];

#endif
}
//
// memory profile, prints max reseident usage of all procs
//
void get_mem(int breakpoint, MPI_Comm comm)
{
#ifdef MEMORY

  int rank;
  MPI_Comm_rank(comm, &rank);

#ifdef BGQ

  uint64_t shared, persist, heapavail, stackavail, stack, heap, heapmax, guard, mmap;

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
             comm);
  if (rank == 0)
    fprintf(stderr, "%d: BGQ max memory = %.0lf MB\n",
            breakpoint, max_heap_mem);

#else // !BGQ

  struct rusage r_usage;
  getrusage(RUSAGE_SELF, &r_usage);

#ifdef __APPLE__
  const int to_mb = 1048576;
#else
  const int to_mb = 1024;
#endif // APPLE

  float res = r_usage.ru_maxrss;
  float mem = res / (float)to_mb;
  float max_mem;
  MPI_Reduce(&mem, &max_mem, 1, MPI_FLOAT, MPI_MAX, 0, comm);
  if (rank == 0)
    fprintf(stderr, "%d: max memory = %0.1f MB\n", breakpoint, max_mem);

#endif // BGQ

#endif // MEMORY
}
// ---------------------------------------------------------------------------
//
// functions from old tess version that have not yet been converted to new version
//
// --------------------------------------------------------------------------

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
  DIY_Init(3, 1, comm);
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
//   nblocks: local number of blocks
//   particles: particles[block_num][particle]
//   where each particle is 3 values, px, py, pz
//   num_particles; number of particles in each block
//   out_file: output file name
void tess(int nblocks, float **particles, int *num_particles, char *out_file) {

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
// CLP
//
//   Add wall particles
//
void wall_particles(struct dblock_t *dblock)
{
  //   init using data_mins and data_maxs
  //   currently assuimg walls on all sides, but format can easily be
  //   modified to be ANY set of walls
  struct wall_t*  walls     = NULL;
  int             num_walls = 0;
  create_walls(&num_walls,&walls);
  int* wall_cut = new int[num_walls];
  std::vector<float> new_points;

  // Find all particles that need to be mirrored.
  for (int p = 0; p < dblock->num_orig_particles; ++p)
  {
    //  zero generate-wall-point array (length of number of walls)
    memset(wall_cut, 0, num_walls*sizeof(int));

    // Determine if the cell is complete
    vector< pair<int, int> > nbrs;
    bool finite = neighbor_edges(nbrs, p, dblock->tets, dblock->vert_to_tet[p]);

    if (!finite)
      //  set the mirror-generate array to all ones
      // (extra calculations but simpler to assume!)
      memset(wall_cut, 1, num_walls*sizeof(int));
    else
    {
      // loop throug the list of all the Voronoi cell vertices of the point.
      // See if any are outside a wall.

      // neighbor edges a vector of (vertex u, tet of vertex u) pairs
      // that neighbor vertex v

      // the following loop is the equivalent of
      // for all faces in a voronoi cell
      for (int i = 0; i < (int)nbrs.size(); ++i)
      {
        // get edge link
        std::vector<int> edge_link;
        fill_edge_link(edge_link, p, nbrs[i].first, nbrs[i].second, dblock->tets);

        // following is equivalent of all vertices in a face
        for (int j = 0; j < (int)edge_link.size(); ++j)
        {
          float pt[3];
          circumcenter(pt,&(dblock->tets[edge_link[j]]), dblock->particles);
          for (int wi = 0; wi < num_walls; ++wi)
            if (!wall_cut[wi])
              wall_cut[wi] = test_outside(pt,&walls[wi]);
        }
      }
    }

    // Make the mirrored particles
    // For each mirror-generate index that is 1, generate the mirror point given site rp
    // and the wall
    // Create list of points
    for (int wi = 0; wi < num_walls; ++wi)
    {
      if (wall_cut[wi])
      {
        float rpt[3];
        generate_mirror(rpt,&dblock->particles[3*p],&walls[wi]);
        new_points.insert(new_points.end(), rpt, rpt+3);
      }
    }
  }

  // Add all the new points to the dblock.
  if (!new_points.empty())
  {
    size_t realloc_size;

    // grow space
    realloc_size      = dblock->num_particles * 3 + new_points.size();
    dblock->particles = (float*) realloc(dblock->particles, realloc_size * sizeof(float));

    realloc_size      = dblock->num_particles + new_points.size() / 3 - dblock->num_orig_particles;
    dblock->rem_gids  = (int*) realloc(dblock->rem_gids, realloc_size * sizeof(int));
    dblock->rem_lids  = (int*) realloc(dblock->rem_lids, realloc_size * sizeof(int));

    // copy new particles
    for (size_t j = 0; j < new_points.size(); j += 3)
    {
      dblock->particles[3 * dblock->num_particles    ] = new_points[j    ];
      dblock->particles[3 * dblock->num_particles + 1] = new_points[j + 1];
      dblock->particles[3 * dblock->num_particles + 2] = new_points[j + 2];
      dblock->rem_gids[dblock->num_particles - dblock->num_orig_particles] = -1;
      dblock->rem_lids[dblock->num_particles - dblock->num_orig_particles] = -1;
      dblock->num_particles++;
    }
  }

  // cleanup
  delete[] wall_cut;
  destroy_walls(num_walls, walls);
}

#endif

// --------------------------------------------------------------------------
//
//   randomly samples a set of particles
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
#if 0
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
#endif
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

#if 0

// CLP
//
//   creates and initializes walls
//
//   walls: pointer to array of walls
//
//   Important!  [a,b,c] must be a unit length vector!
//
//    allocate blocks and headers
//
//
void create_walls(int *num_walls, struct wall_t **walls)
{
  *num_walls  = 6;
  *walls      = new struct wall_t[*num_walls];

  float epsilon = 1e-6f; // TODO parameter/autoadaptive

  // bottom xy wall
  (*walls)[0].a = 0;
  (*walls)[0].b = 0;
  (*walls)[0].c = 1;
  (*walls)[0].d = - data_mins[2] - epsilon;

  // forward xz wall
  (*walls)[1].a = 0;
  (*walls)[1].b = 1;
  (*walls)[1].c = 0;
  (*walls)[1].d = -data_mins[1] - epsilon;

  // left yz wall
  (*walls)[2].a = 1;
  (*walls)[2].b = 0;
  (*walls)[2].c = 0;
  (*walls)[2].d = -data_mins[0] - epsilon;

  // top xy wall
  (*walls)[3].a = 0;
  (*walls)[3].b = 0;
  (*walls)[3].c = -1;
  (*walls)[3].d = data_maxs[2] + epsilon;

  // back xz wall
  (*walls)[4].a = 0;
  (*walls)[4].b = -1;
  (*walls)[4].c = 0;
  (*walls)[4].d = data_maxs[1] + epsilon;

  // right yz wall
  (*walls)[5].a = -1;
  (*walls)[5].b = 0;
  (*walls)[5].c = 0;
  (*walls)[5].d = data_maxs[0] + epsilon;

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

#endif
// ---------------------------------------------------------------------------
