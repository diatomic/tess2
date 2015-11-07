#include <vector>
#include <cstdio>

#include <diy/algorithms.hpp>

#include "tess/tess.h"
#include "tess/delaunay.h"

struct KDTreeBlock
{
  struct Point
  {
    float&	    operator[](unsigned i)		{ return data[i]; }
    const float&    operator[](unsigned i) const	{ return data[i]; }
    float	    data[3];
  };
  std::vector<Point>		    points;
};

struct WrapMaster
{
    diy::Master* master;
    bool	 wrap;
};

void populate_kdtree_block(dblock_t* d, const diy::Master::ProxyWithLink& cp, void* aux_)
{
  WrapMaster*   aux = (WrapMaster*) aux_;
  diy::Master*	kdtree_master = aux->master;
  bool		wrap	      = aux->wrap;

  diy::ContinuousBounds domain = d->data_bounds;

  KDTreeBlock* b = new KDTreeBlock;
  diy::RegularContinuousLink* l = new diy::RegularContinuousLink(3, domain, domain);
  if (wrap)
  {
      // link to self in every direction
      for (int j = 0; j < 3; ++j)
          for (int k = 0; k < 2; ++k)
          {
              diy::BlockID nbr = { cp.gid(), cp.master()->communicator().rank() };
              l->add_neighbor(nbr);

              diy::Direction dir = static_cast<diy::Direction>(1 << (2*j + k));
              l->add_direction(dir);

              l->add_bounds(domain);
          }
  }
  kdtree_master->add(cp.gid(), b, l);

  // copy the particles over
  b->points.resize(d->num_orig_particles);
  for (size_t i = 0; i < d->num_orig_particles; ++i)
  {
    b->points[i][0] = d->particles[3*i + 0];
    b->points[i][1] = d->particles[3*i + 1];
    b->points[i][2] = d->particles[3*i + 2];
  }
}

void extract_kdtree_block(KDTreeBlock* b, const diy::Master::ProxyWithLink& cp, void* aux)
{
  diy::Master*	tess_master = (diy::Master*) aux;

  int           tess_lid = tess_master->lid(cp.gid());
  dblock_t*	d        = (dblock_t*) tess_master->block(tess_lid);	// assumes all the blocks are in memory

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
  diy::RegularContinuousLink* tess_link   = static_cast<diy::RegularContinuousLink*>(tess_master->link(tess_lid));
  diy::RegularContinuousLink* kdtree_link = static_cast<diy::RegularContinuousLink*>(cp.link());
  tess_link->swap(*kdtree_link);

  d->box  = tess_link->bounds();
  for (int i = 0; i < 3; ++i)
  {
    d->mins[i] = tess_link->bounds().min[i];
    d->maxs[i] = tess_link->bounds().max[i];
  }

  delete b;     // safe to do since kdtree_master doesn't own the blocks (no create/destroy supplied)
}

void tess_kdtree_exchange(diy::Master& master, const diy::Assigner& assigner, double* times, bool wrap)
{
  timing(times, EXCH_TIME, -1, master.communicator());

  diy::Master kdtree_master(master.communicator(),  master.threads(), -1);
  WrapMaster wrap_master = { &kdtree_master, wrap };
  master.foreach<dblock_t>(&populate_kdtree_block, &wrap_master);

  int bins = 1024;	// histogram bins; TODO: make a function argument
  diy::ContinuousBounds domain = master.block<dblock_t>(master.loaded_block())->data_bounds;
  diy::kdtree(kdtree_master, assigner, 3, domain, &KDTreeBlock::points, bins, wrap);

  kdtree_master.foreach<KDTreeBlock>(&extract_kdtree_block, &master);
  master.set_expected(kdtree_master.expected());

  timing(times, -1, EXCH_TIME, master.communicator());
}

void tess_kdtree_exchange(diy::Master& master, const diy::Assigner& assigner, bool wrap)
{
  double times[TESS_MAX_TIMES];
  tess_kdtree_exchange(master, assigner, times, wrap);
}
