#include <vector>

#include <diy/reduce.hpp>
#include <diy/partners/swap.hpp>

#include "tess/tess.hpp"

void redistribute(void* b_, const diy::ReduceProxy& srp, const diy::RegularSwapPartners& partners);

void tess_exchange(diy::Master& master, const diy::Assigner& assigner, double* times)
{
  timing(times, EXCH_TIME, -1, master.communicator());
  int k = 2;
  diy::RegularSwapPartners  partners(3, assigner.nblocks(), k, false);
  diy::reduce(master, assigner, partners, &redistribute);
  timing(times, -1, EXCH_TIME, master.communicator());
}

void tess_exchange(diy::Master& master, const diy::Assigner& assigner)
{
  double times[TESS_MAX_TIMES];
  tess_exchange(master, assigner, times);
}

void redistribute(void* b_, const diy::ReduceProxy& srp, const diy::RegularSwapPartners& partners)
{
    dblock_t*                   b        = static_cast<dblock_t*>(b_);
    unsigned                    round    = srp.round();

    //fprintf(stderr, "in_link.size():  %d\n", srp.in_link().size());
    //fprintf(stderr, "out_link.size(): %d\n", srp.out_link().size());

    // step 1: dequeue and merge
    // dequeue all the incoming points and add them to this block's vector
    // could use srp.incoming() instead
    for (unsigned i = 0; i < srp.in_link().size(); ++i)
    {
      int nbr_gid = srp.in_link().target(i).gid;
      if (nbr_gid == srp.gid())
          continue;

      std::vector<float>    in_points;
      srp.dequeue(nbr_gid, in_points);
      int npts = in_points.size() / 3;

      //fprintf(stderr, "[%d] Received %d points from [%d]\n", srp.gid(), npts, nbr_gid);
      b->particles = (float *)realloc(b->particles, (b->num_particles + npts) * 3 * sizeof(float));
      size_t o = b->num_particles * 3;
      for (size_t j = 0; j < in_points.size(); ++j)
	b->particles[o++] = in_points[j];
      b->num_particles += npts;
    }
    b->num_orig_particles = b->num_particles;

    // step 2: subset and enqueue
    //fprintf(stderr, "[%d] out_link().size(): %d\n", srp.gid(), srp.out_link().size());
    if (srp.out_link().size() == 0)        // final round; nothing needs to be sent
        return;

    std::vector< std::vector<float> > out_points(srp.out_link().size());
    int group_size = srp.out_link().size();
    int cur_dim    = partners.dim(round);
    for (size_t i = 0; i < b->num_particles; ++i)
    {
      int loc = floor((b->particles[3*i + cur_dim] - b->box.min[cur_dim]) / (b->box.max[cur_dim] - b->box.min[cur_dim]) * group_size);
      if (loc >= out_points.size() || loc < 0)
	fprintf(stderr, "Warning: loc=%d >= %lu : %f vs [%f,%f]\n",
			loc, out_points.size(),
			b->particles[3*i + cur_dim],
			b->box.min[cur_dim], b->box.max[cur_dim]
		);
      if (loc == out_points.size())
	loc -= 1;

      out_points[loc].push_back(b->particles[3*i]);
      out_points[loc].push_back(b->particles[3*i + 1]);
      out_points[loc].push_back(b->particles[3*i + 2]);
    }
    int pos = -1;
    for (int i = 0; i < group_size; ++i)
    {
      if (srp.out_link().target(i).gid == srp.gid())
      {
	b->particles	 = (float *)realloc(b->particles, out_points[i].size() * sizeof(float));
	for (size_t j = 0; j < out_points[i].size(); ++j)
	  b->particles[j] = out_points[i][j];
	b->num_particles = out_points[i].size() / 3;
	b->num_orig_particles = b->num_particles;
        pos = i;
      }
      else
      {
        srp.enqueue(srp.out_link().target(i), out_points[i]);
        //fprintf(stderr, "[%d] Sent %d points to [%d]\n", srp.gid(), (int) out_points[i].size() / 3, srp.out_link().target(i).gid);
      }
    }
    float new_min = b->box.min[cur_dim] + (b->box.max[cur_dim] - b->box.min[cur_dim])/group_size*pos;
    float new_max = b->box.min[cur_dim] + (b->box.max[cur_dim] - b->box.min[cur_dim])/group_size*(pos + 1);
    b->box.min[cur_dim] = new_min;
    b->box.max[cur_dim] = new_max;
}
