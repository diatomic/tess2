// ---------------------------------------------------------------------------
//
//   parallel density estimator
//
//   Tom Peterka
//   Argonne National Laboratory
//   9700 S. Cass Ave.
//   Argonne, IL 60439
//   tpeterka@mcs.anl.gov
//
//   (C) 2015 by UChicago Argonne, LLC
//   See COPYING in top-level directory.
//
// --------------------------------------------------------------------------

#include "tess/dense.hpp"
#include <diy/point.hpp>
#ifndef TESS_NO_OPENMP
#include <omp.h>
#endif

using namespace std;

// debug: consistency checks and output stats
static float max_dense = 0.0;
static double tot_mass = 0.0; // total output mass
static float check_mass = 0.0; // ground truth total mass

// density estimator
void dense(alg alg_type,              // algorithm DENSE_TESS, DENSE_CIC
           int num_given_bounds,      // number of given physical bounds of grid
           float *given_mins,         // given physical bounds of grid (x,y,z)
	   float *given_maxs,
           bool project,              // whether to project to 2D
           float *proj_plane,         // normal to projection plane (x,y,z)
           float mass,                // mass of one particle
           float *data_mins,          // global data physicsl extents (x,y,z) (output)
           float *data_maxs,
           float *grid_phys_mins,     // global grid physical extents (x,y,z) (output)
           float *grid_phys_maxs,
	   float *grid_step_size,     // physical size of grid space (x,y,z) (output)
           float eps,                 // floating point error threshold
           int *glo_num_idx,          // global number of grid points (i,j,k)
           diy::Master& master)       // diy master object
{
  // local block grid parameters
  int block_min_idx[3];               // global grid index of block minimum grid point
  int block_max_idx[3];               // global grid index of block maximum grid point
  int block_num_idx[3];               // number of grid points in local block

  // find global data bounds
  // TODO: needs to be a foreach function, currently assumes all blocks in memory
  DataBounds(data_mins, data_maxs, master);

  // find grid bounds and step size
  GridStepParams(num_given_bounds, given_mins, given_maxs, data_mins, data_maxs, grid_phys_mins,
                 grid_phys_maxs, grid_step_size, glo_num_idx);

  // initialize auxiliary args for foreach functions
  args_t args;
  args.alg_type = alg_type;
  args.project = project;
  args.proj_plane[0]     = proj_plane[0];
  args.proj_plane[1]     = proj_plane[1];
  args.proj_plane[2]     = proj_plane[2];
  args.mass              = mass;
  args.data_mins[0]      = data_mins[0];
  args.data_mins[1]      = data_mins[1];
  args.data_mins[2]      = data_mins[2];
  args.data_maxs[0]      = data_maxs[0];
  args.data_maxs[1]      = data_maxs[1];
  args.data_maxs[2]      = data_maxs[2];
  args.grid_phys_mins[0] = grid_phys_mins[0];
  args.grid_phys_mins[1] = grid_phys_mins[1];
  args.grid_phys_mins[2] = grid_phys_mins[2];
  args.grid_step_size[0] = grid_step_size[0];
  args.grid_step_size[1] = grid_step_size[1];
  args.grid_step_size[2] = grid_step_size[2];
  args.eps               = eps;
  args.glo_num_idx[0]    = glo_num_idx[0];
  args.glo_num_idx[1]    = glo_num_idx[1];
  args.glo_num_idx[2]    = glo_num_idx[2];

  // allocate and initialize density field
  master.foreach([&](DBlock* b, const diy::Master::ProxyWithLink& cp)
                 { init_dense(b, cp, &args); });

  // divisor for volume (3d density) or area (2d density)
  // assumes projection is to x-y plane
  args.div = (project ? grid_step_size[0] * grid_step_size[1] :
              grid_step_size[0] * grid_step_size[1] * grid_step_size[2]);

  // estimate density
  master.foreach([&](DBlock* b, const diy::Master::ProxyWithLink& cp)
                 { est_dense(b, cp, &args); });

  // exchange grid points
  master.exchange();

  // process received points
  master.foreach([&](DBlock* b, const diy::Master::ProxyWithLink& cp)
                 { recvd_pts(b, cp, &args); });
}

// foreach block function to initialize density
void init_dense(DBlock*                         b,
                const diy::Master::ProxyWithLink& cp,
                args_t*                           a)
{
  // local block grid parameters
  int block_min_idx[3];                    // global grid index of block minimum grid point
  int block_max_idx[3];                    // global grid index of block maximum grid point
  int block_num_idx[3];                    // number of grid points in local block
  BlockGridParams(b, block_min_idx, block_max_idx, block_num_idx, a->grid_phys_mins,
                  a->grid_step_size, a->eps, a->data_mins, a->data_maxs, a->glo_num_idx);

  int npts;                                // total number of points in the block
  if (a->project)
    npts = block_num_idx[0] * block_num_idx[1];
  else
    npts = block_num_idx[0] * block_num_idx[1] * block_num_idx[2];
  b->density = new float[npts];
  b->num_grid_pts = npts;

  // init density
  memset(b->density, 0 , npts * sizeof(float));
}

// foreach block function to estimate density
void est_dense(DBlock*                         b,
                const diy::Master::ProxyWithLink& cp,
                args_t*                           a)
{
  // local block grid parameters
  int block_min_idx[3];                    // global grid index of block minimum grid point
  int block_max_idx[3];                    // global grid index of block maximum grid point
  int block_num_idx[3];                    // number of grid points in local block
  BlockGridParams(b, block_min_idx, block_max_idx, block_num_idx, a->grid_phys_mins,
                  a->grid_step_size, a->eps, a->data_mins, a->data_maxs, a->glo_num_idx);

  // iterate over cells, distributing density onto grid points
  switch (a->alg_type)
  {
  case DENSE_TESS:
#if 0
    // tess-based multithread estimator
    IterateCellsOMP(b, block_min_idx, block_num_idx, a->project, a->proj_plane, a->grid_phys_mins,
                    a->grid_step_size, a->data_mins, a->data_maxs, a->eps, a->mass, cp);
#else
    // tess-based single-thread estimator
    IterateCells(b, block_min_idx, block_num_idx, a->project, a->proj_plane, a->grid_phys_mins,
                 a->grid_step_size, a->data_mins, a->data_maxs, a->eps, a->mass, cp);
#endif
    break;
  case DENSE_CIC:
    // CIC-based estimator (only single threaded for now)
    IterateCellsCic(b, block_min_idx, block_num_idx, a->project, a->proj_plane, a->grid_phys_mins,
                    a->grid_step_size, a->data_maxs, a->eps, a->mass, cp);
    break;
  default:
    break;
  }
}

// foreach block function to receive points
void recvd_pts(DBlock*                         b,
               const diy::Master::ProxyWithLink& cp,
               args_t*                           a)
{
  diy::Link*       l    = cp.link();
  std::vector<int> in;                     // gids of sources

  cp.incoming(in);

  // local block grid parameters
  int block_min_idx[3];                    // global grid index of block minimum grid point
  int block_max_idx[3];                    // global grid index of block maximum grid point
  int block_num_idx[3];                    // number of grid points in local block
  BlockGridParams(b, block_min_idx, block_max_idx, block_num_idx, a->grid_phys_mins,
                  a->grid_step_size, a->eps, a->data_mins, a->data_maxs, a->glo_num_idx);

  for (size_t i = 0; i < in.size(); i++)   // links
  {
    int numpts = cp.incoming(in[i]).buffer.size() / sizeof(grid_pt_t);
    vector<grid_pt_t> grid_pts(numpts);
    cp.dequeue(in[i], &grid_pts[0], numpts);
    for (size_t j = 0; j < numpts; j++)    // items in the link
    {
      // assign the density in the local block array
      int block_grid_idx[3]; // indices in local block array
      Global2LocalIdx(grid_pts[j].idx, block_grid_idx, block_min_idx);
      int idx = index(block_grid_idx, block_num_idx, a->project, a->proj_plane);
      b->density[idx] += (grid_pts[j].mass / a->div);

      // debug
      tot_mass += grid_pts[j].mass;
      if (b->density[idx] > max_dense)
        max_dense = b->density[idx];

    }
  }
}

// iterate over cells and assign single density to grid point
// single thread version
//
// block: local block
// block_min_idx: minimum (i,j,k) grid point index in block
// block_num_idx: number of grid points in block (x,y,z)
// project: whether to project to 2D
// proj_plane: normal to projection plane (x,y,z)
// grid_phys_mins: physical global min grid corner position (x,y,z)
// grid_step_size: physical size of one grid space (x,y,z)
// data_mins, data_maxs: global data physical extent (x,y,z)
// eps: floating point error tolerance
// mass: mass of 1 particle
// cp: communication proxy
//
// side effects: writes density or sends to neighbors
void IterateCells(DBlock* block,
                  int *block_min_idx,
                  int *block_num_idx,
                  bool project,
                  float *proj_plane,
                  float *grid_phys_mins,
                  float *grid_step_size,
		  float *data_mins,
                  float *data_maxs,
                  float eps,
                  float mass,
                  const diy::Master::ProxyWithLink& cp)
{
  int alloc_grid_pts = 0;                       // number of grid points allocated
  grid_pt_t *grid_pts = NULL;                   // grid points covered by the cell
  int *border = NULL;                           // cell border, min,max x index for each y, z index
  int num_grid_pts;                             // number of grid points
  RCLink* l = dynamic_cast<RCLink*>(cp.link()); // link to block neighbors

  // divisor for volume (3d density) or area (2d density)
  // assumes projection is to x-y plane
  float div = (project ? grid_step_size[0] * grid_step_size[1] :
	       grid_step_size[0] * grid_step_size[1] * grid_step_size[2]);

  // cells
  for (int cell = 0; cell < block->num_orig_particles; cell++)
  {
    float cell_min[3], cell_max[3]; // cell bounds
    float grid_pos[3]; // physical position of grid point

    // skip inccomplete cells
    if (block->vert_to_tet[cell] == -1 ||
        !complete(cell, block->tets, block->num_tets, block->vert_to_tet[cell]))
      continue;

    vector <float> normals; // cell normals
    vector <vector <float> > face_verts; // vertex positions in each face

    // cell bounds
    CellBounds(block, cell, cell_min, cell_max, normals, face_verts);

    // grid points covered by this cell
    num_grid_pts = CellGridPts(cell_min, cell_max, grid_pts, border,
			       alloc_grid_pts, normals, face_verts, data_mins,
			       data_maxs, grid_phys_mins, grid_step_size,
			       mass, eps, &(block->particles[3 * cell]));

    if (!num_grid_pts) // cell outside of global data bounds
      continue;

    // debug: check consistency
    check_mass++;

    // grid points covered by cell
    for (int i = 0; i < num_grid_pts; i++)
    {
      idx2phys(grid_pts[i].idx, grid_pos, grid_step_size, grid_phys_mins);

      // assign density to grid points in the block
      if (grid_pos[0] >= block->bounds.min[0] &&
	  grid_pos[0] <= block->bounds.max[0] &&
	  grid_pos[1] >= block->bounds.min[1] &&
	  grid_pos[1] <= block->bounds.max[1] &&
	  grid_pos[2] >= block->bounds.min[2] &&
          grid_pos[2] <= block->bounds.max[2])
      {
	// assign the density to the local block density array
	int block_grid_idx[3]; // local block idx of grid point
	Global2LocalIdx(grid_pts[i].idx, block_grid_idx, block_min_idx);
	int idx = index(block_grid_idx, block_num_idx, project, proj_plane);
	block->density[idx] += (grid_pts[i].mass / div);

	// consistency checks and stats
	tot_mass += grid_pts[i].mass;
	if (block->density[idx] > max_dense)
	  max_dense = block->density[idx];
      }

      // or send grid points to neighboring blocks
      else
      {
        set<int> dests; // destination neighbor edges for this point
        in(*l, diy::Point<float,3> { grid_pos }, std::inserter(dests, dests.end()), block->data_bounds);
        for (set<int>::iterator it = dests.begin(); it != dests.end(); it++)
          cp.enqueue(l->target(*it), grid_pts[i]);
      }
    } // grid points covered by cell
  } // cells

  if (grid_pts)
    free(grid_pts);
  if (border)
    free(border);
}

#ifndef TESS_NO_OPENMP

// iterate over cells and assign single density to grid point
// openMP version
//
// block: local block
// block_min_idx: minimum (i,j,k) grid point index in block
// block_num_idx: number of grid points in block (x,y,z)
// project: whether to project to 2D
// proj_plane: normal to projection plane (x,y,z)
// grid_phys_mins: physical global min grid corner position (x,y,z)
// grid_step_size: physical size of one grid space (x,y,z)
// data_mins, data_maxs: global data physical extent (x,y,z)
// eps: floating point error tolerance
// mass: mass of 1 particle
// cp: communication proxy
//
// side effects: writes density or sends to neighbors
void IterateCellsOMP(DBlock* block,
                     int *block_min_idx,
                     int *block_num_idx,
                     bool project,
                     float *proj_plane,
                     float *grid_phys_mins,
                     float *grid_step_size,
		     float *data_mins,
                     float *data_maxs,
                     float eps,
                     float mass,
                     const diy::Master::ProxyWithLink& cp)
{
  int nthreads;                                 // number of threads currently being used
  int mthreads = omp_get_max_threads();         // max threads that could be used
  vector<grid_pt_t> enq_grid_pts[mthreads];     // enqueued grid pts for each thread
  RCLink* l = dynamic_cast<RCLink*>(cp.link()); // link to block neighbors

  // divisor for volume (3d density) or area (2d density)
  // assumes projection is to x-y plane
  float div = (project ? grid_step_size[0] * grid_step_size[1] :
	       grid_step_size[0] * grid_step_size[1] * grid_step_size[2]);

  omp_set_num_threads(8);  // number of threads for BGQ must be set manually, 8 threads * 8 ppn

#pragma omp parallel
  {
    nthreads = omp_get_num_threads();

    // debug
    // fprintf(stderr, "using OpenMP version, num threads = %d\n", nthreads);

    // objects defined inside the thread block are private to the thread
    int alloc_grid_pts = 0; // number of grid points allocated
    grid_pt_t *grid_pts = NULL; // grid points covered by the cell
    int *border = NULL; // cell border, min and max x index for each y, z index
    int num_grid_pts; // number of grid points
    int tid = omp_get_thread_num(); // thread id

#pragma omp for
    // cells
    for (int cell = 0; cell < block->num_orig_particles; cell++)
    {
      float cell_min[3], cell_max[3]; // cell bounds
      float grid_pos[3]; // physical position of grid point

      // skip inccomplete cells
      if (!complete(cell, block->tets, block->num_tets,
		    block->vert_to_tet[cell]))
	continue;

      vector <float> normals; // cell normals
      vector <vector <float> > face_verts; // vertex positions in each face

      // cell bounds
      CellBounds(block, cell, cell_min, cell_max, normals, face_verts);

      // grid points covered by this cell
      num_grid_pts = CellGridPts(cell_min, cell_max, grid_pts, border,
				 alloc_grid_pts, normals, face_verts, data_mins,
				 data_maxs, grid_phys_mins, grid_step_size,
				 mass, eps, &(block->particles[3 * cell]));

      if (!num_grid_pts) // cell outside of global data bounds
	continue;

      // debug: consistency check
#pragma omp atomic
      check_mass++;

      // iterate over grid points covered by cell
      for (int i = 0; i < num_grid_pts; i++)
      {
	idx2phys(grid_pts[i].idx, grid_pos, grid_step_size, grid_phys_mins);

        // assign density to grid points in the block
        if (grid_pos[0] >= block->bounds.min[0] &&
            grid_pos[0] <= block->bounds.max[0] &&
            grid_pos[1] >= block->bounds.min[1] &&
            grid_pos[1] <= block->bounds.max[1] &&
            grid_pos[2] >= block->bounds.min[2] &&
            grid_pos[2] <= block->bounds.max[2])
        {
	  // assign the density to the local block density array
	  int block_grid_idx[3]; // local block idx of grid point
	  Global2LocalIdx(grid_pts[i].idx, block_grid_idx, block_min_idx);
	  int idx = index(block_grid_idx, block_num_idx, project, proj_plane);
#pragma omp atomic
	  block->density[idx] += (grid_pts[i].mass / div);

	  // consistency check and output stats
#pragma omp atomic // only the next statement is atomic
	  tot_mass += grid_pts[i].mass;
	  if (block->density[idx] > max_dense)
	    max_dense = block->density[idx];
	}

	// or send grid points to neighboring blocks
	else
	  enq_grid_pts[tid].push_back(grid_pts[i]);
      } // grid points covered by cell
    } // cells

    if (grid_pts)
      free(grid_pts);
    if (border)
      free(border);

  } // parallel block

  // send grid points to neighboring blocks
  float grid_pos[3]; // physical position of grid point
  for (int i = 0; i < nthreads; i++)
  {
    for (int j = 0; j < (int)(enq_grid_pts[i].size()); j++)
    {
      idx2phys(enq_grid_pts[i][j].idx, grid_pos, grid_step_size,
	       grid_phys_mins);
        set<int> dests; // destination neighbor edges for this point
        in(*l, grid_pos, std::inserter(dests, dests.end()), block->data_bounds);
        for (set<int>::iterator it = dests.begin(); it != dests.end(); it++)
          cp.enqueue(l->target(*it), enq_grid_pts[i][j]);
    }
  }

  // clean up grid enqueued grid points todo: are they freed automatically
  // when the array of vectors goes out of scope?
  for (int i = 0; i < nthreads; i++)
    enq_grid_pts[i].clear();
}

#endif

// iterate over cells and assigns density to grid points
//  to grid points within a window size of one grid space, ie, CIC for the
//  8 grid points of a cell (vertex centered),
//  equivalent to CIC for 8 neighboring cells (cell-centered)
//
//  Note that we are only using the site (original particle position)
//   from the cell, ignoring rest of voronoi cell for CIC
//
// block: local block
// block_min_idx: minimum (i,j,k) grid point index in block
// block_num_idx: number of grid points in block (output) (x,y,z)
// project: whether to project to 2D
// proj_plane: normal to projection plane (x,y,z)
// grid_phys_mins: physical global min grid corner position (x,y,z)
// grid_step_size: physical size of one grid space (x,y,z)
// eps: floating point error tolerance
// mass: mass of 1 particle
// cp: communication proxy
//
// side effects: writes density or sends to neighbors
void IterateCellsCic(DBlock* block,
                     int *block_min_idx,
                     int *block_num_idx,
                     bool project,
                     float *proj_plane,
                     float *grid_phys_mins,
                     float *grid_step_size,
		     float *data_maxs,
                     float eps,
                     float mass,
                     const diy::Master::ProxyWithLink& cp)
{
  float grid_pos[3];                            // physical position of grid point
  RCLink* l = dynamic_cast<RCLink*>(cp.link()); // link to block neighbors

  // divisor for volume (3d density) or area (2d density)
  // assumes projection is to x-y plane
  float div = (project ? grid_step_size[0] * grid_step_size[1] :
	       grid_step_size[0] * grid_step_size[1] * grid_step_size[2]);

  // cells
  for (int cell = 0; cell < block->num_orig_particles; cell++)
  {
    // consitency check
    check_mass++;

    // distribute mass at cell site to neighboring grid points
    vector<int> grid_idxs; // grid idxs that get a fraction of the mass
    vector<float> grid_masses; // mass given to each grid_idx

    float *pt = &(block->particles[3 * cell]); // x,y,z of particle

    DistributeScalarCIC(pt, mass, grid_idxs, grid_masses, grid_step_size, grid_phys_mins, eps);

    assert((int)grid_idxs.size() / 3 == 8); // sanity

    // (8) grid points for this cell site
    for (int i = 0; i < (int)grid_idxs.size() / 3; i++)
    {
      idx2phys(&(grid_idxs[3 * i]), grid_pos, grid_step_size, grid_phys_mins);

      // assign density to grid points in the block
      if (grid_pos[0] >= block->bounds.min[0] &&
	  grid_pos[0] <= block->bounds.max[0] &&
	  grid_pos[1] >= block->bounds.min[1] &&
	  grid_pos[1] <= block->bounds.max[1] &&
	  grid_pos[2] >= block->bounds.min[2] &&
          grid_pos[2] <= block->bounds.max[2])
      {
	// assign the density to the local block density array
	int block_grid_idx[3]; // local block idx of grid point
	Global2LocalIdx(&(grid_idxs[3 * i]), block_grid_idx, block_min_idx);
	int idx = index(block_grid_idx, block_num_idx, project, proj_plane);
	block->density[idx] += (grid_masses[i] / div);

	// consistency checks and output stats
	tot_mass += grid_masses[i];
	if (block->density[idx] > max_dense)
	  max_dense = block->density[idx];
      }

      // or send grid points to neighboring blocks
      else
      {
	grid_pt_t grid_pt;
	grid_pt.idx[0] = grid_idxs[3 * i];
	grid_pt.idx[1] = grid_idxs[3 * i + 1];
	grid_pt.idx[2] = grid_idxs[3 * i + 2];
	grid_pt.mass = grid_masses[i];
        set<int> dests; // destination neighbor edges for this point
        in(*l, diy::Point<float,3> { grid_pos }, std::inserter(dests, dests.end()), block->data_bounds);
        for (set<int>::iterator it = dests.begin(); it != dests.end(); it++)
          cp.enqueue(l->target(*it), grid_pt);
      }
    } // (8) grid points for this cell site
  } // cells
}

// grid parameters of one local block
//
// dblock: block local id
// block_min_idx: global grid idx of block minimum grid point (output) (i,j,k)
// block_max_idx: global grid idx of block maximum grid point (output) (i,j,k)
// block_num_idx: number of grid points in block (output) (i,j,k)
// grid_phys_mins: physical min corner of global grid (x,y,z)
// grid_step_size: physical size of one grid space (x,y,z)
// eps: floating point error tolerance
// data_mins, data_maxs: physical global data extents (x,y,z)
// glo_num_idx: global grid size
void BlockGridParams(DBlock *dblock,
                     int *block_min_idx,
                     int *block_max_idx,
                     int *block_num_idx,
		     float *grid_phys_mins,
                     float *grid_step_size,
                     float eps,
                     float *data_mins,
                     float *data_maxs,
                     int *glo_num_idx)
{
  float pos[3]; // temporary position (x,y,z)

  // global grid index of block minimum grid point
  phys2idx(&(dblock->bounds.min[0]), block_min_idx, grid_step_size, grid_phys_mins);
  idx2phys(block_min_idx, pos, grid_step_size, grid_phys_mins);
  if (pos[0] < dblock->bounds.min[0])
    block_min_idx[0]++;
  if (pos[1] < dblock->bounds.min[1])
    block_min_idx[1]++;
  if (pos[2] < dblock->bounds.min[2])
    block_min_idx[2]++;
  idx2phys(block_min_idx, pos, grid_step_size, grid_phys_mins); // double check adjusted position
  assert(pos[0] >= dblock->bounds.min[0] && pos[1] >= dblock->bounds.min[1] &&
	 pos[2] >= dblock->bounds.min[2]);

  // global grid index of block maximum grid point
  phys2idx(&(dblock->bounds.max[0]), block_max_idx, grid_step_size, grid_phys_mins);
  idx2phys(block_max_idx, pos, grid_step_size, grid_phys_mins);
  if (pos[0] + grid_step_size[0] <= dblock->bounds.max[0])
    block_max_idx[0]++;
  if (pos[1] + grid_step_size[1] <= dblock->bounds.max[1])
    block_max_idx[1]++;
  if (pos[2] + grid_step_size[2] <= dblock->bounds.max[2])
    block_max_idx[2]++;
  idx2phys(block_max_idx, pos, grid_step_size, grid_phys_mins); // double check adjusted position
  assert(pos[0] <= dblock->bounds.max[0] && pos[1] <= dblock->bounds.max[1] &&
	 pos[2] <= dblock->bounds.max[2]);

  // eliminate duplication at the maximum block border
  if (fabs(data_mins[0] + block_max_idx[0] * grid_step_size[0] -
	   dblock->bounds.max[0]) < eps &&
      fabs(dblock->bounds.max[0] - data_maxs[0]) > grid_step_size[0])
    block_max_idx[0]--;
  if (fabs(data_mins[1] + block_max_idx[1] * grid_step_size[1] -
      dblock->bounds.max[1]) < eps &&
      fabs(dblock->bounds.max[1] - data_maxs[1]) > grid_step_size[1])
    block_max_idx[1]--;
  if (fabs(data_mins[2] + block_max_idx[2] * grid_step_size[2] -
      dblock->bounds.max[2]) < eps &&
      fabs(dblock->bounds.max[2] - data_maxs[2]) > grid_step_size[2])
    block_max_idx[2]--;

  // possibly extend minimum end of blacks at the minimum end of the domain
  if (fabs(dblock->bounds.min[0] - data_mins[0]) < grid_step_size[0])
    block_min_idx[0] = 0;
  if (fabs(dblock->bounds.min[1] - data_mins[1]) < grid_step_size[1])
    block_min_idx[1] = 0;
  if (fabs(dblock->bounds.min[2] - data_mins[2]) < grid_step_size[2])
    block_min_idx[2] = 0;

  // possibly extend maximum end of blacks at the maximum end of the domain
  if (fabs(dblock->bounds.max[0] - data_maxs[0]) < grid_step_size[0])
    block_max_idx[0] = glo_num_idx[0] - 1;
  if (fabs(dblock->bounds.max[1] - data_maxs[1]) < grid_step_size[1])
    block_max_idx[1] = glo_num_idx[1] - 1;
  if (fabs(dblock->bounds.max[2] - data_maxs[2]) < grid_step_size[2])
    block_max_idx[2] = glo_num_idx[2] - 1;

  // compute number of grid points in local block
  block_num_idx[0] = block_max_idx[0] - block_min_idx[0] + 1;
  block_num_idx[1] = block_max_idx[1] - block_min_idx[1] + 1;
  block_num_idx[2] = block_max_idx[2] - block_min_idx[2] + 1;
}

// get cell bounds, face vertices, and normals for all cell faces
//
// dblock: one delaunay block
// cell: current cell counter
// cell_min, cell_max: cell bounds (output)
// normals: face normals (nx_0,ny_0,nz_0,nx_1,ny_1,nz_1, ...) (output)
// face_verts: vertex positions for each face (output)
void CellBounds(DBlock *dblock,
                int cell,
                float *cell_min,
                float *cell_max,
		vector<float> &normals,
		vector <vector <float> > &face_verts)
{
  float n[3]; // face normal

  // a tet containing the cell site
  int t = dblock->vert_to_tet[cell];

  // neighbor edges is a vector of (vertex u, tet of vertex u) pairs
  // that neighbor vertex v
  vector< pair<int, int> > nbrs;
  bool finite = neighbor_edges(nbrs, cell, dblock->tets, t);

  // infinte cells should have been filtered by the caller
  assert(finite);

  // grow vectors to correct size, num_faces = nbrs.size()
  normals.reserve(3 * (int)nbrs.size());
  face_verts.resize((int)nbrs.size());

  // get cell bounds
  for (int k = 0; k < (int)nbrs.size(); k++) // faces
  {
    // get edge link
    int u  = nbrs[k].first;
    int ut = nbrs[k].second;
    std::vector<int> edge_link;
    fill_edge_link(edge_link, cell, u, ut, dblock->tets);

    for (int l = 0; l < (int)edge_link.size(); l++) // vertices
    {
      face_verts[k].reserve(3 * (int)edge_link.size());

      float vv[3]; // voronoi vertex position
      circumcenter(vv, &(dblock->tets[edge_link[l]]), dblock->particles);
      face_verts[k].push_back(vv[0]);
      face_verts[k].push_back(vv[1]);
      face_verts[k].push_back(vv[2]);

      // extrema for entire cell
      if ((k == 0 && l == 0) || vv[0] < cell_min[0])
	cell_min[0] = vv[0];
      if ((k == 0 && l == 0) || vv[0] > cell_max[0])
	cell_max[0] = vv[0];

      if ((k == 0 && l == 0) || vv[1] < cell_min[1])
	cell_min[1] = vv[1];
      if ((k == 0 && l == 0) || vv[1] > cell_max[1])
	cell_max[1] = vv[1];

      if ((k == 0 && l == 0) || vv[2] < cell_min[2])
	cell_min[2] = vv[2];
      if ((k == 0 && l == 0) || vv[2] > cell_max[2])
	cell_max[2] = vv[2];
    } // vertices

    // normal
    NewellNormal(&(face_verts[k][0]), edge_link.size(), n);
    // check sign of dot product of normal with vector from site
    // to first face vertex to see if normal has correct direction
    // want outward normal
    float v[3];
    v[0] = face_verts[k][0] - dblock->particles[3 * cell];
    v[1] = face_verts[k][1] - dblock->particles[3 * cell + 1];
    v[2] = face_verts[k][2] - dblock->particles[3 * cell + 2];
    if (v[0] * n[0] + v[1] * n[1] + v[2] * n[2] < 0.0)
    {
      n[0] *= -1.0;
      n[1] *= -1.0;
      n[2] *= -1.0;
    }
    normals.push_back(n[0]);
    normals.push_back(n[1]);
    normals.push_back(n[2]);
  } // faces
}

// write density grid
//
// mblocks: max number of blocks in any process
// tblocks: total (global) number of blocks
// outfile: output file name
// project: whether to project to 2D
// glo_num_idx: global number of grid points (i,j,k)
// eps: floating point error tolerance
// data_mins, data_maxs: data global physical extents (x,y,z)
// num_fiven_bounds: number of given extents
// given_mins, given_maxs: given global data extents (x,y,z)
// master: diy master object
// assigner: diy assigner object
void WriteGrid(int mblocks,
               int tblocks,
               char *outfile,
               bool project,
               int *glo_num_idx,
               float eps,
               float *data_mins,
	       float *data_maxs,
               int num_given_bounds,
               float *given_mins,
               float *given_maxs,
               diy::Master& master,
               diy::Assigner& assigner)
{
  MPI_Status status;
  int pts_written;
  MPI_File fd;
  int sizes[3]; // sizes of global array
  int subsizes[3]; // sizes of subarrays
  int starts[3]; // starting offsets of subarrays
  MPI_Datatype dtype; // subarray datatype
  MPI_Comm comm = master.communicator();

  // array of pointers to all my local blocks
  int nblocks = master.size();
  DBlock** dblocks = new DBlock*[nblocks];
  for (int i = 0; i < nblocks; i++)
    dblocks[i] = master.block<DBlock>(i);

  // open
  int retval = MPI_File_open(comm, (char *)outfile,
			     MPI_MODE_WRONLY | MPI_MODE_CREATE,
			     MPI_INFO_NULL, &fd);
  assert(retval == MPI_SUCCESS);
  MPI_File_set_size(fd, 0); // start with an empty file every time

  // global grid parameters
  float grid_phys_mins[3], grid_phys_maxs[3]; // global grid extents
  float grid_step_size[3]; // physical grid space size
  GridStepParams(num_given_bounds, given_mins, given_maxs, data_mins, data_maxs, grid_phys_mins,
                 grid_phys_maxs, grid_step_size, glo_num_idx);

  // project
  if (project)
    ProjectGrid(tblocks, glo_num_idx, eps, data_mins, data_maxs,
                grid_phys_mins, grid_step_size, master, assigner);

  // write
  for (int block = 0; block < mblocks; block++)
  {
    if (block < nblocks) // non-null block
    {
      int num_pts; // total number of points per block

      // local block grid parameters
      int block_min_idx[3]; // global grid index of block minimum grid point
      int block_max_idx[3]; // global grid index of block maximum grid point
      int block_num_idx[3]; // number of grid points in local block
      BlockGridParams(dblocks[block], block_min_idx, block_max_idx, block_num_idx, grid_phys_mins,
                      grid_step_size, eps, data_mins, data_maxs, glo_num_idx);

      if (project)
      {
	// reversed order intentional
	sizes[0] = glo_num_idx[1];
	sizes[1] = glo_num_idx[0];
	starts[0] = block_min_idx[1];
	starts[1] = block_min_idx[0];
	subsizes[0] = block_num_idx[1];
	subsizes[1] = block_num_idx[0];

	MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &dtype);
	MPI_Type_commit(&dtype);
	MPI_File_set_view(fd, 0, MPI_FLOAT, dtype, (char *)"native", MPI_INFO_NULL);

        // blocks not at z0 write 0 points
	num_pts = block_min_idx[2] ? 0 : block_num_idx[0] * block_num_idx[1];
      }
      else
      {
	// reversed order intentional
	sizes[0] = glo_num_idx[2];
	sizes[1] = glo_num_idx[1];
	sizes[2] = glo_num_idx[0];
	starts[0] = block_min_idx[2];
	starts[1] = block_min_idx[1];
	starts[2] = block_min_idx[0];
	subsizes[0] = block_num_idx[2];
	subsizes[1] = block_num_idx[1];
	subsizes[2] = block_num_idx[0];

	MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &dtype);
	MPI_Type_commit(&dtype);
	MPI_File_set_view(fd, 0, MPI_FLOAT, dtype, (char *)"native", MPI_INFO_NULL);

	num_pts = block_num_idx[0] * block_num_idx[1] * block_num_idx[2];
      }

      // write block
      int errcode = MPI_File_write_all(fd, dblocks[block]->density, num_pts, MPI_FLOAT, &status);
      if (errcode != MPI_SUCCESS)
	handle_error(errcode, (char *)"MPI_File_write_all nonempty datatype", comm);
      MPI_Get_count(&status, MPI_FLOAT, &pts_written);
      assert(pts_written == num_pts);

      MPI_Type_free(&dtype);
    }

    else // null block
    {
      float unused;
      MPI_File_set_view(fd, 0, MPI_FLOAT, MPI_FLOAT, (char *)"native", MPI_INFO_NULL);
      MPI_File_write_all(fd, &unused, 0, MPI_FLOAT, &status);
    }
  }

  // cleanup
  MPI_File_close(&fd);
  delete[] dblocks; // pointers to blocks, not the blocks themselves
}

// project density to 2d
//
// gnblocks: total (global) number of blocks
// glo_num_idx: global number of grid points (i,j,k)
// eps: floating point error tolerance
// data_mins, data_maxs: data global physical extents (x,y,z)
// grid_phys_mins, grid_step_size: physical global grid parameters (x, y, z)
// master: diy master object
// assigner: diy assigner object
void ProjectGrid(int gnblocks,
                 int *glo_num_idx,
                 float eps,
                 float *data_mins,
                 float *data_maxs,
                 float *grid_phys_mins,
                 float *grid_step_size,
                 diy::Master& master,
                 diy::Assigner& assigner)
{
  MPI_Comm comm = master.communicator();

  // array of pointers to all my local blocks
  int nblocks = master.size();
  DBlock** dblocks = new DBlock*[nblocks];
  for (int i = 0; i < nblocks; i++)
    dblocks[i] = master.block<DBlock>(i);

  // ------ Blocks info -------
  struct blockgeometry
  {
    int gid;
    int min_idx[3];
  };

  struct blockinfo
  {
    int gid;
    int min_idx[3];
    int max_idx[3];
    int num_idx[3];
    int size;
    int root_gid;
    int zcount;
  };

  struct blockinfo*     block_info      =
    static_cast<struct blockinfo*>(calloc(sizeof(blockinfo), nblocks));
  struct blockgeometry* local_geometry  =
    static_cast<struct blockgeometry*>(calloc(sizeof(blockgeometry), gnblocks));
  struct blockgeometry* global_geometry =
    static_cast<struct blockgeometry*>(calloc(sizeof(blockgeometry), gnblocks));

  for (int block = 0; block < nblocks; ++block)
  {
    // ------------------------------------------------------------------------
    block_info[block].gid = dblocks[block]->gid;
    BlockGridParams(dblocks[block],
                    block_info[block].min_idx,
                    block_info[block].max_idx,
                    block_info[block].num_idx,
                    grid_phys_mins, grid_step_size,
                    eps, data_mins, data_maxs, glo_num_idx);
    block_info[block].size = block_info[block].num_idx[0] * block_info[block].num_idx[1];
    // ------------------------------------------------------------------------
    local_geometry[block_info[block].gid].gid    = block_info[block].gid;
    local_geometry[block_info[block].gid].min_idx[0] = block_info[block].min_idx[0];
    local_geometry[block_info[block].gid].min_idx[1] = block_info[block].min_idx[1];
    local_geometry[block_info[block].gid].min_idx[2] = block_info[block].min_idx[2];
    // ------------------------------------------------------------------------
  }

  MPI_Allreduce((int*)local_geometry, (int*)global_geometry, 4*gnblocks, MPI_INT, MPI_MAX, comm);
  free(local_geometry);

  for (int block = 0; block < nblocks; ++block)
    for (int i = 0; i < gnblocks; ++i)
      if (   block_info[block].min_idx[0] == global_geometry[i].min_idx[0]
             && block_info[block].min_idx[1] == global_geometry[i].min_idx[1] )
      {
        if (!global_geometry[i].min_idx[2])
          block_info[block].root_gid = global_geometry[i].gid;
        ++block_info[block].zcount;
      }
  free(global_geometry);

  vector<MPI_Request> reqs;
  reqs.reserve(nblocks);

  // ------ blocks send -----
  for (int block = 0; block < nblocks; ++block)
  {
    // z=0 blocks are accumulators and dont send any data
    if (!block_info[block].min_idx[2]) continue;

    // rank of block
    int rank;
    MPI_Comm_rank(comm, &rank);

    // rank of root block
    int root_rank;
    root_rank = assigner.rank(block_info[block].root_gid);

    if (rank != root_rank) // if block's rank and projected block's rank differ
    {
      reqs.resize(reqs.size()+1);
      // Send density to projected block's proc
      MPI_Isend(dblocks[block]->density, block_info[block].size, MPI_FLOAT,
                              root_rank, block_info[block].root_gid, comm, &reqs.back());
    }
    else
    {
      int root_block = master.lid(block_info[block].root_gid);
      for (int i = 0; i < block_info[block].size; ++i)
        dblocks[root_block]->density[i] += dblocks[block]->density[i];
      --block_info[root_block].zcount;
    }
  }

  // ------ blocks recv -------
  for (int block = 0; block < nblocks; ++block)
  {
    if (block_info[block].min_idx[2]) continue; // Only z=0 blocks receive data (accumulators)

    // buffer for retreived values
    float* density_buffer = new float[block_info[block].size];

    // For all block on z axis (external block)
    for(int zidx = 1; zidx < block_info[block].zcount; ++zidx)
    {
      // Retrieve density (in buffer) for external block on other proc
      MPI_Recv(density_buffer, block_info[block].size, MPI_FLOAT, MPI_ANY_SOURCE,
                             block_info[block].root_gid, comm, MPI_STATUS_IGNORE);

      // add external block density to local block
      float block_tot_dense = 0.0;  // debug
      float buf_tot_dense = 0.0;    // debug
      for (int i = 0; i < block_info[block].size; ++i)
      {
        dblocks[block]->density[i] += density_buffer[i];
        block_tot_dense += dblocks[block]->density[i];  // debug
        buf_tot_dense += density_buffer[i];             // debug
      }
    }
    delete[] density_buffer;
  }

  // ------ cleanup ------
  if (reqs.size())
    MPI_Waitall(reqs.size(), &reqs[0], MPI_STATUSES_IGNORE);
  free(block_info);
  delete[] dblocks; // pointers to blocks, not the blocks themselves
}

// MPI error handler
// decodes and prints MPI error messages
void handle_error(int errcode,
                  char *str,
                  MPI_Comm comm)
{
  char msg[MPI_MAX_ERROR_STRING];
  int resultlen;
  MPI_Error_string(errcode, msg, &resultlen);
  fprintf(stderr, "%s: %s\n", str, msg);
  MPI_Abort(comm, 1);
}

// compute 1-d index in a block
// points in a block are listed in row major order
//
// block_grid_idx: 3d index in this block (x,y,z)
// block_num_idx: number of pts in each dimension in this block (x,y,z)
// project: whether to project to 2D
// proj_plane: normal of projection plane (x,y,z)
//
// returns: 1-d index
int index(int *block_grid_idx,
          int *block_num_idx,
          bool project,
	  float *proj_plane)
{
  int idx[3]; // index after possible projection
  float proj_length; // projected length of idx onto normal vector
  float shift[3]; // shift in idx due to projection

  // project index into plane
  if (project)
  {
    // todo: does this work for yz and xz planes?

    // projection length is the dot product of idx with proj_plane (normal)
    proj_length =
      block_grid_idx[0] * proj_plane[0] +
      block_grid_idx[1] * proj_plane[1] +
      block_grid_idx[2] * proj_plane[2];

    // compute shift
    shift[0] = proj_length * proj_plane[0];
    shift[1] = proj_length * proj_plane[1];
    shift[2] = proj_length * proj_plane[2];

    idx[0] = block_grid_idx[0] - shift[0];
    idx[1] = block_grid_idx[1] - shift[1];
    idx[2] = block_grid_idx[2] - shift[2];

    // debug, testing x-y plane case
    assert(shift[0] == 0.0 && shift[1] == 0.0 && idx[2] == 0);
  }
  else
  {
    idx[0] = block_grid_idx[0];
    idx[1] = block_grid_idx[1];
    idx[2] = block_grid_idx[2];

  }

  return (idx[2] * block_num_idx[1] * block_num_idx[0] +
	  idx[1] * block_num_idx[0] +
	  idx[0]);
}

// physical position (x,y,z) of a global grid index (i,j,k)
//
// grid_idx: global grid index (x,y,z)
// pos: physical position (x,y,z) (output)
// grid_step_size: physical size of one grid space (x,y,z)
// grid_phys_mins: physical global grid min position (x,y,z)
void idx2phys(int *grid_idx,
              float *pos,
              float *grid_step_size,
	      float *grid_phys_mins)
{
  pos[0] = grid_idx[0] * grid_step_size[0] + grid_phys_mins[0];
  pos[1] = grid_idx[1] * grid_step_size[1] + grid_phys_mins[1];
  pos[2] = grid_idx[2] * grid_step_size[2] + grid_phys_mins[2];
}

// global grid index (i,j,k) of physical position (x,y,z)
// grid index found by integer division (truncating, not rounding)
// so grid index will be at or just before the physical position
//
// pos: physical position (x,y,z)
// grid_idx: global grid index (x,y,z) (output)
// grid_step_size: physical size of one grid space (x,y,z)
// grid_phys_mins: physical global grid min position (x,y,z)
//
void phys2idx(float* pos,
              int *grid_idx,
              float *grid_step_size,
	      float *grid_phys_mins)
{
  grid_idx[0] = (pos[0] - grid_phys_mins[0]) / grid_step_size[0];
  grid_idx[1] = (pos[1] - grid_phys_mins[1]) / grid_step_size[1];
  grid_idx[2] = (pos[2] - grid_phys_mins[2]) / grid_step_size[2];
}

// compute normal of a face using Newell's method
//
// Newell's method is more robust than simply computing the cross product of
//   three points when the points are colinear or slightly nonplanar.
void NewellNormal(float *verts,
                  int num_verts,
                  float *normal)
{
  normal[0] = 0.0;
  normal[1] = 0.0;
  normal[2] = 0.0;

  for (int i = 0; i < num_verts; i++)
  {
    int cur = i;
    int next = (i + 1) % num_verts;
    normal[0] += (verts[3 * cur + 1] - verts[3 * next + 1]) *
      (verts[3 * cur + 2] + verts[3 * next + 2]);
    normal[1] += (verts[3 * cur + 2] - verts[3 * next + 2]) *
      (verts[3 * cur + 0] + verts[3 * next + 0]);
    normal[2] += (verts[3 * cur + 0] - verts[3 * next + 0]) *
      (verts[3 * cur + 1] + verts[3 * next + 1]);
  }

  float mag = sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
		   normal[2] * normal[2]);
  // normalize
  normal[0] /= mag;
  normal[1] /= mag;
  normal[2] /= mag;

  // direction is inward, need to invert
  normal[0] *= -1.0;
  normal[1] *= -1.0;
  normal[2] *= -1.0;
}

// whether a point lies inside a cell
//
// pt: point
// normals: face normals (nx_0,ny_0,nz_0,nx_1,ny_1,nz_1, ...)
// face_verts: vertex positions for each face
// eps: floating point error threshold
//
// returns whether point is in cell (true) or not (false)
bool PtInCell(float *pt,
              vector<float> &normals,
              vector <vector <float> > &face_verts,
              float eps)
{
  int sign; // sign of distance (1 or -1)
  int old_sign = 0; // previous sign, 0 = uninitialized
  float dist = 0.0; // signed distance from point to plane

  for (int k = 0; k < (int)face_verts.size(); k++) // faces
  {
    // compute distance from point to face
    dist = 0.0;

    float *n = &(normals[3 * k]); // current normal
    dist = n[0] * (pt[0] - face_verts[k][0]) +
      n[1] * (pt[1] - face_verts[k][1]) +
      n[2] * (pt[2] - face_verts[k][2]);

    // check sign of distance only if non-zero
    if (fabs(dist) > eps)
    {
      sign = (dist >= 0.0 ? 1 : -1);
      if (old_sign == 0)
        old_sign = sign;
      if (old_sign != sign)
        return false;
    }
  } // faces

  return true;
}

// convert global grid index to local block grid index
//
// global_idx: global grid index (i,j,k)
// local_idx: (output) local grid index in block (i,j,k)
// block_min_dx: minimum index (i,j,k) in local block
void Global2LocalIdx(int *global_idx,
                     int *local_idx,
                     int *block_min_idx)
{
  // compute local index of current grid point in this block
  local_idx[0] = global_idx[0] - block_min_idx[0];
  local_idx[1] = global_idx[1] - block_min_idx[1];
  local_idx[2] = global_idx[2] - block_min_idx[2];
}

// finds global data bounds
void DataBounds(float *data_mins,
                float *data_maxs,
                diy::Master& master)
{
  float block_mins[3]; // mins of all local blocks
  float block_maxs[3]; // maxs of all local blocks
  int rank;

  MPI_Comm comm = master.communicator();
  MPI_Comm_rank(comm, &rank);

  // array of pointers to all my local blocks
  int nblocks = master.size();
  DBlock** dblocks = new DBlock*[nblocks];
  for (int i = 0; i < nblocks; i++)
    dblocks[i] = master.block<DBlock>(i);

  for (int i = 0; i < nblocks; i++)
  {
    if (i == 0)
    {
      block_mins[0] = dblocks[i]->bounds.min[0];
      block_mins[1] = dblocks[i]->bounds.min[1];
      block_mins[2] = dblocks[i]->bounds.min[2];
      block_maxs[0] = dblocks[i]->bounds.max[0];
      block_maxs[1] = dblocks[i]->bounds.max[1];
      block_maxs[2] = dblocks[i]->bounds.max[2];
    }
    else
    {
      if (dblocks[i]->bounds.min[0] < block_mins[0])
	block_mins[0] = dblocks[i]->bounds.min[0];
      if (dblocks[i]->bounds.min[1] < block_mins[1])
	block_mins[1] = dblocks[i]->bounds.min[1];
      if (dblocks[i]->bounds.min[2] < block_mins[2])
	block_mins[2] = dblocks[i]->bounds.min[2];
      if (dblocks[i]->bounds.max[0] > block_maxs[0])
	block_maxs[0] = dblocks[i]->bounds.max[0];
      if (dblocks[i]->bounds.max[1] > block_maxs[1])
	block_maxs[1] = dblocks[i]->bounds.max[1];
      if (dblocks[i]->bounds.max[2] > block_maxs[2])
	block_maxs[2] = dblocks[i]->bounds.max[2];
    }
  }

  MPI_Allreduce(block_mins, data_mins, 3, MPI_FLOAT, MPI_MIN, comm);
  MPI_Allreduce(block_maxs, data_maxs, 3, MPI_FLOAT, MPI_MAX, comm);

  if (rank == 0)
    fprintf(stderr, "data bounds: min = [%.3f %.3f %.3f] max = [%.3f %.3f %.3f]\n",
	    data_mins[0], data_mins[1], data_mins[2],
	    data_maxs[0], data_maxs[1], data_maxs[2]);

  delete[] dblocks; // only pointers to blocks, not the actual blocks
}

// print summary stats
//
// times: timing info
// master: diy master
// grid_step_size: physical size of one grid space (x,y,z)
// grid_phys_mins: physical min corner of global grid (x,y,z)
// glo_num_idx: global size of the grid (i,j,k)
void dense_stats(double *times,
                 diy::Master& master,
                 float *grid_step_size,
                 float *grid_phys_mins,
                 int *glo_num_idx)
{
  MPI_Comm comm = master.communicator();
  int rank;
  MPI_Comm_rank(comm, &rank);

  float glo_max_dense = 0; // global max density
  double glo_tot_mass = 0; // global total mass
  float glo_check_mass = 0; // global reference total mass

  MPI_Reduce(&max_dense, &glo_max_dense, 1, MPI_FLOAT, MPI_MAX, 0, comm);
  MPI_Reduce(&tot_mass, &glo_tot_mass, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(&check_mass, &glo_check_mass, 1, MPI_FLOAT, MPI_SUM, 0, comm);

  // physical positions of global grid extents
  float grid_min_pos[3], grid_max_pos[3];
  int idx[3];  // grid index
  idx[0] = 0;
  idx[1] = 0;
  idx[2] = 0;
  idx2phys(idx, grid_min_pos, grid_step_size, grid_phys_mins);
  idx[0] = glo_num_idx[0] - 1;
  idx[1] = glo_num_idx[1] - 1;
  idx[2] = glo_num_idx[2] - 1;
  idx2phys(idx, grid_max_pos, grid_step_size, grid_phys_mins);

  if (rank == 0)
  {
    fprintf(stderr, "--------------Summary--------------\n");
    fprintf(stderr, "Global 3D grid size = %d x %d x %d\n",
	    glo_num_idx[0], glo_num_idx[1], glo_num_idx[2]);
    fprintf(stderr, "Physical grid extends from min [%.4e %.4e %.4e]\n"
	    "to max [%.4e %.4e %.4e]\n"
	    "by steps of [%.4e %.4e %.4e]\n",
	    grid_min_pos[0], grid_min_pos[1], grid_min_pos[2],
	    grid_max_pos[0], grid_max_pos[1], grid_max_pos[2],
	    grid_step_size[0], grid_step_size[1], grid_step_size[2]);
    fprintf(stderr, "max_dense = %.3e tot_mass = %.3e (should be %.3e)\n",
	    glo_max_dense, glo_tot_mass, glo_check_mass);
    fprintf(stderr, "Total time = %.3lf s = \n", times[TOTAL_TIME]);
    fprintf(stderr, "%.3lf s input + %.3lf s density computation + "
	    "%.3lf s output\n",
	    times[INPUT_TIME], times[COMP_TIME], times[OUTPUT_TIME]);
    fprintf(stderr, "-----------------------------------\n");
  }
}

// grid points covered by one cell
//
//  if the cell covers at least one grid point, then actual number of grid
//    points will be returned and the cell mass will be distributed evenly
//    over that nubmer of points
//  if the cell does not cover any grid points, then the nearest grid point
//     will be returned and the mass of the cell will be deposited there
//
// todo: see if it is more accurate to deposit over a minimum of 8 grid points
//
// cell_mins: minimum cell vertex (x,y,z)
// cell_maxs: maximum cell vertex (x,y,z)
// grid_pts: (output) grid points covered by this cell, allocated by this
//   function, caller's responsibility to free
// border: cell border, min and max x index for each y and z index
// alloc_grid_pts: number of grid points currently allocated, this function
//   will realloc to the new size if needed, otherwise will leave old size
// normals: face normals (nx_0,ny_0,nz_0,nx_1,ny_1,nz_1, ...)
// face_verts: vertex positions for each face
// data_mins, data_maxs; global data physical extents (x,y,z)
// grid_phys_mins: global physical min grid point position (x,y,z)
// grid_step_size: physical size of grid space (x,y,z)
// mass: mass of one particle
// eps: floating point error tolerance
// site: site of the cell (x,y,z)
//
// returns: number of grid points covered by this cell
// 0 indicates cell is outside of global data bounds (skip it)
int CellGridPts(float *cell_mins,
		float *cell_maxs,
                grid_pt_t* &grid_pts,
                int* &border,
		int &alloc_grid_pts,
                vector<float> &normals,
		vector <vector <float> > &face_verts,
                float *data_mins,
		float *data_maxs,
                float *grid_phys_mins,
		float *grid_step_size,
                float mass,
                float eps,
                float *site)
{
  float center[3]; // cell center
  int num_grid_pts; // number of grid points covered by this cell

  float epsilon[3]; // extend domain by epsilon to include wall generated cells
  for (size_t i=0; i<3; ++i) epsilon[i] = (data_maxs[i] - data_mins[i]) *
                               2.0f * std::numeric_limits<float>::epsilon();

  // filter out cells that are outside of global data bounds
  if (   cell_mins[0] < data_mins[0] - epsilon[0]
      || cell_mins[1] < data_mins[1] - epsilon[1]
      || cell_mins[2] < data_mins[2] - epsilon[2]
      || cell_maxs[0] > data_maxs[0] + epsilon[0]
      || cell_maxs[1] > data_maxs[1] + epsilon[1]
      || cell_maxs[2] > data_maxs[2] + epsilon[2])
    return 0;

  // global grid index of cell minimum grid point
  int cell_min_grid_idx[3];
  phys2idx(cell_mins, cell_min_grid_idx, grid_step_size, grid_phys_mins);

  // global grid index of cell maximum grid point
  int cell_max_grid_idx[3];
  phys2idx(cell_maxs, cell_max_grid_idx, grid_step_size, grid_phys_mins);

  // cell minimum grid point physical position
  float cell_min_grid_pos[3];
  idx2phys(cell_min_grid_idx, cell_min_grid_pos, grid_step_size, grid_phys_mins);

  // number of grid points covered by cell bounding box
  int cell_grid_pts[3];
  cell_grid_pts[0] = cell_max_grid_idx[0] - cell_min_grid_idx[0] + 1;
  cell_grid_pts[1] = cell_max_grid_idx[1] - cell_min_grid_idx[1] + 1;
  cell_grid_pts[2] = cell_max_grid_idx[2] - cell_min_grid_idx[2] + 1;

  // grid_pts and border memory allocation
  int npts = cell_grid_pts[0] * cell_grid_pts[1] * cell_grid_pts[2];
  if (npts < 8)
    npts = 8; // allocate at least 8 points in case using CIC for small cell case

  if (!alloc_grid_pts)
  {
    grid_pts = (grid_pt_t *)malloc(npts * sizeof(grid_pt_t));
    border = (int *)malloc(npts * 2 * sizeof(int)); // more than large enough
    alloc_grid_pts = npts;
  }
  else if (npts > alloc_grid_pts)
  {
    grid_pts = (grid_pt_t *)realloc(grid_pts, npts * sizeof(grid_pt_t));
    border = (int *)realloc(border, npts * sizeof(grid_pt_t));
    alloc_grid_pts = npts;
  }
  memset(grid_pts, 0 , npts * sizeof(grid_pt_t));

  num_grid_pts = CellInteriorGridPts(cell_grid_pts, cell_min_grid_idx,
				     cell_min_grid_pos, grid_pts, border,
				     normals, face_verts, grid_step_size,
				     eps, mass);

  // if no grid points covered by cell, distribute mass to 8 neighboring grid points using CIC
  if (!num_grid_pts)
  {
    vector<int> grid_idxs; // grid idxs that get a fraction of the mass
    vector<float> grid_masses; // mass given to each grid_idx
    DistributeScalarCIC(site, mass, grid_idxs, grid_masses, grid_step_size,
			grid_phys_mins, eps);
    assert((int)grid_idxs.size() / 3 == 8); // sanity

    // store idx and mass at each of the 8 grid points
    for (int i = 0; i < (int)grid_idxs.size() / 3; i++)
    {
      grid_pts[i].idx[0] = grid_idxs[3 * i    ];
      grid_pts[i].idx[1] = grid_idxs[3 * i + 1];
      grid_pts[i].idx[2] = grid_idxs[3 * i + 2];
      grid_pts[i].mass = grid_masses[i];
    }

    num_grid_pts = 8;
  }

  return num_grid_pts;
}

// finds interior grid points in cell and sets density at them
// current version further improved by limiting y scan
//
// cell_grid_pts: number of grid points covered by cell bounding box
// cell_min_grid_idx: cell minimum grid point global index
// cell_min_grid_pos: cell minimum grid point physical position
// grid_pts: (output) grid points covered by this cell, allocated by caller
// border: cell border, min and max x index for each y and z index
// normals: face normals (nx_0,ny_0,nz_0,nx_1,ny_1,nz_1, ...)
// face_verts: vertex positions for each face
// grid_step_size: physical size of grid space (x,y,z)
// eps: floating point error tolerance
// mass: mass of one particle
//
// returns: number of interior grid points
int CellInteriorGridPts(int *cell_grid_pts,
			int *cell_min_grid_idx,
                        float *cell_min_grid_pos,
			grid_pt_t *grid_pts,
                        int *border,
			vector<float> &normals,
			vector <vector <float> > &face_verts,
			float *grid_step_size,
                        float eps,
                        float mass)
{
  int num_grid_pts = 0; // current number of grid points interior to cell
  int tot_num_grid_pts = 0; // total number of grid points interior to cell
  float grid_pos[3]; // physical position of current grid point
  int x_left = cell_grid_pts[0] / 2; // x index stepping left
  int x_right = cell_grid_pts[0] / 2; // x index stepping right
  int y_start = 0; // y index start
  bool x_in_left, x_in_right; // pt inside cell while stepping left, right
  int min_xi, max_xi; // min, max x index of border crossing
  bool border_found = false; // found border intersection at current z
  bool z_step_done = false; // this z step is done
  int xi, yi, zi; // indices for x, y, z
  int yj; // second, temporary index in y
  int first_x = 0; // x index of border crossing at first y line in each z

  int y_steps = 0; // see how many y_steps we ended up making

  // find the border points of the cell

  // z step
  for (zi = 0; zi < cell_grid_pts[2]; zi++)
  {
    grid_pos[2] = cell_min_grid_pos[2] + zi * grid_step_size[2];
    border_found = false; // init
    z_step_done = false;

    // initialize (with min > max) unused y-scan lines
    for (yj = 0; yj < y_start; yj++) // prior to start
    {
      border[2 * (zi * cell_grid_pts[1] + yj)]     = 1; // min
      border[2 * (zi * cell_grid_pts[1] + yj) + 1] = 0; // max
    }

    // y step
    // even though the upper loop bound is the full number of grid points,
    // early termination will occur deep in the body of the loop when the
    // number of x-intersections becomes 0
    for (yi = y_start; yi < cell_grid_pts[1]; yi++)
    {
      // debug
      y_steps++;

      grid_pos[1] = cell_min_grid_pos[1] + yi * grid_step_size[1];

      // init the x stepping
      grid_pos[0] = cell_min_grid_pos[0] + x_left * grid_step_size[0];
      grid_pos[1] = cell_min_grid_pos[1] + yi * grid_step_size[1];
      grid_pos[2] = cell_min_grid_pos[2] + zi * grid_step_size[2];
      if (PtInCell(grid_pos, normals, face_verts, eps))
      {
	x_in_left = true;
	x_in_right = true;
      }
      else
      {
	x_in_left = false;
	x_in_right = false;
      }
      min_xi = cell_grid_pts[0] - 1;
      max_xi = 0;

      // x step left
      for (xi = x_left; xi >= 0 && xi < cell_grid_pts[0];)
      {
	grid_pos[0] = cell_min_grid_pos[0] + xi * grid_step_size[0];
	if (PtInCell(grid_pos, normals, face_verts, eps))
        {
	  if (x_in_left) // remains interior, keep stepping
          {
	    if (xi < min_xi)
	      min_xi = xi;
	    if (xi > max_xi)
	      max_xi = xi;
	    xi--;
	  }
	  else // edge crossing from interior to exterior
          {
	    if (xi < min_xi)
	      min_xi = xi;
	    if (xi > max_xi)
	      max_xi = xi;
	    x_left = xi;
	    break;
	  }
	} // pt is inside the cell

	// pt is outside the cell
	else
        {
	  if (!x_in_left) // remains exterior, keep stepping
	    xi++;
	  else // edge crossing from exterior to interior
          {
	    x_left = xi;
	    break;
	  }
	} // pt is outside the cell
      } // x step left

      // x step right
      for (xi = x_right; xi >= 0 && xi < cell_grid_pts[0];)
      {
	grid_pos[0] = cell_min_grid_pos[0] + xi * grid_step_size[0];
	if (PtInCell(grid_pos, normals, face_verts, eps))
        {
	  if (x_in_right) // remains interior, keep stepping
          {
	    if (xi < min_xi)
	      min_xi = xi;
	    if (xi > max_xi)
	      max_xi = xi;
	    xi++;
	  }
	  else // edge crossing from interior to exterior
          {
	    if (xi < min_xi)
	      min_xi = xi;
	    if (xi > max_xi)
	      max_xi = xi;
	    x_right = xi;
	    break;
	  }
	} // pt is inside the cell

	// pt is outside the cell
	else
        {
	  if (!x_in_right) // remains exterior, keep stepping
	    xi--;
	  else // edge crossing from exterior to interior
          {
	    x_right = xi;
	    break;
	  }
	} // pt is outside the cell
      } // x step right

      border[2 * (zi * cell_grid_pts[1] + yi)]     = min_xi;
      border[2 * (zi * cell_grid_pts[1] + yi) + 1] = max_xi;

      // min_xi > max_xi is the signal that no points were found

      // intersection found in this y-scan line
      if (min_xi <= max_xi)
      {
	tot_num_grid_pts += (max_xi - min_xi + 1);
	if (yi == y_start)
	  first_x = (min_xi + max_xi) / 2;
      }

      // note the first y line with an intersection in this z step
      int first_y; // y line with first border points
      first_y = y_start; // initial values change nothing unless reset below
      if (min_xi <= max_xi && !border_found)
      {
	first_y = yi;
	border_found = true;
      }

      if (min_xi > max_xi && border_found)
	z_step_done = true;

      // when this z step is done and if there are more z steps to do,
      // check if y_start needs to be loosened for next z step
      if ((yi == cell_grid_pts[1] - 1 || z_step_done)
	  && zi + 1 < cell_grid_pts[2])
      {
	grid_pos[2] = cell_min_grid_pos[2] + (zi + 1) * grid_step_size[2];
	grid_pos[0] = cell_min_grid_pos[0] + first_x * grid_step_size[0];
	for (yj = first_y; yj > 0; yj--)
        {
	  grid_pos[1] = cell_min_grid_pos[1] + yj * grid_step_size[1];
	  if (!PtInCell(grid_pos, normals, face_verts, eps))
	    break;
	}
	y_start = yj;
      } // loosen y start

      // terminate the yi loop (this z step) early if it is done
      // also initialize the borders of unused y-scan lines skipped at the end
      if (z_step_done)
      {
	for (yj = yi + 1; yj < cell_grid_pts[1]; yj++) // after end
        {
	  border[2 * (zi * cell_grid_pts[1] + yj)]     = 1; // min
	  border[2 * (zi * cell_grid_pts[1] + yj) + 1] = 0; // max
	}

	break; // y step
      }
    } // y step
  } // z step

  // deposit the mass in the interior of the cell
  num_grid_pts = 0;
  for (zi = 0; zi < cell_grid_pts[2]; zi++) // z
  {
    for (yi = 0; yi < cell_grid_pts[1]; yi++) // y
    {
      min_xi = border[2 * (zi * cell_grid_pts[1] + yi)];
      max_xi = border[2 * (zi * cell_grid_pts[1] + yi) + 1];
      for (xi = min_xi; xi <= max_xi; xi++) // x
      {
	grid_pts[num_grid_pts].idx[0] = cell_min_grid_idx[0] + xi;
	grid_pts[num_grid_pts].idx[1] = cell_min_grid_idx[1] + yi;
	grid_pts[num_grid_pts].idx[2] = cell_min_grid_idx[2] + zi;
	// deposit mass onto grid points, density to be computed later
	grid_pts[num_grid_pts].mass = mass / (float)tot_num_grid_pts;
	num_grid_pts++;
      }
    }
  }

  // cleanup
  assert(tot_num_grid_pts == num_grid_pts); // sanity
  return num_grid_pts;
}

// computes grid step parameters
//
// int num_given_bounds:
//  0 = none, 1 = x bounds given, 2 = x,y bounds given, 3 = x,y,z bounds given
// given_mins, given_maxs: given bounds (x,y,z)
// data_mins, data_maxs: data global physical extents (x,y,z)
// grid_phys_mins, grid_phys_maxs: grid global physical extents (x,y,z) (output)
// grid_step_size: physical size of one grid space (x,y,z) (output)
// glo_num_idx: global number of grid points (i,j,k)
void GridStepParams(int num_given_bounds,
                    float *given_mins,
		    float *given_maxs,
                    float *data_mins, float *data_maxs,
		    float *grid_phys_mins,
                    float *grid_phys_maxs,
		    float *grid_step_size,
                    int *glo_num_idx)
{
  // max data size
  float max_data_size; // max data extent in x, y, or z
  max_data_size = data_maxs[0] - data_mins[0];
  if (data_maxs[1] - data_mins[1] > max_data_size)
    max_data_size = data_maxs[1] - data_mins[1];
  if (data_maxs[2] - data_mins[2] > max_data_size)
    max_data_size = data_maxs[2] - data_mins[2];

  // grid physical bounds
  grid_phys_mins[0] = data_mins[0] -
    (max_data_size - (data_maxs[0] - data_mins[0])) / 2.0;
  grid_phys_mins[1] = data_mins[1] -
    (max_data_size - (data_maxs[1] - data_mins[1])) / 2.0;
  grid_phys_mins[2] = data_mins[2] -
    (max_data_size - (data_maxs[2] - data_mins[2])) / 2.0;
  grid_phys_maxs[0] = data_maxs[0] +
    (max_data_size - (data_maxs[0] - data_mins[0])) / 2.0;
  grid_phys_maxs[1] = data_maxs[1] +
    (max_data_size - (data_maxs[1] - data_mins[1])) / 2.0;
  grid_phys_maxs[2] = data_maxs[2] +
    (max_data_size - (data_maxs[2] - data_mins[2])) / 2.0;

  // override grid bounds
  if (num_given_bounds >= 1)
  {
    grid_phys_mins[0] = given_mins[0];
    grid_phys_maxs[0] = given_maxs[0];
  }
  if (num_given_bounds >= 2)
  {
    grid_phys_mins[1] = given_mins[1];
    grid_phys_maxs[1] = given_maxs[1];
  }
  if (num_given_bounds >= 3)
  {
    grid_phys_mins[2] = given_mins[2];
    grid_phys_maxs[2] = given_maxs[2];
  }

  // grid step size
  grid_step_size[0] = (grid_phys_maxs[0] - grid_phys_mins[0]) /
    (glo_num_idx[0] - 1);
  grid_step_size[1] = (grid_phys_maxs[1] - grid_phys_mins[1]) /
    (glo_num_idx[1] - 1);
  grid_step_size[2] = (grid_phys_maxs[2] - grid_phys_mins[2]) /
    (glo_num_idx[2] - 1);
}

// distributes scalar value to grid points within a window size of one
//  grid space, ie, CIC for the 8 grid points of a cell (vertex centered),
//  equivalent to CIC for 8 neighboring cells (cell-centered)
//
// pt: input point
// scalar: scalar value at input point
// grid_idxs: global grid indices of grid points within window size of
//  input point (i,j,k,i,j,k,...) (output)
// grid_scalars: distributed scalars at each grid_idx (output)
// grid_step_size: physical size of one grid space (x,y,z)
// grid_phys_mins: physical global grid minimum (x,y,z)
// eps: floating point tolerance
//
// The implementation below is more complicated than plain CIC because
// it can handle larger window sizes one grid space,
// hence the computation of vol / v0 and eventually normalizing by
// tot_weight. This would be uneccessary in ordinary CIC, but the
// result is the same as CIC when the window is one grid space.
void DistributeScalarCIC(float *pt,
                         float scalar,
			 vector <int> &grid_idxs,
                         vector <float> &grid_scalars,
			 float *grid_step_size,
                         float *grid_phys_mins,
			 float eps)
{
  // global grid indices of window min and max grid points
  //
  // This is the only part simplified for CIC, min and max window points
  // are just one step apart. After this, the rest of the computation is
  // completely general for any window size.
  int min_win_idx[3];
  int max_win_idx[3];
  phys2idx(pt, min_win_idx, grid_step_size, grid_phys_mins);
  max_win_idx[0] = min_win_idx[0] + 1;
  max_win_idx[1] = min_win_idx[1] + 1;
  max_win_idx[2] = min_win_idx[2] + 1;

  float tot_weight = 0.0f; // total of weights in the window, should be 1.0

  // distribute fractional densities onto grid points in the window
  float v0 = 0.0; // volume of first box computed
  vector <float> weights; // weights accociated with grid points in the windonw
  int ijk[3]; // grid index
  for (ijk[2] = min_win_idx[2]; ijk[2] <= max_win_idx[2]; ijk[2]++)
  {
    for (ijk[1] = min_win_idx[1]; ijk[1] <= max_win_idx[1]; ijk[1]++)
    {
      for (ijk[0] = min_win_idx[0]; ijk[0] <= max_win_idx[0]; ijk[0]++)
      {
	grid_idxs.push_back(ijk[0]);
	grid_idxs.push_back(ijk[1]);
	grid_idxs.push_back(ijk[2]);

	// move point a little if it lies on a grid line
	float grid_pos[3]; // physical position of grid point
	float p[3] = { pt[0], pt[1], pt[2] }; // temp copy of pt
	idx2phys(ijk, grid_pos, grid_step_size, grid_phys_mins);
	if (fabs(p[0] - grid_pos[0]) < eps)
	  p[0] += 2 * eps;
	if (fabs(p[1] - grid_pos[1]) < eps)
	  p[1] += 2 * eps;
	if (fabs(p[2] - grid_pos[2]) < eps)
	  p[2] += 2 * eps;

	// volume of box formed by input point and grid point
	float vol = fabs((grid_pos[0] - p[0]) * (grid_pos[1] - p[1]) *
			 (grid_pos[2] - p[2]));
	assert(vol > 0.0f); // sanity
	if (v0 == 0.0) // set v0 to first volume computed
	  v0 = vol;

	float v = v0 / vol; // volume as a factor of v0
	weights.push_back(v);
	tot_weight += v;

      }
    }
  }

  // debug
  float tot_norm_weight = 0.0f; // total normalized weight

  // normalize weights and deposit densities
  for (int i = 0; i < (int)weights.size(); i++)
  {
    weights[i] /= tot_weight; // normalized weight
    grid_scalars.push_back(weights[i] * scalar); // scalar on the grid point
    // debug
    tot_norm_weight += weights[i]; // for sanity check later, add to 1.0
  }

  // debug
//   fprintf(stderr, "pt [%.3f %.3f %.3f] min_win_idx [%d %d %d]\n",
// 	  pt[0], pt[1], pt[2], min_win_idx[0], min_win_idx[1], min_win_idx[2]);
//   for (int i = 0; i < (int)grid_scalars.size(); i++)
//     fprintf(stderr, "mass[%d %d %d] = %.3f ",
// 	    grid_idxs[3 * i], grid_idxs[3 * i + 1], grid_idxs[3 * i + 2],
// 	    grid_scalars[i]);
//   fprintf(stderr, "\n");

  // debug
  assert(fabs(tot_norm_weight - 1.0f) < eps); // sanity
}
