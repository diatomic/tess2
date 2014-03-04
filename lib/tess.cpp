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

// using new tet data model (eventually default)
#define TET

#include "mpi.h"
#include "diy.h"
#include "tess.h"
#include "io.h"

#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <sys/resource.h>

#include <vector>
#include <set>
#include <algorithm>
#include "tet.h"
#include "tet-neighbors.h"

using namespace std;

static int dim = 3; // everything 3D 
static float data_mins[3], data_maxs[3]; // extents of overall domain 
MPI_Comm comm; // MPI communicator 
static float min_vol, max_vol; // cell volume range 
static int nblocks; // number of blocks per process 
static double *times; // timing info 
static int wrap_neighbors; // whether wraparound neighbors are used 
// CLP - if wrap_neighbors is 0 then check this condition. 
static int walls_on;

// c source files such as tess-qhull.c do not want to see
// C++ arguments, so I hid these function prototypes here-TP
void create_dblocks(int num_blocks, struct dblock_t* &dblocks, int** &hdrs,
		    float **particles, int *num_particles);
void reset_dblocks(int num_blocks, struct dblock_t* &dblocks);
void fill_vert_to_tet(dblock_t* dblock);
void incomplete_dcells_initial(struct dblock_t *tblock, int lid,
			       vector <sent_t> &sent_particles,
			       vector <int> &convex_hull_particles);
void incomplete_dcells_final(struct dblock_t *dblock,
			    int lid,
			    vector <sent_t> &sent_particles,
			    vector <int> &convex_hull_particles);
void neighbor_d_is_complete(int nblocks, struct dblock_t *dblocks,
			    struct remote_ic_t **rics,
			    vector <struct sent_t> *sent_particles);

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
  for (i = 0; i < MAX_TIMES; i++)
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
  for (i = 0; i < MAX_TIMES; i++)
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

#ifdef TET

  delaunay(nblocks, particles, num_particles, times, out_file);

#else

  voronoi_delaunay(nblocks, particles, num_particles, times, out_file);

#endif

}
// ------------------------------------------------------------------------
//
//   test of parallel tesselation
//
//   tot_blocks: total number of blocks in the domain
//   data_size: domain grid size (x, y, z)
//   jitter: maximum amount to randomly move each particle
//   minvol, maxvol: filter range for which cells to keep
//   pass -1.0 to skip either or both bounds
//   wrap: whether wraparound neighbors are used
//   twalls_on: whether walls boundaries are used
//   times: times for particle exchange, voronoi cells, convex hulls, and output
//   outfile: output file name
//
void tess_test(int tot_blocks, int *data_size, float jitter, 
	       float minvol, float maxvol, int wrap, int twalls_on, 
	       double *times, char *outfile) {

  float **particles; // particles[block_num][particle] 
		     // where each particle is 3 values, px, py, pz 
  int *num_particles; // number of particles in each block 
  int dim = 3; // 3D 
  int given[3] = {0, 0, 0}; // no constraints on decomposition in {x, y, z} 
  int ghost[6] = {0, 0, 0, 0, 0, 0}; // ghost in {-x, +x, -y, +y, -z, +z} 
  int nblocks; // my local number of blocks 
  int i;

  comm = MPI_COMM_WORLD;
  min_vol = minvol;
  max_vol = maxvol;
  wrap_neighbors = wrap;
  walls_on = twalls_on;
  
  // debug 
  // fprintf(stderr, "Wrap %d, Walls %d\n", wrap_neighbors,walls_on); 

  // data extents 
  for(i = 0; i < 3; i++) {
    data_mins[i] = 0.0;
    data_maxs[i] = data_size[i] - 1.0;
  }

  // have DIY do the decomposition 
  DIY_Init(dim, 1, comm);
  DIY_Decompose(ROUND_ROBIN_ORDER, data_size, tot_blocks, &nblocks, 1, 
		ghost, given, wrap);

  // generate test points in each block 
  particles = (float **)malloc(nblocks * sizeof(float *));
  num_particles = (int *)malloc(nblocks * sizeof(int));
  for (i = 0; i < nblocks; i++)
    num_particles[i] = gen_particles(i, &particles[i], jitter);

  // save particles in a separate file for future reference 
  // write_particles(nblocks, particles, num_particles, "pts.out"); 

#ifdef TET

  // compute tessellations 
  delaunay(nblocks, particles, num_particles, times, outfile);

#else

  // compute tessellations 
  voronoi_delaunay(nblocks, particles, num_particles, times, outfile);

#endif

  // cleanup 
  for (i = 0; i < nblocks; i++)
    free(particles[i]);
  free(particles);
  free(num_particles);

  DIY_Finalize();

}
// --------------------------------------------------------------------------
//
//   parallel voronoi and delaunay tesselation
//
//   nblocks: local number of blocks
//   particles: particles[block_num][particle] 
//   where each particle is 3 values, px, py, pz
//   num_particles; number of particles in each block
//   times: times for particle exchange, voronoi cells, convex hulls, and output
//   out_file: output file name
// 
void voronoi_delaunay(int nblocks, float **particles, int *num_particles, 
		      double *times, char *out_file) {

  int *num_orig_particles; // number of original particles, before any
			   // neighbor exchange 
  int dim = 3; // 3D 
  int rank; // MPI rank 
  int i;
  void* ds; // persistent delaunay data structures

  MPI_Comm_rank(comm, &rank);

  // init timing 
  for (i = 0; i < MAX_TIMES; i++)
    times[i] = 0.0;

  int **hdrs; // headers 
  struct vblock_t *vblocks; // voronoi blocks 
  struct vblock_t *tblocks; // temporary voronoi blocks 

  num_orig_particles = (int *)malloc(nblocks * sizeof(int));
  for (i = 0; i < nblocks; i++)
    num_orig_particles[i] = num_particles[i];

  ds = init_delaunay_data_structures(nblocks);

  // allocate and initialize blocks 
  create_blocks(nblocks, &vblocks, &hdrs); // final 
  create_blocks(nblocks, &tblocks, NULL); // temporary 
  
  // CLP - if !wrap_neighbors && walls, then initialize wall structure 
  //   using data_mins and data_maxs
  //   Currently assuimg walls on all sides, but format can easily be 
  //   modified to be ANY set of walls 
  struct wall_t *walls;
  int num_walls = 0;
  if (!wrap_neighbors && walls_on)
    create_walls(&num_walls,&walls);

#ifdef MEMORY
  int dwell = 0;
  get_mem(1, dwell);
#endif

#ifdef TIMING
  MPI_Barrier(comm);
  times[LOCAL_TIME] = MPI_Wtime();
#endif

  // create local voronoi cells
  std::vector<tet_t*>	tets(nblocks);
  std::vector<int>	ntets(nblocks);
  local_cells(nblocks, tblocks, dim, num_particles, particles, ds, &tets[0], &ntets[0]);
  
  #ifdef TIMING
  MPI_Barrier(comm);
  times[LOCAL_TIME] = MPI_Wtime() - times[LOCAL_TIME];
  if (rank == 0)
    fprintf(stderr, "-----------------------------------\n");
  times[EXCH_TIME] = MPI_Wtime();
#endif

#ifdef MEMORY
  get_mem(2, dwell);
#endif

  // keep track of which particles lie on the convex hull of the local points 
  int** convex_hull_particles	  = (int**) malloc(nblocks * sizeof(int*));
  int*  num_convex_hull_particles = (int*)  malloc(nblocks * sizeof(int));
  for (i = 0; i < nblocks; i++)
    convex_hull_particles[i] = NULL;

  // determine which cells are incomplete or too close to neighbor 
  for (i = 0; i < nblocks; i++)
    incomplete_cells_initial(&tblocks[i], &vblocks[i], i,
			     &convex_hull_particles[i],
			     &num_convex_hull_particles[i]);

#ifdef MEMORY
  get_mem(3, dwell);
#endif

  // cleanup local temporary blocks 
  destroy_blocks(nblocks, tblocks, NULL);

  // exchange particles with neighbors 
  int **gids; // owner global block ids of received particles 
  int **nids; // owner native particle ids of received particles 
  unsigned char **dirs; // wrapping directions of received articles 
  gids = (int **)malloc(nblocks * sizeof(int *));
  nids = (int **)malloc(nblocks * sizeof(int *));
  dirs = (unsigned char **)malloc(nblocks * sizeof(unsigned char *));
  // intialize to NULL to make realloc inside neighbor_particles just work 
  for (i = 0; i < nblocks; ++i) {
    gids[i] = NULL;
    nids[i] = NULL;
    dirs[i] = NULL;
  }

#ifdef MEMORY
  get_mem(4, dwell);
#endif

  neighbor_particles(nblocks, particles, num_particles, num_orig_particles,
		     gids, nids, dirs);

#ifdef MEMORY
  get_mem(5, dwell);
#endif

  // Second, decisive phase 

  // reset real blocks and allocate and initialize temporary blocks again 
  reset_blocks(nblocks, vblocks);
  create_blocks(nblocks, &tblocks, NULL); // temporary 

  // Clean-up tets; local_cells() will refill them
  for(int i = 0; i < nblocks; ++i)
    free(tets[i]);

  // Recompute local cells
  local_cells(nblocks, tblocks, dim, num_particles, particles, ds, &tets[0], &ntets[0]);

#ifdef MEMORY
  get_mem(6, dwell);
#endif

  // CLP - Create  pointers to wall-mirror particles for each block 
  float** mirror_particles;
  int *num_mirror_particles; // number of received particles for each block 
  mirror_particles = (float **)malloc(nblocks * sizeof(float *));
  for (i = 0; i < nblocks; i++)
    mirror_particles[i] = NULL;
  num_mirror_particles = (int *)malloc(nblocks * sizeof(int));
  
  // CLP - give walls and pointer for creating wall-mirror particles 
  //    to function call 
  for (i = 0; i < nblocks; i++)
    incomplete_cells_final(&tblocks[i], &vblocks[i], i, 
			   convex_hull_particles[i], 
			   num_convex_hull_particles[i], walls, num_walls, 
			   &mirror_particles[i], &num_mirror_particles[i]);

#ifdef MEMORY
  get_mem(7, dwell);
#endif

  // cleanup local temporary blocks 
  destroy_blocks(nblocks, tblocks, NULL);
  
#ifdef MEMORY
  get_mem(8, dwell);
#endif

  // exchange particles with neighbors 
  neighbor_particles(nblocks, particles, num_particles, num_orig_particles,
		     gids, nids, dirs);
    
#ifdef MEMORY
  get_mem(9, dwell);
#endif

  // CLP - Function to add wall-mirror particles to particles 
  //   (see neighbor_particles()) 
  add_mirror_particles(nblocks, mirror_particles,
		       num_mirror_particles, particles,num_particles,
		       num_orig_particles, gids, nids, dirs);
  
  // cleanup convex_hull_particles 
  for (i = 0; i < nblocks; ++i) {
    if (convex_hull_particles)
      free(convex_hull_particles[i]);
  }
  free(convex_hull_particles);
  free(num_convex_hull_particles);

  // CLP cleanup 
  destroy_walls(num_walls, walls);
  for (i = 0; i < nblocks; ++i) {
    if (mirror_particles[i])
      free(mirror_particles[i]);
  }
  free(num_mirror_particles);

#ifdef TIMING
  MPI_Barrier(comm);
  times[EXCH_TIME] = MPI_Wtime() - times[EXCH_TIME];
  if (rank == 0)
    fprintf(stderr, "-----------------------------------\n");
  times[CELL_TIME] = MPI_Wtime();
#endif

#ifdef MEMORY
  get_mem(10, dwell);
#endif

  // Clean-up tets; all_cells() will refill them
  for(int i = 0; i < nblocks; ++i)
    free(tets[i]);

  // create all final voronoi cells 
  all_cells(nblocks, vblocks, dim, num_particles, num_orig_particles,
	    particles, gids, nids, dirs, times, ds, &tets[0], &ntets[0]);

#ifdef MEMORY
  get_mem(11, dwell);
#endif

  // cleanup 
  for (i = 0; i < nblocks; i++) {
    free(gids[i]);
    free(nids[i]);
  }
  free(gids);
  free(nids);
  clean_delaunay_data_structures(ds);

#ifdef TIMING
  // previously no barrier here; want min and max time;
  //   changed to barrier and simple time now 
  MPI_Barrier(comm);
  times[CELL_TIME] = MPI_Wtime() - times[CELL_TIME];
  // MPI_Barrier(comm); 
  times[VOL_TIME] = MPI_Wtime();
#endif

#ifdef MEMORY
  get_mem(12, dwell);
#endif

  // compute volume and surface area manually (not using convex hulls) 
  cell_vols(nblocks, vblocks, particles);

#ifdef TIMING
  // previously no barrier here; want min and max time;
  //   changed to barrier and simple time now 
  MPI_Barrier(comm);
  times[VOL_TIME] = MPI_Wtime() - times[VOL_TIME];
  // MPI_Barrier(comm); 
  times[OUT_TIME] = MPI_Wtime();
#endif

  // prepare for output 
  prep_out(nblocks, vblocks);

  // save headers 
  save_headers(nblocks, vblocks, hdrs);

#ifdef MEMORY
  get_mem(13, dwell);
#endif

  // write output 
  if (out_file[0]) {
#ifdef PNETCDF_IO
    char out_ncfile[256];
    strncpy(out_ncfile, out_file, sizeof(out_ncfile));
    strncat(out_ncfile, ".nc", sizeof(out_file));
    pnetcdf_write(nblocks, vblocks, out_ncfile, comm);
#else
    diy_write(nblocks, vblocks, hdrs, out_file);
#endif
  }

#ifdef TIMING
  MPI_Barrier(comm);
  times[OUT_TIME] = MPI_Wtime() - times[OUT_TIME];
#endif
 
#ifdef MEMORY
  get_mem(14, dwell);
#endif

  // collect stats 
  collect_stats(nblocks, vblocks, times);

  // cleanup 
  destroy_blocks(nblocks, vblocks, hdrs);
  free(num_orig_particles);
  
  // clenaup tets
  for(int i = 0; i < nblocks; ++i)
    free(tets[i]);

#ifdef MEMORY
  get_mem(15, dwell);
#endif

}
// --------------------------------------------------------------------------
//
//   parallel delaunay tesselation
//
//   nblocks: local number of blocks
//   particles: particles[block_num][particle] 
//   where each particle is 3 values, px, py, pz
//   num_particles; number of particles in each block
//   times: times for particle exchange, voronoi cells, convex hulls, and output
//   out_file: output file name
// 
void delaunay(int nblocks, float **particles, int *num_particles, 
	      double *times, char *out_file) {

  int dim = 3; // 3D 
  int rank; // MPI rank 
  void* ds; // persistent delaunay data structures

  MPI_Comm_rank(comm, &rank);

  // init timing 
  for (int i = 0; i < MAX_TIMES; i++)
    times[i] = 0.0;

  // initialize data structures
  int **hdrs; // headers 
  struct dblock_t *dblocks; // voronoi blocks 
  ds = init_delaunay_data_structures(nblocks);
  create_dblocks(nblocks, dblocks, hdrs, particles, num_particles);
  
#ifdef MEMORY
  int dwell = 10;
  get_mem(1, dwell);
#endif

#ifdef TIMING
  MPI_Barrier(comm);
  times[LOCAL_TIME] = MPI_Wtime();
#endif

  // create local delaunay cells
  local_dcells(nblocks, dblocks, dim, ds);

  #ifdef TIMING
  MPI_Barrier(comm);
  times[LOCAL_TIME] = MPI_Wtime() - times[LOCAL_TIME];
  if (rank == 0)
    fprintf(stderr, "-----------------------------------\n");
  times[EXCH_TIME] = MPI_Wtime();
#endif

#ifdef MEMORY
  get_mem(2, dwell);
#endif

  // particles on the convex hull of the local points 
  vector <int> *convex_hull_particles  = new vector<int>[nblocks];

  // information about particles sent to neighbors
  vector <sent_t> *sent_particles = new vector<sent_t>[nblocks];

  // determine which cells are incomplete or too close to neighbor 
  for (int i = 0; i < nblocks; i++)
    incomplete_dcells_initial(&dblocks[i], i, sent_particles[i],
			      convex_hull_particles[i]);

  // debug
  fprintf(stderr, "3: num_particles[0] = %d\n", dblocks[0].num_particles);

#ifdef MEMORY
  get_mem(3, dwell);
#endif

  // cleanup local temporary blocks 
  reset_dblocks(nblocks, dblocks);

  // exhcange particles
  neighbor_d_particles(nblocks, dblocks);

  // debug
  fprintf(stderr, "5: num_particles[0] = %d\n", dblocks[0].num_particles);

#ifdef MEMORY
  get_mem(4, dwell);
#endif

  // Second, decisive phase 

  // Recompute local cells
  local_dcells(nblocks, dblocks, dim, ds);

#ifdef MEMORY
  get_mem(5, dwell);
#endif

  for (int i = 0; i < nblocks; i++)
    incomplete_dcells_final(&dblocks[i], i, sent_particles[i],
			    convex_hull_particles[i]);

#ifdef MEMORY
  get_mem(6, dwell);
#endif

  // cleanup local temporary blocks 
  reset_dblocks(nblocks, dblocks);
  
  // exchange particles with neighbors 
  neighbor_d_particles(nblocks, dblocks);
    
#ifdef MEMORY
  get_mem(7, dwell);
#endif
  
  // cleanup convex hull particles
  for (int i = 0; i < nblocks; ++i)
    convex_hull_particles[i].clear();
  delete[] convex_hull_particles;

#ifdef TIMING
  MPI_Barrier(comm);
  times[EXCH_TIME] = MPI_Wtime() - times[EXCH_TIME];
  if (rank == 0)
    fprintf(stderr, "-----------------------------------\n");
  times[CELL_TIME] = MPI_Wtime();
#endif

  // create all final cells 
  local_dcells(nblocks, dblocks, dim, ds);

#ifdef MEMORY
  get_mem(8, dwell);
#endif

  // cleanup delaunay data structure and sent particles
  clean_delaunay_data_structures(ds);

  // cleanup sent particles
  for (int i = 0; i < nblocks; ++i)
    sent_particles[i].clear();
  delete[] sent_particles;

#ifdef TIMING
  // previously no barrier here; want min and max time;
  //   changed to barrier and simple time now 
  MPI_Barrier(comm);
  times[CELL_TIME] = MPI_Wtime() - times[CELL_TIME];
  // MPI_Barrier(comm); 
  times[VOL_TIME] = MPI_Wtime();
#endif

#ifdef MEMORY
  get_mem(9, dwell);
#endif

  // prepare for output 
  prep_d_out(nblocks, dblocks, hdrs);

  // write output 
  if (out_file[0]) {
#ifdef PNETCDF_IO
    char out_ncfile[256];
    strncpy(out_ncfile, out_file, sizeof(out_ncfile));
    strncat(out_ncfile, ".nc", sizeof(out_file));
    pnetcdf_d_write(nblocks, dblocks, out_ncfile, comm);
#else
    diy_dwrite(nblocks, dblocks, hdrs, out_file);
#endif
  }

#ifdef TIMING
  MPI_Barrier(comm);
  times[OUT_TIME] = MPI_Wtime() - times[OUT_TIME];
#endif
 
  // collect stats 
  collect_dstats(nblocks, dblocks, times);

  // cleanup 
  destroy_dblocks(nblocks, dblocks, hdrs);
  
#ifdef MEMORY
  get_mem(10, dwell);
#endif

}
// --------------------------------------------------------------------------
//
// for each vertex saves a tet that contains it
//
void fill_vert_to_tet(dblock_t* dblock) {

  dblock->vert_to_tet = 
    (int*)realloc(dblock->vert_to_tet, sizeof(int) * dblock->num_particles);

  for (int t = 0; t < dblock->num_tets; ++t) {
    for (int v = 0; v < 4; ++v) {
      int p = dblock->tets[t].verts[v];
      dblock->vert_to_tet[p] = t;	// the last one wins
    }
  }

}
// --------------------------------------------------------------------------
//
//   computes volume and surface area for completed cells
//
//   nblocks: number of blocks
//   vblocks: pointer to array of vblocks
//   particles: particles in each block, particles[block_num][particle] include
//    particles received from neighbors
//
void cell_vols(int nblocks, struct vblock_t *vblocks, float **particles) {

  int b, j, f;

  // compute areas of all faces 
  face_areas(nblocks, vblocks);

  // for all blocks 
  for (b = 0; b < nblocks; b++) {

    vblocks[b].areas = (float *)malloc(vblocks[b].temp_num_complete_cells *
					   sizeof(float));
    vblocks[b].vols = (float *)malloc(vblocks[b].temp_num_complete_cells *
					   sizeof(float));

    // for all complete cells 
    for (j = 0; j < vblocks[b].temp_num_complete_cells; j++) {

      int cell = vblocks[b].temp_complete_cells[j]; // current cell 
      int num_faces; // number of faces in the current cell 
      vblocks[b].areas[j] = 0.0;
      vblocks[b].vols[j] = 0.0;
      float temp_vol = 0.0; // temporaries 
      float temp_area = 0.0;

      if (cell < vblocks[b].num_orig_particles - 1)
	num_faces = vblocks[b].cell_faces_start[cell + 1] -
	  vblocks[b].cell_faces_start[cell];
      else
	num_faces = vblocks[b].tot_num_cell_faces -
	  vblocks[b].cell_faces_start[cell];

      // for all faces 
      for (f = 0; f < num_faces; f++) {

	// current face 
	int fid = vblocks[b].cell_faces[vblocks[b].cell_faces_start[cell] + f];

	// input particles of cells sharing the face 
	float p0[3], p1[3]; 
	int p; // index of particle (could be neighbor's) 

	p = vblocks[b].faces[fid].cells[0];
	p0[0] = particles[b][3 * p];
	p0[1] = particles[b][3 * p + 1];
	p0[2] = particles[b][3 * p + 2];
	p = vblocks[b].faces[fid].cells[1];
	p1[0] = particles[b][3 * p];
	p1[1] = particles[b][3 * p + 1];
	p1[2] = particles[b][3 * p + 2];

	// height of pyramid from site to face = 
	//   distance between sites sharing the face / 2 
	float height = sqrt((p0[0] - p1[0]) * (p0[0] - p1[0]) +
			    (p0[1] - p1[1]) * (p0[1] - p1[1]) +
			    (p0[2] - p1[2]) * (p0[2] - p1[2])) / 2.0;

	// add the volume of the pyramid formed by site and current face
	//   to the volume of the cell and add the face area to the surface
	//   area of the cell 
	temp_vol += vblocks[b].face_areas[fid] * height / 3.0;
	temp_area += vblocks[b].face_areas[fid];

      } // for all faces 

      // store the cell permanently if it passes the volume thresholds 
      if ((min_vol < 0 || temp_vol >= min_vol) &&
	  (max_vol < 0 || temp_vol <= max_vol)) {

	vblocks[b].vols[j] = temp_vol;
	vblocks[b].areas[j] = temp_area;
	vblocks[b].complete_cells[vblocks[b].num_complete_cells] =
	  vblocks[b].temp_complete_cells[j];
	vblocks[b].num_complete_cells++;

      }

    } // for all complete cells 

  } // for all blocks 

}
// --------------------------------------------------------------------------
//
//   computes areas of all faces
//
//   nblocks: number of blocks
//   vblocks: pointer to array of vblocks
//
void face_areas(int nblocks, struct vblock_t *vblocks) {

  int b, f, v;

  // for all blocks 
  for (b = 0; b < nblocks; b++) {

    vblocks[b].face_areas = (float *)malloc(vblocks[b].num_faces *
					    sizeof(float));

    // for all faces 
    for (f = 0; f < vblocks[b].num_faces; f++) {

      vblocks[b].face_areas[f] = 0.0;

      // all triangles fan out from same vertex 
      int v0 = vblocks[b].faces[f].verts[0];

      // for all vertices in a face 
      for (v = 2; v <vblocks[b].faces[f].num_verts; v++) {

	// remaining 2 vertices of one triangle in the polygon 
	int v1 = vblocks[b].faces[f].verts[v - 1];
	int v2 = vblocks[b].faces[f].verts[v];

	// vectors for two sides of triangle v1v1 and v0v2 
	float s1[3], s2[3]; 
	s1[0] = vblocks[b].verts[3 * v1] - 
	  vblocks[b].verts[3 * v0];
	s1[1] = vblocks[b].verts[3 * v1 + 1] - 
	  vblocks[b].verts[3 * v0 + 1];
	s1[2] = vblocks[b].verts[3 * v1 + 2] - 
	  vblocks[b].verts[3 * v0 + 2];
	s2[0] = vblocks[b].verts[3 * v2] - 
	  vblocks[b].verts[3 * v0];
	s2[1] = vblocks[b].verts[3 * v2 + 1] - 
	  vblocks[b].verts[3 * v0 + 1];
	s2[2] = vblocks[b].verts[3 * v2 + 2] - 
	  vblocks[b].verts[3 * v0 + 2];

	// cross product of s1 and s2 
	float c[3];
	c[0] = s1[1] * s2[2] - s1[2] * s2[1];
	c[1] = s1[2] * s2[0] - s1[0] * s2[2];
	c[2] = s1[0] * s2[1] - s1[1] * s2[0];

	// area of triangle is |c| / 2 
	float a = sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]) / 2.0;
	vblocks[b].face_areas[f] += a;

      } // for all vertices 

    } // for all faces 

  } // for all blocks 

}
// --------------------------------------------------------------------------
//
//   exchanges particles with neighbors
//
//   nblocks: local number of blocks
//   particles: particles before and after neighbor exchange (input / output)
//   num_particles: number of new particles in each block (input / output)
//   gids: global block ids of owners of received particles in each of my blocks
//   nids: native particle ids of received particles in each of my blocks
//    (allocated by this function, user's responsibility to free)
//   dirs: wrapping direction of received particles in each of my blocks
//
//   to send the site to the neighbor
// 
void neighbor_particles(int nblocks, float **particles,
			int *num_particles, int *num_orig_particles,
			int **gids, int **nids, unsigned char **dirs) {

  void ***recv_particles; // pointers to particles in ecah block 
			  //   that are received from neighbors 
  int *num_recv_particles; // number of received particles for each block 
  int i, j;

  recv_particles = (void ***)malloc(nblocks * sizeof(void **));
  num_recv_particles = (int *)malloc(nblocks * sizeof(int));

  // particles were previously enqueued by local_cells(), ready to
  //   be exchanged 
  DIY_Exchange_neighbors(0, recv_particles, num_recv_particles, 1.0, 
			 &item_type);

  // copy received particles to particles 
  for (i = 0; i < nblocks; i++) {

    int n = (num_particles[i] - num_orig_particles[i]);
    int new_remote_particles = num_recv_particles[i] + n;

    gids[i] = (int *)realloc(gids[i], new_remote_particles * sizeof(int));
    nids[i] = (int *)realloc(nids[i], new_remote_particles * sizeof(int));
    dirs[i] = (unsigned char *)realloc(dirs[i], new_remote_particles);
 
    if (num_recv_particles[i]) {

      // grow space 
      particles[i] = 
	(float *)realloc(particles[i], 
			 (num_particles[i] + num_recv_particles[i]) *
			 3 * sizeof(float));

      // copy received particles 
      for (j = 0; j < num_recv_particles[i]; j++) { 

	particles[i][3 * num_particles[i]] =
	  DIY_Exchd_item(struct remote_particle_t, recv_particles, i, j)->x;
	particles[i][3 * num_particles[i] + 1] =
	  DIY_Exchd_item(struct remote_particle_t, recv_particles, i, j)->y;
	particles[i][3 * num_particles[i] + 2] =
	  DIY_Exchd_item(struct remote_particle_t, recv_particles, i, j)->z;
	gids[i][n] = 
	  DIY_Exchd_item(struct remote_particle_t, recv_particles, i, j)->gid;
	nids[i][n] = 
	  DIY_Exchd_item(struct remote_particle_t, recv_particles, i, j)->nid;
	dirs[i][n] = 
	  DIY_Exchd_item(struct remote_particle_t, recv_particles, i, j)->dir;

	num_particles[i]++;
	n++;

      } // copy received particles 

    } // if num_recv_particles 

  } // for all blocks 

  // clean up 
  DIY_Flush_neighbors(0, recv_particles, num_recv_particles, &item_type);
  free(num_recv_particles);
  free(recv_particles);

}
// --------------------------------------------------------------------------
//
//   exchanges particles with neighbors
//
//   nblocks: local number of blocks
//   dblocks: local blocks
//
void neighbor_d_particles(int nblocks, dblock_t *dblocks) {

  void ***recv_particles; // pointers to particles in ecah block 
			  //   that are received from neighbors 
  int *num_recv_particles; // number of received particles for each block 
  int i, j;

  recv_particles = (void ***)malloc(nblocks * sizeof(void **));
  num_recv_particles = (int *)malloc(nblocks * sizeof(int));

  // particles were previously enqueued by local_cells(), ready to
  //   be exchanged 
  DIY_Exchange_neighbors(0, recv_particles, num_recv_particles, 1.0, 
			 &item_type);

  // copy received particles to dblock
  for (i = 0; i < nblocks; i++) {

    int n = (dblocks[i].num_particles - dblocks[i].num_orig_particles);
    int new_remote_particles = num_recv_particles[i] + n;
    dblocks[i].num_rem_tet_verts = new_remote_particles;
    dblocks[i].rem_tet_verts = 
      (struct remote_vert_t *)realloc(dblocks[i].rem_tet_verts, 
				      new_remote_particles * 
				      sizeof(struct remote_vert_t));
 
    if (num_recv_particles[i]) {

      // grow space 
      dblocks[i].particles = 
	(float *)realloc(dblocks[i].particles, 
			 (dblocks[i].num_particles + num_recv_particles[i]) *
			 3 * sizeof(float));

      // copy received particles 
      for (j = 0; j < num_recv_particles[i]; j++) { 

	dblocks[i].particles[3 * dblocks[i].num_particles] =
	  DIY_Exchd_item(struct remote_particle_t, recv_particles, i, j)->x;
	dblocks[i].particles[3 * dblocks[i].num_particles + 1] =
	  DIY_Exchd_item(struct remote_particle_t, recv_particles, i, j)->y;
	dblocks[i].particles[3 * dblocks[i].num_particles + 2] =
	  DIY_Exchd_item(struct remote_particle_t, recv_particles, i, j)->z;
	dblocks[i].rem_tet_verts[n].gid = 
	  DIY_Exchd_item(struct remote_particle_t, recv_particles, i, j)->gid;
	dblocks[i].rem_tet_verts[n].nid = 
	  DIY_Exchd_item(struct remote_particle_t, recv_particles, i, j)->nid;
	dblocks[i].rem_tet_verts[n].dir = 
	  DIY_Exchd_item(struct remote_particle_t, recv_particles, i, j)->dir;

	dblocks[i].num_particles++;
	n++;

      } // copy received particles 

    } // if num_recv_particles 

  } // for all blocks 

  // clean up 
  DIY_Flush_neighbors(0, recv_particles, num_recv_particles, &item_type);
  free(num_recv_particles);
  free(recv_particles);

}
// --------------------------------------------------------------------------
//
//   exchanges is_complete list for exchanged particles with neighbors
//
//   nblocks: number of blocks
//   vblocks: local blocks
//   rics: completion satus of received particles in each of my blocks
//    (allocated by this function, user's responsibility to free)
//
// 
void neighbor_is_complete(int nblocks, struct vblock_t *vblocks,
			  struct remote_ic_t **rics) {

  void ***recv_ics; // pointers to is_complete entries in ecah block 
                    //  that are received from neighbors 
  int *num_recv_ics; // number of received is_completes for each block 
  int i, j;
  struct remote_ic_t ic; // completion status being sent or received 

  recv_ics = (void ***)malloc(nblocks * sizeof(void **));
  num_recv_ics = (int *)malloc(nblocks * sizeof(int));

  // for all blocks 
  for (i = 0; i < nblocks; i++) {

    // for all particles in the current block 
    for (j = 0; j < vblocks[i].num_sent_particles; j++) {
      int p = vblocks[i].sent_particles[j].particle;
      ic.is_complete = vblocks[i].is_complete[p];
      ic.gid = DIY_Gid(0, i);
      ic.nid = p;
      DIY_Enqueue_item_gbs(0, i, (void *)&ic,
			   NULL, sizeof(struct remote_ic_t),
			   vblocks[i].sent_particles[j].neigh_gbs,
			   vblocks[i].sent_particles[j].num_gbs, NULL);
    }

  } // for all blocks 

  // exchange neighbors 
  DIY_Exchange_neighbors(0, recv_ics, num_recv_ics, 1.0, &ic_type);

  // copy received is_completed entries 
  for (i = 0; i < nblocks; i++) {

    rics[i] = (struct remote_ic_t *)malloc(num_recv_ics[i] * 
					   sizeof(struct remote_ic_t));

    for (j = 0; j < num_recv_ics[i]; j++) {
      rics[i][j].is_complete = 
	DIY_Exchd_item(struct remote_ic_t, recv_ics, i, j)->is_complete;
      rics[i][j].gid = 
	DIY_Exchd_item(struct remote_ic_t, recv_ics, i, j)->gid;
      rics[i][j].nid = 
	DIY_Exchd_item(struct remote_ic_t, recv_ics, i, j)->nid;
    }

  }

  // clean up 
  DIY_Flush_neighbors(0, recv_ics, num_recv_ics, &ic_type);
  free(num_recv_ics);
  free(recv_ics);

}
// --------------------------------------------------------------------------
//
//  makes DIY datatype for sending / receiving one item
// 
void item_type(DIY_Datatype *dtype) {

  struct map_block_t map[] = {
    {DIY_FLOAT, OFST, 1, offsetof(struct remote_particle_t, x)         },
    {DIY_FLOAT, OFST, 1, offsetof(struct remote_particle_t, y)         },
    {DIY_FLOAT, OFST, 1, offsetof(struct remote_particle_t, z)         },
    {DIY_INT,   OFST, 1, offsetof(struct remote_particle_t, gid)       },
    {DIY_INT,   OFST, 1, offsetof(struct remote_particle_t, nid)       },
    {DIY_BYTE,  OFST, 1, offsetof(struct remote_particle_t, dir)       },
  };
  DIY_Create_struct_datatype(0, 6, map, dtype);

}
// --------------------------------------------------------------------------
//
//  makes DIY datatype for sending / receiving one is_complete entry
// 
void ic_type(DIY_Datatype *dtype) {

  struct map_block_t map[] = {
    {DIY_INT, OFST, 1, offsetof(struct remote_ic_t, is_complete) },
    {DIY_INT, OFST, 1, offsetof(struct remote_ic_t, gid)         },
    {DIY_INT, OFST, 1, offsetof(struct remote_ic_t, nid)         },
  };
  DIY_Create_struct_datatype(0, 3, map, dtype);

}
// --------------------------------------------------------------------------
//
//   collects statistics
//
//   nblocks: number of blocks
//   vblocks: pointer to array of vblocks
//   times: timing info
// 
void collect_stats(int nblocks, struct vblock_t *vblocks, double *times) {

  int i, j, k, m, n, v, f;
  int tot_num_cell_verts = 0; // number of cell verts in all local blocks 
  int tot_num_face_verts = 0; // number of face verts in all local blocks 
  int *unique_verts = NULL; // unique vertices in one cell 
  int num_unique_verts; // number of unique vertices in one cell
  static int max_unique_verts; // allocated number of unique vertices 
  int chunk_size = 128; // allocation chunk size for unique_verts 
  float vol_bin_width; // width of a volume histogram bin 
  float dense_bin_width; // width of a density histogram bin 
  float tot_cell_vol = 0.0; // sum of cell volumes 
  float tot_cell_dense = 0.0; // sum of cell densities 
  struct stats_t stats; // local stats 
  static int first_dense = 1; // first density value saved 
  int rank;

  MPI_Comm_rank(comm, &rank);

  // timing range 
  stats.min_cell_time = times[CELL_TIME];
  stats.max_cell_time = times[CELL_TIME];
  stats.min_vol_time = times[VOL_TIME];
  stats.max_vol_time = times[VOL_TIME];

  // --- first pass: find average number of vertices per cell and 
  //   volume range --- 

  stats.tot_tets  = 0;
  stats.tot_cells = 0;
  stats.tot_faces = 0;
  stats.tot_verts = 0;
  stats.avg_cell_verts = 0.0;
  stats.avg_cell_faces = 0.0;
  stats.avg_face_verts = 0.0;
  stats.avg_cell_vol   = 0.0;
  stats.avg_cell_dense = 0.0;

  for (i = 0; i < nblocks; i++) { // for all blocks 

    stats.tot_tets += (vblocks[i].num_loc_tets + vblocks[i].num_rem_tets);
    stats.tot_cells += vblocks[i].num_complete_cells;
    stats.tot_verts += vblocks[i].num_verts;

    // for all complete cells in the current block 
    f = 0;
    v = 0;
    for (j = 0; j < vblocks[i].num_complete_cells; j++) {

      if (vblocks[i].vols[j] == 0.0)
	fprintf(stderr, "found cell with 0.0 volume--this should not happen\n");

      tot_cell_vol += vblocks[i].vols[j];
      float dense = 0.0;
      if (vblocks[i].vols[j] > 0.0) {
	dense = 1.0 / vblocks[i].vols[j];
	tot_cell_dense += dense;
      }

      int cell = vblocks[i].complete_cells[j]; // current cell 
      int num_faces; // number of face in the current cell 
      int num_verts; // number of vertices in current face 

      if (cell < vblocks[i].num_orig_particles - 1)
	num_faces = vblocks[i].cell_faces_start[cell + 1] -
	  vblocks[i].cell_faces_start[cell];
      else
	num_faces = vblocks[i].tot_num_cell_faces -
	  vblocks[i].cell_faces_start[cell];

      stats.tot_faces += num_faces; // not unique, but total for all cells 
      num_unique_verts = 0;

      // volume range 
      if (i == 0 && j == 0) {
	stats.min_cell_vol = vblocks[i].vols[j];
	stats.max_cell_vol = vblocks[i].vols[j];
      }
      else {
	if (vblocks[i].vols[j] < stats.min_cell_vol)
	  stats.min_cell_vol = vblocks[i].vols[j];
	if (vblocks[i].vols[j] > stats.max_cell_vol)
	  stats.max_cell_vol = vblocks[i].vols[j];
      }

      // density range 
      if (first_dense && vblocks[i].vols[j] > 0.0) {
	stats.min_cell_dense = dense;
	stats.max_cell_dense = stats.min_cell_dense;
	first_dense = 0;
      }
      else if (vblocks[i].vols[j] > 0.0) {
	if (dense < stats.min_cell_dense)
	  stats.min_cell_dense = dense;
	if (dense > stats.max_cell_dense)
	  stats.max_cell_dense = dense;
      }

      // for all faces in the current cell 
      for (k = 0; k < num_faces; k++) {

	int start = vblocks[i].cell_faces_start[cell];
	int face = vblocks[i].cell_faces[start + k];
	num_verts = vblocks[i].faces[face].num_verts;

	tot_num_face_verts += num_verts;

	// for all verts in the current face 
	for (m = 0; m < num_verts; m++) {

	  // check if we already counted it 
	  for (n = 0; n < num_unique_verts; n++) {
	    if (vblocks[i].faces[face].verts[m] == unique_verts[n])
	      break;
	  }
	  if (n == num_unique_verts)
	    add_int(vblocks[i].faces[face].verts[m], &unique_verts, 
		    &num_unique_verts, &max_unique_verts, chunk_size);
	  v++;

	} // for all verts 

	f++;

      } // for all faces 

      tot_num_cell_verts += num_unique_verts;

    } // for all complete cells 


  } // for all blocks 

  free(unique_verts);

  // compute local averages 
  if (stats.tot_cells) { // don't divide by 0 
    stats.avg_cell_verts = tot_num_cell_verts / stats.tot_cells;
    stats.avg_cell_faces = stats.tot_faces / stats.tot_cells;
    stats.avg_face_verts = tot_num_face_verts / stats.tot_faces;
    stats.avg_cell_vol = tot_cell_vol / stats.tot_cells;
    stats.avg_cell_dense = tot_cell_dense / stats.tot_cells;
  }

  // aggregate totals across all procs and compute average on that 
  aggregate_stats(nblocks, vblocks, &stats);

  // --- print output --- 

  // global stats 
  vol_bin_width = (stats.max_cell_vol - stats.min_cell_vol) / 
    stats.num_vol_bins;
  dense_bin_width = (stats.max_cell_dense - stats.min_cell_dense) / 
    stats.num_dense_bins;
  if (rank == 0) {
    fprintf(stderr, "----------------- global stats ------------------\n");
    fprintf(stderr, "local voronoi / delaunay time = %.3lf s\n",
	    times[LOCAL_TIME]);
    fprintf(stderr, "particle exchange time = %.3lf s\n", times[EXCH_TIME]);
    // fprintf(stderr, "[min, max] voronoi / delaunay time = [%.3lf, %.3lf] s\n", 
    // 	    stats.min_cell_time, stats.max_cell_time); 
    fprintf(stderr, "Voronoi / delaunay time = %.3lf s\n", times[CELL_TIME]);
    // fprintf(stderr, "[min, max] cell volume / area time = [%.3lf, %.3lf] s\n", 
    // 	    stats.min_vol_time, stats.max_vol_time); 
    fprintf(stderr, "Cell volume / area time = %.3lf s\n", times[VOL_TIME]);
    fprintf(stderr, "output time = %.3lf s\n", times[OUT_TIME]);
    fprintf(stderr, "-----\n");
    fprintf(stderr, "total tets found = %d\n", stats.tot_tets);
    fprintf(stderr, "total cells found = %d\n", stats.tot_cells);
    fprintf(stderr, "total cell vertices found = %d\n", stats.tot_verts);
    fprintf(stderr, "average number of vertices per cell = %.0lf\n",
	    stats.avg_cell_verts);
    fprintf(stderr, "average number of faces per cell = %.0lf\n",
	    stats.avg_cell_faces);
    fprintf(stderr, "average number of vertices per face = %.0lf\n",
	    stats.avg_face_verts);
    fprintf(stderr, "-----\n");
    fprintf(stderr, "min cell volume = %.3lf max cell volume = %.3lf "
	    "avg cell volume = %.3lf units^3\n",
	    stats.min_cell_vol, stats.max_cell_vol, stats.avg_cell_vol);
    fprintf(stderr, "number of cell volume histogram bins = %d\n",
	    stats.num_vol_bins);
    fprintf(stderr, "-----\n");
    fprintf(stderr, "cell volume histogram:\n");
    fprintf(stderr, "min value\tcount\t\tmax value\n");
    for (k = 0; k < stats.num_vol_bins; k++)
      fprintf(stderr, "%.3lf\t\t%d\t\t%.3lf\n", 
	      stats.min_cell_vol + k * vol_bin_width, stats.vol_hist[k], 
	      stats.min_cell_vol + (k + 1) * vol_bin_width);
    fprintf(stderr, "-----\n");
    fprintf(stderr, "min cell density = %.3lf max cell density = %.3lf "
	    "avg cell density = %.3lf units^3\n",
	    stats.min_cell_dense, stats.max_cell_dense, stats.avg_cell_dense);
    fprintf(stderr, "-----\n");
    fprintf(stderr, "cell density histogram:\n");
    fprintf(stderr, "min value\tcount\t\tmax value\n");
    for (k = 0; k < stats.num_dense_bins; k++)
      fprintf(stderr, "%.3lf\t\t%d\t\t%.3lf\n", 
	      stats.min_cell_dense + k * dense_bin_width, stats.dense_hist[k], 
	      stats.min_cell_dense + (k + 1) * dense_bin_width);
    fprintf(stderr, "-------------------------------------------------\n");
  }

}
// --------------------------------------------------------------------------
//
//   collects statistics
//
//   nblocks: number of blocks
//   dblocks: local delaunay blocks
//   times: timing info
// 
void collect_dstats(int nblocks, struct dblock_t *dblocks, double *times) {

  int rank;

  MPI_Comm_rank(comm, &rank);

  // --- print output --- 

  // global stats 
  if (rank == 0) {
    fprintf(stderr, "----------------- global stats ------------------\n");
    fprintf(stderr, "local delaunay time = %.3lf s\n",
	    times[LOCAL_TIME]);
    fprintf(stderr, "particle exchange time = %.3lf s\n", times[EXCH_TIME]);
    fprintf(stderr, "Voronoi / delaunay time = %.3lf s\n", times[CELL_TIME]);
    fprintf(stderr, "output time = %.3lf s\n", times[OUT_TIME]);
    fprintf(stderr, "-------------------------------------------------\n");
  }

}
// --------------------------------------------------------------------------
//
//   aggregates local statistics into global statistics
//
//   nblocks: number of blocks
//   vblocks: pointer to array of vblocks
//   loc_stats: local statistics
// 
void aggregate_stats(int nblocks, struct vblock_t *vblocks, 
		     struct stats_t *loc_stats) {

  float vol_bin_width; // width of a volume histogram bin 
  float dense_bin_width; // width of a density histogram bin 
  struct stats_t glo_stats; // global stats 
  struct stats_t fin_stats; // final (global) stats 
  int groupsize; // MPI usual 
  DIY_Datatype dtype; // custom datatype 
  MPI_Op op1, op2; // custom operators 
  int i, j, k;

  MPI_Comm_size(comm, &groupsize);

  if (groupsize > 1) {

    // create datatype 
    struct map_block_t map[] = {

      { DIY_INT,   OFST, 1, 
	offsetof(struct stats_t, tot_tets)      },
      { DIY_INT,   OFST, 1, 
	offsetof(struct stats_t, tot_cells)      },
      { DIY_INT,   OFST, 1, 
	offsetof(struct stats_t, tot_faces)      },
      { DIY_INT,   OFST, 1, 
	offsetof(struct stats_t, tot_verts)      },
      { DIY_FLOAT, OFST, 1, 
	offsetof(struct stats_t, avg_cell_verts) },
      { DIY_FLOAT, OFST, 1, 
	offsetof(struct stats_t, avg_cell_faces) },
      { DIY_FLOAT, OFST, 1, 
	offsetof(struct stats_t, avg_face_verts) },
      { DIY_FLOAT, OFST, 1, 
	offsetof(struct stats_t, min_cell_vol)   },
      { DIY_FLOAT, OFST, 1, 
	offsetof(struct stats_t, max_cell_vol)   },
      { DIY_FLOAT, OFST, 1, 
	offsetof(struct stats_t, avg_cell_vol)   },
      { DIY_FLOAT, OFST, 1, 
	offsetof(struct stats_t, min_cell_dense) },
      { DIY_FLOAT, OFST, 1, 
	offsetof(struct stats_t, max_cell_dense) },
      { DIY_FLOAT, OFST, 1, 
	offsetof(struct stats_t, avg_cell_dense) },
      { DIY_FLOAT, OFST, 1, 
	offsetof(struct stats_t, min_cell_time)  },
      { DIY_FLOAT, OFST, 1, 
	offsetof(struct stats_t, max_cell_time)  },
      { DIY_FLOAT, OFST, 1, 
	offsetof(struct stats_t, min_vol_time)  },
      { DIY_FLOAT, OFST, 1, 
	offsetof(struct stats_t, max_vol_time)  },
      { DIY_INT,   OFST, 1, 
	offsetof(struct stats_t, num_vol_bins)   },
      { DIY_INT,   OFST, 1, 
	offsetof(struct stats_t, num_dense_bins) },
      { DIY_INT,   OFST, MAX_HIST_BINS, 
	offsetof(struct stats_t, vol_hist)       },
      { DIY_INT,   OFST, MAX_HIST_BINS, 
	offsetof(struct stats_t, dense_hist)     },

    };

    DIY_Create_struct_datatype(0, 21, map, &dtype);

    MPI_Op_create(&average, 1, &op1);
    MPI_Op_create(&histogram, 1, &op2);

    // first reduction computes averages and ranges 
    MPI_Reduce(loc_stats, &glo_stats, 1, dtype, op1, 0, comm);
    // broadcast global stats to all process 
    MPI_Bcast(&glo_stats, 1, dtype, 0, comm);

  }

  else {

    glo_stats.tot_tets       = loc_stats->tot_tets;
    glo_stats.tot_cells      = loc_stats->tot_cells;
    glo_stats.tot_faces      = loc_stats->tot_faces;
    glo_stats.tot_verts      = loc_stats->tot_verts;
    glo_stats.avg_cell_verts = loc_stats->avg_cell_verts;
    glo_stats.avg_cell_faces = loc_stats->avg_cell_faces;
    glo_stats.avg_face_verts = loc_stats->avg_face_verts;
    glo_stats.min_cell_vol   = loc_stats->min_cell_vol;
    glo_stats.max_cell_vol   = loc_stats->max_cell_vol;
    glo_stats.avg_cell_vol   = loc_stats->avg_cell_vol;
    glo_stats.min_cell_dense = loc_stats->min_cell_dense;
    glo_stats.max_cell_dense = loc_stats->max_cell_dense;
    glo_stats.avg_cell_dense = loc_stats->avg_cell_dense;
    glo_stats.min_cell_time  = loc_stats->min_cell_time;
    glo_stats.max_cell_time  = loc_stats->max_cell_time;
    glo_stats.min_vol_time  = loc_stats->min_vol_time;
    glo_stats.max_vol_time  = loc_stats->max_vol_time;
    glo_stats.num_vol_bins   = 50;
    glo_stats.num_dense_bins = 100;

  }

  // find local cell volume and density histograms 
  vol_bin_width = (glo_stats.max_cell_vol - glo_stats.min_cell_vol) / 
    glo_stats.num_vol_bins; // volume 
  dense_bin_width = (glo_stats.max_cell_dense - glo_stats.min_cell_dense) / 
    glo_stats.num_dense_bins; // density 
  for (k = 0; k < glo_stats.num_vol_bins; k++) // volume 
    glo_stats.vol_hist[k] = 0;
  for (k = 0; k < glo_stats.num_dense_bins; k++) // density 
    glo_stats.dense_hist[k] = 0;
  for (i = 0; i < nblocks; i++) { // for all blocks 

    for (j = 0; j < vblocks[i].num_complete_cells; j++) { // for all cells 

      // volume 
      for (k = 0; k < glo_stats.num_vol_bins; k++) { // for all bins 
	if (vblocks[i].vols[j] >= glo_stats.min_cell_vol + k * vol_bin_width && 
	    vblocks[i].vols[j] < 
	    glo_stats.min_cell_vol + (k + 1) * vol_bin_width) {
	  glo_stats.vol_hist[k]++;
	  break;
	}
      } // for all bins 
      if (k == glo_stats.num_vol_bins)
	glo_stats.vol_hist[k - 1]++; // catch roundoff error and open
				     // interval on right side of bin 

      // density 
      for (k = 0; k < glo_stats.num_dense_bins; k++) { // for all bins 
	if (vblocks[i].vols[j] > 0.0) {
	  float dense = 1.0 /vblocks[i].vols[j];
	  if (dense >= glo_stats.min_cell_dense + k * dense_bin_width && 
	      dense <  glo_stats.min_cell_dense + (k + 1) * dense_bin_width) {
	    glo_stats.dense_hist[k]++;
	    break;
	  }
	}
      } // for all bins 
      if (k == glo_stats.num_dense_bins)
	glo_stats.dense_hist[k - 1]++; // catch roundoff error and open
				       // interval on right side of bin 

    } // for all cells 

  } // for all blocks 

  if (groupsize > 1) {

    // second reduction computes global histogram 
    MPI_Reduce(&glo_stats, &fin_stats, 1, dtype, op2, 0, comm);

    // copy global stats back to local stats 
    loc_stats->tot_tets      = fin_stats.tot_tets;
    loc_stats->tot_cells      = fin_stats.tot_cells;
    loc_stats->tot_faces      = fin_stats.tot_faces;
    loc_stats->tot_verts      = fin_stats.tot_verts;
    loc_stats->avg_cell_verts = fin_stats.avg_cell_verts;
    loc_stats->avg_cell_faces = fin_stats.avg_cell_faces;
    loc_stats->avg_face_verts = fin_stats.avg_face_verts;
    loc_stats->min_cell_vol   = fin_stats.min_cell_vol;
    loc_stats->max_cell_vol   = fin_stats.max_cell_vol;
    loc_stats->avg_cell_vol   = fin_stats.avg_cell_vol;
    loc_stats->min_cell_dense = fin_stats.min_cell_dense;
    loc_stats->max_cell_dense = fin_stats.max_cell_dense;
    loc_stats->avg_cell_dense = fin_stats.avg_cell_dense;
    loc_stats->min_cell_time  = fin_stats.min_cell_time;
    loc_stats->max_cell_time  = fin_stats.max_cell_time;
    loc_stats->min_vol_time  = fin_stats.min_vol_time;
    loc_stats->max_vol_time  = fin_stats.max_vol_time;
    loc_stats->num_vol_bins   = fin_stats.num_vol_bins;
    loc_stats->num_dense_bins   = fin_stats.num_dense_bins;
    for (i = 0; i < MAX_HIST_BINS; i++) {
      loc_stats->vol_hist[i] = fin_stats.vol_hist[i];
      loc_stats->dense_hist[i] = fin_stats.dense_hist[i];
    }

    DIY_Destroy_datatype(&dtype);
    MPI_Op_free(&op1);
    MPI_Op_free(&op2);

  }
  else {

    loc_stats->num_vol_bins   = glo_stats.num_vol_bins;
    loc_stats->num_dense_bins = glo_stats.num_dense_bins;
    for (i = 0; i < MAX_HIST_BINS; i++) {
      loc_stats->vol_hist[i] = glo_stats.vol_hist[i];
      loc_stats->dense_hist[i] = glo_stats.dense_hist[i];
    }

  }

}
// --------------------------------------------------------------------------
//
//   reduces averages and ranges
//
//   in: input 1
//   inout: input 2 and output
//   len: 1
//   datatype: unused
// 
void average(void *in, void *inout, int *len, MPI_Datatype *type) {

  // quiet compiler warnings about unused variables 
  type = type;
  len = len;

  struct stats_t *stats1 = (struct stats_t *)in;
  struct stats_t *stats2 = (struct stats_t *)inout;

  // weights for weighted averages based on cell counts 
  float w1 = (float)stats1->tot_cells / 
    (stats1->tot_cells + stats2->tot_cells);
  float w2 = (float)stats2->tot_cells / 
    (stats1->tot_cells + stats2->tot_cells);

  // weighted average of two averages 
  stats2->avg_cell_verts = w1 * stats1->avg_cell_verts +
    w2 * stats2->avg_cell_verts;
  stats2->avg_cell_faces = w1 * stats1->avg_cell_faces +
    w2 * stats2->avg_cell_faces;
  stats2->avg_cell_vol = w1 * stats1->avg_cell_vol +
    w2 * stats2->avg_cell_vol;
  stats2->avg_cell_dense = w1 * stats1->avg_cell_dense +
    w2 * stats2->avg_cell_dense;

  // new weights for weighted averages based on face counts 
  w1 = (float)stats1->tot_faces / 
    (stats1->tot_faces + stats2->tot_faces);
  w2 = (float)stats2->tot_faces / 
    (stats1->tot_faces + stats2->tot_faces);

  // weighted average of two averages 
  stats2->avg_face_verts = w1 * stats1->avg_face_verts + 
    w2 * stats2->avg_face_verts;

  stats2->tot_tets += stats1->tot_tets;
  stats2->tot_cells += stats1->tot_cells;
  stats2->tot_verts += stats1->tot_verts;

  if (stats1->min_cell_vol < stats2->min_cell_vol)
    stats2->min_cell_vol = stats1->min_cell_vol;
  if (stats1->max_cell_vol > stats2->max_cell_vol)
    stats2->max_cell_vol = stats1->max_cell_vol;
  if (stats1->min_cell_dense < stats2->min_cell_dense)
    stats2->min_cell_dense = stats1->min_cell_dense;
  if (stats1->max_cell_dense > stats2->max_cell_dense)
    stats2->max_cell_dense = stats1->max_cell_dense;
  if (stats1->min_cell_time < stats2->min_cell_time)
    stats2->min_cell_time = stats1->min_cell_time;
  if (stats1->max_cell_time > stats2->max_cell_time)
    stats2->max_cell_time = stats1->max_cell_time;
  if (stats1->min_vol_time < stats2->min_vol_time)
    stats2->min_vol_time = stats1->min_vol_time;
  if (stats1->max_vol_time > stats2->max_vol_time)
    stats2->max_vol_time = stats1->max_vol_time;

  // ought to do a cross-validation to find correct number of bins
  //   for now just pick a number 
  stats2->num_vol_bins = 50;
  stats2->num_dense_bins = 100;

}
// --------------------------------------------------------------------------
//
//   reduces histograms
//
//   in: input 1
//   inout: input 2 and output
//   len: 1
//   datatype: unused
// 
void histogram(void *in, void *inout, int *len, MPI_Datatype *type) {

  // quiet compiler warnings about unused variables 
  type = type;
  len = len;

  struct stats_t *stats1 = (struct stats_t *)in;
  struct stats_t *stats2 = (struct stats_t *)inout;
  int i;

  stats2->tot_tets      = stats1->tot_tets;
  stats2->tot_cells      = stats1->tot_cells;
  stats2->tot_faces      = stats1->tot_faces;
  stats2->tot_verts      = stats1->tot_verts;
  stats2->avg_cell_verts = stats1->avg_cell_verts;
  stats2->avg_cell_faces = stats1->avg_cell_faces;
  stats2->avg_face_verts = stats1->avg_face_verts;
  stats2->min_cell_vol   = stats1->min_cell_vol;
  stats2->max_cell_vol   = stats1->max_cell_vol;
  stats2->avg_cell_vol   = stats1->avg_cell_vol;
  stats2->min_cell_dense = stats1->min_cell_dense;
  stats2->max_cell_dense = stats1->max_cell_dense;
  stats2->avg_cell_dense = stats1->avg_cell_dense;
  stats2->min_cell_time  = stats1->min_cell_time;
  stats2->max_cell_time  = stats1->max_cell_time;
  stats2->min_vol_time  = stats1->min_vol_time;
  stats2->max_vol_time  = stats1->max_vol_time;
  stats2->num_vol_bins   = stats1->num_vol_bins;
  stats2->num_dense_bins = stats1->num_dense_bins;
  for (i = 0; i < stats2->num_vol_bins; i++)
    stats2->vol_hist[i] += stats1->vol_hist[i];
  for (i = 0; i < stats2->num_dense_bins; i++)
    stats2->dense_hist[i] += stats1->dense_hist[i];

}
// --------------------------------------------------------------------------
//
//   prepare for output
//
//   nblocks: number of blocks
//   vblocks: pointer to array of vblocks
// 
void prep_out(int nblocks, struct vblock_t *vblocks) {

  struct bb_t bounds; // block bounds 
  int i, j;

  // save extents 
  for (i = 0; i < nblocks; i++) {
    DIY_Block_bounds(0, i, &bounds);
    vblocks[i].mins[0] = bounds.min[0];
    vblocks[i].mins[1] = bounds.min[1];
    vblocks[i].mins[2] = bounds.min[2];
    vblocks[i].maxs[0] = bounds.max[0];
    vblocks[i].maxs[1] = bounds.max[1];
    vblocks[i].maxs[2] = bounds.max[2];
  }

  // save vertices (float version) 
  for (i = 0; i < nblocks; i++) {
    vblocks[i].save_verts = (float *)malloc(vblocks[i].num_verts * 3 * 
					       sizeof(float));
    for (j = 0; j < vblocks[i].num_verts; j++) {

        vblocks[i].save_verts[3 * j]     = vblocks[i].verts[3 * j];
        vblocks[i].save_verts[3 * j + 1] = vblocks[i].verts[3 * j + 1];
        vblocks[i].save_verts[3 * j + 2] = vblocks[i].verts[3 * j + 2];

    }
    free (vblocks[i].verts);
    vblocks[i].verts = NULL;
  }

}
// --------------------------------------------------------------------------
//
//   save headers
//
//   nblocks: number of blocks
//   vblocks: pointer to array of vblocks
//   hdrs: block headers
// 
void save_headers(int nblocks, struct vblock_t *vblocks, int **hdrs) {

  int i;

  for (i = 0; i < nblocks; i++) {

    hdrs[i][NUM_VERTS] = vblocks[i].num_verts;
    hdrs[i][TOT_NUM_CELL_VERTS] = vblocks[i].tot_num_cell_verts;
    hdrs[i][NUM_COMPLETE_CELLS] = vblocks[i].num_complete_cells;
    hdrs[i][NUM_ORIG_PARTICLES] = vblocks[i].num_orig_particles;
    hdrs[i][NUM_LOC_TETS] = vblocks[i].num_loc_tets;
    hdrs[i][NUM_REM_TETS] = vblocks[i].num_rem_tets;
    hdrs[i][NUM_FACES] = vblocks[i].num_faces;
    hdrs[i][TOT_NUM_CELL_FACES] = vblocks[i].tot_num_cell_faces;

  }

}
// --------------------------------------------------------------------------
//
//   prepare for delaunay output, including saving headers
//
//   nblocks: number of blocks
//   dblocks: local blocks
//   hdrs: block headers
// 
void prep_d_out(int nblocks, struct dblock_t *dblocks, int **hdrs) {

  struct bb_t bounds; // block bounds 

  // save extents 
  for (int i = 0; i < nblocks; i++) {

    DIY_Block_bounds(0, i, &bounds);

    dblocks[i].mins[0] = bounds.min[0];
    dblocks[i].mins[1] = bounds.min[1];
    dblocks[i].mins[2] = bounds.min[2];
    dblocks[i].maxs[0] = bounds.max[0];
    dblocks[i].maxs[1] = bounds.max[1];
    dblocks[i].maxs[2] = bounds.max[2];

    hdrs[i][NUM_ORIG_PARTICLES] = dblocks[i].num_orig_particles;
    hdrs[i][NUM_PARTICLES] = dblocks[i].num_particles;
    hdrs[i][NUM_TETS] = dblocks[i].num_tets;
    hdrs[i][NUM_REM_TET_VERTS] = dblocks[i].num_rem_tet_verts;

  }

}
// --------------------------------------------------------------------------
//
//   creates and initializes blocks and headers
//
//   num_blocks: number of blocks
//   vblocks: pointer to array of vblocks
//   hdrs: pointer to array of headers, pass NULL if not used
//
//   side effects: allocates memory for blocks and headers
// 
void create_blocks(int num_blocks, struct vblock_t **vblocks, int ***hdrs) {

  int i, j;

  // allocate blocks and headers 
  *vblocks = (struct vblock_t*)malloc(sizeof(struct vblock_t) * 
				      num_blocks);
  if (hdrs)
    *hdrs = (int **)malloc(sizeof(int*) * num_blocks);

  for (i = 0; i < num_blocks; i++) {

    (*vblocks)[i].num_verts = 0;
    (*vblocks)[i].verts = NULL;
    (*vblocks)[i].save_verts = NULL;
    (*vblocks)[i].num_cell_verts = NULL;
    (*vblocks)[i].tot_num_cell_verts = 0;
    (*vblocks)[i].cells = NULL;
    (*vblocks)[i].sites = NULL;
    (*vblocks)[i].temp_num_complete_cells = 0;
    (*vblocks)[i].temp_complete_cells = NULL;
    (*vblocks)[i].num_complete_cells = 0;
    (*vblocks)[i].complete_cells = NULL;
    (*vblocks)[i].is_complete = NULL;
    (*vblocks)[i].areas = NULL;
    (*vblocks)[i].vols = NULL;
    (*vblocks)[i].face_areas = NULL;
    (*vblocks)[i].loc_tets = NULL;
    (*vblocks)[i].num_loc_tets = 0;
    (*vblocks)[i].rem_tet_gids = NULL;
    (*vblocks)[i].rem_tet_nids = NULL;
    (*vblocks)[i].rem_tet_wrap_dirs = NULL;
    (*vblocks)[i].num_rem_tets = 0;
    (*vblocks)[i].num_sent_particles = 0;
    (*vblocks)[i].alloc_sent_particles = 0;
    (*vblocks)[i].sent_particles = NULL;
    (*vblocks)[i].num_faces = 0;
    (*vblocks)[i].tot_num_cell_faces = 0;
    (*vblocks)[i].faces = NULL;
    (*vblocks)[i].cell_faces_start = NULL;
    (*vblocks)[i].cell_faces = NULL;

    if (hdrs) {
      (*hdrs)[i] = (int *)malloc(sizeof(int) * DIY_MAX_HDR_ELEMENTS);
      for (j = 0; j < DIY_MAX_HDR_ELEMENTS; j++)
	((*hdrs)[i])[j] = 0;
    }

  }

}
// ---------------------------------------------------------------------------
//
//   creates and initializes delaunay blocks and headers
//   copies original particles into the blocks
//
//   num_blocks: number of blocks
//   dblocks: local dblocks
//   hdrs: block headers, pass NULL if not used
//   particles: particles[block_num][particle] 
//   where each particle is 3 values, px, py, pz
//   num_particles; number of particles in each block
//
//   side effects: allocates memory for blocks and headers
// 
void create_dblocks(int num_blocks, struct dblock_t* &dblocks, int** &hdrs,
		    float **particles, int *num_particles) {

  // allocate
  dblocks = new dblock_t[num_blocks];
  hdrs = new int*[num_blocks];
  for (int i = 0; i < num_blocks; i++) {
    hdrs[i] = new int[DIY_MAX_HDR_ELEMENTS];
    memset(hdrs[i], 0, DIY_MAX_HDR_ELEMENTS * sizeof(int));
  }

  // initialize
  for (int i = 0; i < num_blocks; i++) {

    dblocks[i].num_orig_particles = num_particles[i];
    dblocks[i].num_particles = num_particles[i];
    // malloc instead of new so that can be realloc'ed later
    dblocks[i].particles = (float*)malloc(num_particles[i] * 3 * sizeof(float));
    for (int j = 0; j < num_particles[i]; j++) {
      dblocks[i].particles[3 * j] = particles[i][3 * j];
      dblocks[i].particles[3 * j + 1] = particles[i][3 * j + 1];
      dblocks[i].particles[3 * j + 2] = particles[i][3 * j + 2];
    }
    dblocks[i].num_tets = 0;
    dblocks[i].tets = NULL;
    dblocks[i].num_rem_tet_verts = 0;
    dblocks[i].rem_tet_verts = NULL;
    dblocks[i].vert_to_tet = NULL;

  }

}
// ---------------------------------------------------------------------------
//
//   frees blocks and headers
//
//   num_blocks: number of blocks
//   vblocks: array of vblocks
//   hdrs: pointer to array of headers, pass NULL if not used
// 
void destroy_blocks(int num_blocks, struct vblock_t *vblocks, int **hdrs) {

  int i;

  for (i = 0; i < num_blocks; i++) {
    if (hdrs && hdrs[i])
      free(hdrs[i]);
    if (vblocks[i].verts)
      free(vblocks[i].verts);
    if (vblocks[i].save_verts)
      free(vblocks[i].save_verts);
    if (vblocks[i].num_cell_verts)
      free(vblocks[i].num_cell_verts);
    if (vblocks[i].cells)
      free(vblocks[i].cells);
    if (vblocks[i].sites)
      free(vblocks[i].sites);
    if (vblocks[i].temp_complete_cells)
      free(vblocks[i].temp_complete_cells);
    if (vblocks[i].complete_cells)
      free(vblocks[i].complete_cells);
    if (vblocks[i].is_complete)
      free(vblocks[i].is_complete);
    if (vblocks[i].areas)
      free(vblocks[i].areas);
    if (vblocks[i].vols)
      free(vblocks[i].vols);
    if (vblocks[i].face_areas)
      free(vblocks[i].face_areas);
    if (vblocks[i].loc_tets)
      free(vblocks[i].loc_tets);
    if (vblocks[i].rem_tet_gids)
      free(vblocks[i].rem_tet_gids);
    if (vblocks[i].rem_tet_nids)
      free(vblocks[i].rem_tet_nids);
    if (vblocks[i].rem_tet_wrap_dirs)
      free(vblocks[i].rem_tet_wrap_dirs);
    if (vblocks[i].sent_particles)
      free(vblocks[i].sent_particles);
    if (vblocks[i].faces)
      free(vblocks[i].faces);
    if (vblocks[i].cell_faces_start)
      free(vblocks[i].cell_faces_start);
    if (vblocks[i].cell_faces)
      free(vblocks[i].cell_faces);
  }

  free(hdrs);
  free(vblocks);

}
// ---------------------------------------------------------------------------
//
//   frees delaunay blocks and headers
//
//   num_blocks: number of blocks
//   dblocks: array of dblocks
//   hdrs: pointer to array of headers, pass NULL if not used
// 
void destroy_dblocks(int num_blocks, struct dblock_t *dblocks, int **hdrs) {

  for (int i = 0; i < num_blocks; i++) {
    if (hdrs && hdrs[i])
      delete[] hdrs[i];
    if (dblocks[i].particles)
      free(dblocks[i].particles);
    if (dblocks[i].tets)
      free(dblocks[i].tets);
    if (dblocks[i].rem_tet_verts)
      free(dblocks[i].rem_tet_verts);
    if (dblocks[i].vert_to_tet)
      free(dblocks[i].vert_to_tet);
  }

  delete[] hdrs;
  delete[] dblocks;

}
// ---------------------------------------------------------------------------
//
//   resets blocks and headers between phases
//   destroys and re-creates most of the block, but keeps data related
//   sent particles
//
//   num_blocks: number of blocks
//   vblocks: array of vblocks
// 
void reset_blocks(int num_blocks, struct vblock_t *vblocks) {

  int i;

  for (i = 0; i < num_blocks; i++) {

    // free old data 
    if (vblocks[i].verts)
      free(vblocks[i].verts);
    if (vblocks[i].save_verts)
      free(vblocks[i].save_verts);
    if (vblocks[i].num_cell_verts)
      free(vblocks[i].num_cell_verts);
    if (vblocks[i].cells)
      free(vblocks[i].cells);
    if (vblocks[i].sites)
      free(vblocks[i].sites);
    if (vblocks[i].temp_complete_cells)
      free(vblocks[i].temp_complete_cells);
    if (vblocks[i].complete_cells)
      free(vblocks[i].complete_cells);
    if (vblocks[i].is_complete)
      free(vblocks[i].is_complete);
    if (vblocks[i].areas)
      free(vblocks[i].areas);
    if (vblocks[i].vols)
      free(vblocks[i].vols);
    if (vblocks[i].face_areas)
      free(vblocks[i].face_areas);
    if (vblocks[i].loc_tets)
      free(vblocks[i].loc_tets);
    if (vblocks[i].rem_tet_gids)
      free(vblocks[i].rem_tet_gids);
    if (vblocks[i].rem_tet_nids)
      free(vblocks[i].rem_tet_nids);
    if (vblocks[i].rem_tet_wrap_dirs)
      free(vblocks[i].rem_tet_wrap_dirs);
    // keep sent_particles, don't free them 
    if (vblocks[i].faces)
      free(vblocks[i].faces);
    if (vblocks[i].cell_faces_start)
      free(vblocks[i].cell_faces_start);
    if (vblocks[i].cell_faces)
      free(vblocks[i].cell_faces);

    // initialize new data 
    vblocks[i].num_verts = 0;
    vblocks[i].verts = NULL;
    vblocks[i].save_verts = NULL;
    vblocks[i].num_cell_verts = NULL;
    vblocks[i].tot_num_cell_verts = 0;
    vblocks[i].cells = NULL;
    vblocks[i].sites = NULL;
    vblocks[i].temp_num_complete_cells = 0;
    vblocks[i].temp_complete_cells = NULL;
    vblocks[i].num_complete_cells = 0;
    vblocks[i].complete_cells = NULL;
    vblocks[i].is_complete = NULL;
    vblocks[i].areas = NULL;
    vblocks[i].vols = NULL;
    vblocks[i].face_areas = NULL;
    vblocks[i].loc_tets = NULL;
    vblocks[i].num_loc_tets = 0;
    vblocks[i].rem_tet_gids = NULL;
    vblocks[i].rem_tet_nids = NULL;
    vblocks[i].rem_tet_wrap_dirs = NULL;
    vblocks[i].num_rem_tets = 0;
    // keep sent_particles, don't reset num_sent_particles,
    //   alloc_sent_particles, sent_particles 
    vblocks[i].num_faces = 0;
    vblocks[i].tot_num_cell_faces = 0;
    vblocks[i].faces = NULL;
    vblocks[i].cell_faces_start = NULL;
    vblocks[i].cell_faces = NULL;

  }

}
// ---------------------------------------------------------------------------
//
//   resets blocks between phases
//
//   num_blocks: number of blocks
//   dblocks: local dblocks
//
void reset_dblocks(int num_blocks, struct dblock_t* &dblocks) {

  // the entire data structure is listed below and commented out are the
  // fields that don't get reset - todo: clean up eventually

  // initialize
  for (int i = 0; i < num_blocks; i++) {

    // free old data

//     if (dblocks[i].particles)
//       free(dblocks[i].particles);
//     if (dblocks[i].is_complete)
//       free(dblocks[i].is_complete);
    if (dblocks[i].tets)
      free(dblocks[i].tets);
//     if (dblocks[i].rem_tet_verts)
//       free(dblocks[i].rem_tet_verts);
    if (dblocks[i].vert_to_tet)
      free(dblocks[i].vert_to_tet);

    // initialize new data

//     dblocks[i].num_orig_particles = 0;
//     dblocks[i].particles = NULL;
    dblocks[i].num_tets = 0;
    dblocks[i].tets = NULL;
//     dblocks[i].num_rem_tet_verts = 0;
//     dblocks[i].rem_tet_verts = NULL;
    dblocks[i].vert_to_tet = NULL;

  }

}
// ---------------------------------------------------------------------------
//
//   finds the direction of the nearest block to the given point
//
//   p: coordinates of the point
//   bounds: block bounds
// 
unsigned char nearest_neighbor(float* p, struct bb_t* bounds) {

  // TODO: possibly find the 3 closest neighbors, and look at the ratio of
  //   the distances to deal with the corners  

  int		i;
  float	        dists[6];
  unsigned char dirs[6] = { DIY_X0, DIY_X1, DIY_Y0, DIY_Y1, DIY_Z0, DIY_Z1 };

  for (i = 0; i < 3; ++i) {
    dists[2*i]     = p[i] - bounds->min[i];
    dists[2*i + 1] = bounds->max[i] - p[i];
  }

  int   smallest = 0;
  for (i = 1; i < 6; ++i) {
    if (dists[i] < dists[smallest])
      smallest = i;
  }

  return dirs[smallest];

}
// ---------------------------------------------------------------------------
//
//   determines cells that are incomplete or too close to neighbor such that
//   they might change after neighbor exchange. The particles corresponding
//   to sites of these cells are enqueued for exchange with neighors
//
//   tblock: one temporary voronoi block
//   vblock: one voronoi block
//   lid: local id of block
//   convex_hull_particles: pointer to convex hull particles to recheck later
//   num_convex_hull_particles: number of convex hull particles
// 
void incomplete_cells_initial(struct vblock_t *tblock, struct vblock_t *vblock,
			      int lid,
			      int** convex_hull_particles,
			      int*  num_convex_hull_particles) {

  struct bb_t bounds; // block bounds 
  int vid; // vertex id 
  int i, j, k, n;
  int chunk_size = 128; // allocation chunk size for sent particles 
  struct remote_particle_t rp; // particle being sent or received 
  struct sent_t sent; // info about sent particle saved for later 
  int complete; // no vertices in the cell are the infinite vertex 

  int allocated_convex_hull = 0;
  *num_convex_hull_particles = 0;

  DIY_Block_bounds(0, lid, &bounds);

  // get gids of all neighbors, in case a particle needs to be
  //   sent to all neighbors
  //   (enumerating all gids manually (not via DIY_Enqueue_Item_all)
  //   to be consisent with enumerating particular neighbors) 
  int num_all_neigh_gbs = DIY_Num_neighbors(0, lid);
  struct gb_t all_neigh_gbs[MAX_NEIGHBORS];
  DIY_Get_neighbors(0, lid, all_neigh_gbs);

  n = 0; // index into tblock->cells 

  // for all cells 
  for (j = 0; j < tblock->num_orig_particles; j++) {

    complete = 1; // assume complete cell unless found otherwise 
    sent.num_gbs = 0;

    // for all vertex indices in the current cell 
    
    for (k = 0; k < tblock->num_cell_verts[j]; k++) {

      vid = tblock->cells[n];

      // if a vertex is not the infinite vertex, add any neighbors
      // within the delaunay circumsphere radius of block bounds 
      if (vid) {
	float pt[3]; // target point as a float (verts are double) 
	pt[0] = tblock->verts[3 * vid];
	pt[1] = tblock->verts[3 * vid + 1];
	pt[2] = tblock->verts[3 * vid + 2];
	// radius of delaunay circumshpere is the distance from
	//   voronoi vertex to voronoi site 
	float sph_rad = distance(pt, &tblock->sites[3 * j]);
	DIY_Add_gbs_all_near(0, lid, sent.neigh_gbs, &(sent.num_gbs),
			     MAX_NEIGHBORS, pt, sph_rad);

          
      }
      else {
      	// should do this at most once per cell because at most
	//   one infinite vertex 
	complete = 0;
	add_int(j, convex_hull_particles, num_convex_hull_particles,
		&allocated_convex_hull, chunk_size);
      }
      
      n++;

    } // for all vertex indices in this cell 
    

    rp.x = tblock->sites[3 * j];
    rp.y = tblock->sites[3 * j + 1];
    rp.z = tblock->sites[3 * j + 2];
    rp.gid = DIY_Gid(0, lid);
    rp.nid = j;
    rp.dir = 0x00;
    

    // particle needs to be sent either to particular neighbors or to all 
    if (!complete || sent.num_gbs) {

      if (!complete) {

	// incomplete cell goes to the closest neighbor 
	unsigned char nearest_dir = 
	  nearest_neighbor(&(tblock->sites[3 * j]), &bounds);

	// Send the particle to the neighbor in direction dirs[smallest] 
	sent.num_gbs = 0;
	for (i = 0; i < num_all_neigh_gbs; i++) {

	  if (all_neigh_gbs[i].neigh_dir != 0x00 &&
	      all_neigh_gbs[i].neigh_dir == nearest_dir) {
	    sent.neigh_gbs[sent.num_gbs].gid = all_neigh_gbs[i].gid;
	    sent.neigh_gbs[sent.num_gbs].neigh_dir = 
	      all_neigh_gbs[i].neigh_dir;
	    sent.num_gbs++;
	  }

	}
	assert(sent.num_gbs <= 1); // sanity 

      }

      DIY_Enqueue_item_gbs(0, lid, (void *)&rp,
			   NULL, sizeof(struct remote_particle_t),
			   sent.neigh_gbs, sent.num_gbs,
			   &transform_particle);

      // save the details of the sent particle for later sending
      // completion status of sent particles to same neighbors 
      sent.particle = j;
      add_sent(sent, &(vblock->sent_particles),
	       &(vblock->num_sent_particles),
	       &(vblock->alloc_sent_particles), chunk_size);

    } // if (!complete || sent.num_gbs) 

  } // for all cells 

}
// ---------------------------------------------------------------------------

bool operator<(const gb_t& x, const gb_t& y) { return x.gid < y.gid || (x.gid == y.gid && x.neigh_dir < y.neigh_dir); }

//   determines cells that are incomplete or too close to neighbor such that
//   they might change after neighbor exchange. The particles corresponding
//   to sites of these cells are enqueued for exchange with neighors
//
//   tblock: one temporary voronoi block
//   vblock: one voronoi block
//   lid: local id of block
//   convex_hull_particles: pointer to convex hull particles to recheck later
//   num_convex_hull_particles: number of convex hull particles
// 
void incomplete_dcells_initial(struct dblock_t *dblock, int lid,
			       vector <sent_t> &sent_particles,
			       vector <int> &convex_hull_particles) {

  struct bb_t bounds; // block bounds 
  struct remote_particle_t rp; // particle being sent or received 
  struct sent_t sent; // info about one sent particle

  DIY_Block_bounds(0, lid, &bounds);

  // get gids of all neighbors, in case a particle needs to be
  //   sent to all neighbors
  //   (enumerating all gids manually (not via DIY_Enqueue_Item_all)
  //   to be consisent with enumerating particular neighbors) 
  int num_all_neigh_gbs = DIY_Num_neighbors(0, lid);
  struct gb_t all_neigh_gbs[MAX_NEIGHBORS];
  DIY_Get_neighbors(0, lid, all_neigh_gbs);

  // keep track of where we have queued each particle
  std::vector< std::set<gb_t> > destinations(dblock->num_orig_particles);

  // identify and queue convex hull particles
  for (int p = 0; p < dblock->num_orig_particles; ++p) {

    // debug
//     int rank;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     if (rank == 7)
//       fprintf(stderr, "%p %p %p p = %d num_tets = %d "
// 	      "num_orig_particles = %d\n", 
// 	      dblock, dblock->tets, dblock->vert_to_tet, p, dblock->num_tets,
// 	      dblock->num_orig_particles);

    // on convex hull = less than 4 neighbors
    if (dblock->num_tets == 0 || !complete(p, dblock->tets, dblock->vert_to_tet[p])) {

      // add to list of convex hull particles
      convex_hull_particles.push_back(p);

      // incomplete cell goes to the closest neighbor 
      unsigned char nearest_dir = 
	nearest_neighbor(&(dblock->particles[3 * p]), &bounds);

      // send the particle to the neighbor in direction nearest_dir
      sent.num_gbs = 0;
      for (int i = 0; i < num_all_neigh_gbs; i++) {
	if (all_neigh_gbs[i].neigh_dir != 0x00 &&
	    all_neigh_gbs[i].neigh_dir == nearest_dir) {
	  sent.neigh_gbs[sent.num_gbs].gid = all_neigh_gbs[i].gid;
	  sent.neigh_gbs[sent.num_gbs].neigh_dir = 
	    all_neigh_gbs[i].neigh_dir;
	  sent.num_gbs++;
	}
      }
      assert(sent.num_gbs <= 1); // sanity 

      // save the desination so we don't duplicate later
      if (sent.num_gbs > 0)
	destinations[p].insert(sent.neigh_gbs[0]);

    } // incomplete

  }

  // for all tets
  for (int t = 0; t < dblock->num_tets; t++) {
    float center[3]; // circumcenter
    circumcenter(center, &dblock->tets[t], dblock->particles);

    // radius is distance from circumcenter to any tet vertex
    int p = dblock->tets[t].verts[0];
    float rad = distance(center, &dblock->particles[3 * p]);
    DIY_Add_gbs_all_near(0, lid, sent.neigh_gbs, &(sent.num_gbs),
			 MAX_NEIGHBORS, center, rad);

    // there is at least one destination block
    if (sent.num_gbs) {

      // send all 4 verts
      for (int v = 0; v < 4; v++) {

	int p = dblock->tets[t].verts[v];

	// select neighbors we haven't sent to, yet
	for (int i = 0; i < sent.num_gbs; ++i) {
	  destinations[p].insert(sent.neigh_gbs[i]);
	}

      } // all 4 verts

    } // at least one destination block

  } // for all tets

  // queue the actual particles
  for (int p = 0; p < dblock->num_orig_particles; ++p) {
    if (!destinations[p].empty()) {
      std::vector<gb_t>	    gbs(destinations[p].begin(),
				destinations[p].end());

      rp.x = dblock->particles[3 * p];
      rp.y = dblock->particles[3 * p + 1];
      rp.z = dblock->particles[3 * p + 2];
      rp.gid = DIY_Gid(0, lid);
      rp.nid = p;
      rp.dir = 0x00;

      DIY_Enqueue_item_gbs(0, lid, (void *)&rp,
			   NULL, sizeof(struct remote_particle_t),
			   &gbs[0], gbs.size(),
			   &transform_particle);

      // save the details of the sent particle
      sent.particle = p;
      sent_particles.push_back(sent);
    }
  }
}

// ---------------------------------------------------------------------------
//
//    Go through the original convex hull particles and enqueue any remaining
//    incomplete cells to all neighbors
//
//    tblock: one temporary voronoi block
//    vblock: one voronoi block
//    lid: local id of block
//    convex_hull_particles: convex hull particles to check
//    num_convex_hull_particles: number of particles
//    CLP - Recieve List of Planar Walls
//    walls: walls
//    num_walls: number of walls
//    CLP - Return List of Mirror Points
//    mirror_particles: sites reflected across walls
//    num_mirror_particles: the number of mirror particles
// 
void incomplete_cells_final(struct vblock_t *tblock, struct vblock_t *vblock,
			    int lid,
			    int* convex_hull_particles,
			    int  num_convex_hull_particles,
			    struct wall_t *walls,
			    int num_walls,
			    float** mirror_particles,
			    int*  num_mirror_particles) {

  int i, j, k, l, n, wi;
  int vid; // vertex id 
  struct bb_t bounds; // local block bounds 
  struct remote_particle_t rp; // particle being sent or received 
  struct sent_t sent; // info about sent particle saved for later 
  int chunk_size = 128; // allocation chunk size for sent particles 
  
  DIY_Block_bounds(0, lid, &bounds);

  // get gids of all neighbors, in case a particle needs to be
  //   sent to all neighbors
  //   (enumerating all gids manually (not via DIY_Enqueue_Item_all)
  //   to be consisent with enumerating particular neighbors) 
  int num_all_neigh_gbs = DIY_Num_neighbors(0, lid);
  struct gb_t all_neigh_gbs[MAX_NEIGHBORS];
  DIY_Get_neighbors(0, lid, all_neigh_gbs);

  // We must go through all the particles to maintain n
  //   (the index into the vertices array) 

  // CLP 
  int* wall_cut = (int *)malloc(num_walls * sizeof(int));
  int allocated_mirror_particles = 0;
  *num_mirror_particles = 0;
  
  i = 0; n = 0;
  for (j = 0; j < tblock->num_orig_particles; j++) {

    sent.num_gbs = 0;

    // CLP - zero generate-wall-point array (length of number of walls) 
    for (wi = 0; wi < num_walls; wi++)
      wall_cut[wi] = 0;

    // CLP - adding a check so that the array convex_hull_particles 
    //   is not accessed where it is uninitialized 
    if (i < num_convex_hull_particles && j == convex_hull_particles[i]) {
      
      // direction of closest neighbor 
      unsigned char nearest_dir = 
	nearest_neighbor(&(tblock->sites[3*j]), &bounds);

      for (k = 0; k < tblock->num_cell_verts[j]; ++k) {

	vid = tblock->cells[n + k];

	if (!vid) {

	  // local point still on the convex hull goes to everybody
	  // it hasn't gone to yet 
	  sent.num_gbs = 0;
	  for (l = 0; l < num_all_neigh_gbs; l++) {
	    if (all_neigh_gbs[l].neigh_dir != 0x00 &&
		all_neigh_gbs[l].neigh_dir != nearest_dir) {
	      sent.neigh_gbs[sent.num_gbs].gid =
		all_neigh_gbs[l].gid;
	      sent.neigh_gbs[sent.num_gbs].neigh_dir =
		all_neigh_gbs[l].neigh_dir;
	      sent.num_gbs++;
	    }
	  }
 
	  // CLP - set the mirror-generate array to all ones  
	  // (extra calculations but simpler to assume!) 
	  for (wi = 0; wi < num_walls; wi++)
            wall_cut[wi] = 1;
      
	} // !vid 

	else { // vid 

	  float pt[3]; // target point as a float (verts are double) 
	  pt[0] = tblock->verts[3 * vid];
	  pt[1] = tblock->verts[3 * vid + 1];
	  pt[2] = tblock->verts[3 * vid + 2];

	  // radius of delaunay circumshpere is the distance from
	  //   voronoi vertex to voronoi site 
	  float sph_rad = distance(pt, &tblock->sites[3 * j]);
	  DIY_Add_gbs_all_near(0, lid, sent.neigh_gbs, &(sent.num_gbs),
			       MAX_NEIGHBORS, pt, sph_rad);

	  // remove the neighbor in the nearest_dir;
	  // we've already sent to it 
	  for (l = 0; l < sent.num_gbs; ++l) {
	    if (sent.neigh_gbs[l].neigh_dir == nearest_dir) {
	      if (l < sent.num_gbs - 1)
		sent.neigh_gbs[l] = sent.neigh_gbs[sent.num_gbs - 1];
	      sent.num_gbs--;
	    }
	  }
     
	  // CLP - for each wall 
	  for (wi = 0; wi < num_walls; wi++) {

	    // CLP - If the mirror-generate[wall-index] is not 1 
	    if (!wall_cut[wi])
	      // CLP - generate the sign of pt relative to the wall.  
	      // If it is not positive (wall vectors must point inward!) 
	      // then set the mirror-generate[wall-index] to one  
	      wall_cut[wi] += test_outside(pt,&walls[wi]);

	  }

	} // vid 

      } // for k = 0; k < tblock->num_cell_verts[j] 

      if (sent.num_gbs) {

	rp.x = tblock->sites[3 * j];
	rp.y = tblock->sites[3 * j + 1];
	rp.z = tblock->sites[3 * j + 2];
	rp.gid = DIY_Gid(0, lid);
	rp.nid = j;
	rp.dir = 0x00;

	DIY_Enqueue_item_gbs(0, lid, (void *)&rp,
			     NULL, sizeof(struct remote_particle_t),
			     sent.neigh_gbs, sent.num_gbs,
			     &transform_particle);

	sent.particle = j;
	add_sent(sent, &(vblock->sent_particles),
		 &(vblock->num_sent_particles),
		 &(vblock->alloc_sent_particles), chunk_size);

      }

      ++i;

      // CLP disabling this break when there are walls because 
      // I need to iterate through all the particles. 
      if (!num_walls && i >= num_convex_hull_particles)
        break;

    } // if (j == convex_hull_particles[i]) 

    // CLP - else { add my test, (loop through vids, test looks like above }
    else {

      for (k = 0; k < tblock->num_cell_verts[j]; ++k) {

	vid = tblock->cells[n + k];
	float pt[3]; // target point as a float (verts are double) 
	pt[0] = tblock->verts[3 * vid];
	pt[1] = tblock->verts[3 * vid + 1];
	pt[2] = tblock->verts[3 * vid + 2];

	// CLP - for each wall 
	for (wi = 0; wi < num_walls; wi++) {
	  // CLP - If the mirror-generate[wall-index] is not 1 
	  if (!wall_cut[wi])
	    // CLP - generate the sign of pt relative to the wall.  
	    // If it is not positive (wall vectors must point inward!) 
	    // then set the mirror-generate[wall-index] to one  
	    wall_cut[wi] += test_outside(pt,&walls[wi]);
	}

      } // for k 
    
    } // else 
    
    // CLP - for each mirror-generate index that is 1, 
    //   generate the mirror point given site rp and the wall
    //   Here I am building a list of points.  Where do they go? 
    //   Return as pointer 
    for (wi =0; wi < num_walls; wi++) {

      if (wall_cut[wi]) {
	float rpt[3];
	float spt[3];
	spt[0] = tblock->sites[3 * j];
	spt[1] = tblock->sites[3 * j + 1];
	spt[2] = tblock->sites[3 * j + 2];
	generate_mirror(rpt,spt,&walls[wi]);
	add_pt(rpt, mirror_particles, num_mirror_particles,
	       &allocated_mirror_particles, chunk_size);
      }

    } // for 
      
    n += tblock->num_cell_verts[j];
    
  }
  
  // CLP cleanup 
  free(wall_cut);

}

// ---------------------------------------------------------------------------
//
//    Go through the original convex hull particles and enqueue any remaining
//    incomplete cells to all neighbors
//
//    dblock: one delaunay voronoi block
//    lid: local id of block
//    sent_particles: convex hull particles to check
//    convex_hull_particles: convex hull particles to check
// 
void incomplete_dcells_final(struct dblock_t *dblock, int lid,
			    vector <sent_t> &sent_particles,
			    vector <int> &convex_hull_particles) {

  struct bb_t bounds; // block bounds 
  struct remote_particle_t rp; // particle being sent or received 
  struct sent_t sent; // info about one sent particle

  DIY_Block_bounds(0, lid, &bounds);

  // get gids of all neighbors, in case a particle needs to be
  //   sent to all neighbors
  //   (enumerating all gids manually (not via DIY_Enqueue_Item_all)
  //   to be consisent with enumerating particular neighbors) 
  int num_all_neigh_gbs = DIY_Num_neighbors(0, lid);
  struct gb_t all_neigh_gbs[MAX_NEIGHBORS];
  DIY_Get_neighbors(0, lid, all_neigh_gbs);

  // identify and queue convex hull particles
  int old_sent = 0;
  int last_sent = sent_particles.size();
  for (int j = 0; j < (int)convex_hull_particles.size(); ++j) {

    int p = convex_hull_particles[j];

    while (old_sent < last_sent && sent_particles[old_sent].particle != p)
      ++old_sent;

    std::vector<int> nbrs;
    bool complete = neighbor_tets(nbrs, p, dblock->tets, 
				  dblock->vert_to_tet[p]);

    if (!complete) {

      // local point still on the convex hull goes to everybody
      // it hasn't gone to yet 

      sent.num_gbs = 0;
      for (int l = 0; l < num_all_neigh_gbs; l++) {
	if (all_neigh_gbs[l].neigh_dir != 0x00) {
	  // sent_particles[p].neigh_gbs is sorted (was inserted from a set),
	  // so we can use a binary search
	  bool exists;
	  if (sent_particles.size())
	    exists = 
	      std::binary_search(sent_particles[old_sent].neigh_gbs,
				 sent_particles[old_sent].neigh_gbs + 
				 sent_particles[old_sent].num_gbs,
				 all_neigh_gbs[l]);
	  else
	    exists = false;
	  if (!exists) {
	    sent.neigh_gbs[sent.num_gbs].gid	    = all_neigh_gbs[l].gid;
	    sent.neigh_gbs[sent.num_gbs].neigh_dir  = all_neigh_gbs[l].neigh_dir;
	    sent.num_gbs++;
	  }
	}
      }
    
    } // !complete

    else { // complete

      std::set<gb_t> destinations;
      for (int j = 0; j < (int)nbrs.size(); ++j) {
	int t = nbrs[j];
	float center[3];
	circumcenter(center, &dblock->tets[t], dblock->particles);

	// radius is distance from circumcenter to any tet vertex
	int p0 = dblock->tets[t].verts[0];
	float rad = distance(center, &dblock->particles[3 * p0]);
	gb_t candidates[MAX_NEIGHBORS];
	int num_candidates = 0;
	DIY_Add_gbs_all_near(0, lid, candidates, &num_candidates,
			     MAX_NEIGHBORS, center, rad);

	// remove the neighbors we've already sent to
	for (int l = 0; l < num_candidates; ++l) {
	  bool exists = std::binary_search(sent_particles[old_sent].neigh_gbs,
					   sent_particles[old_sent].neigh_gbs + 
					   sent_particles[old_sent].num_gbs,
					   candidates[l]);
	  if (!exists)
	    destinations.insert(candidates[l]);
	}
      } // for t in nbrs

      sent.num_gbs = 0;
      for (std::set<gb_t>::const_iterator cur = destinations.begin();
					  cur != destinations.end(); ++cur) {
	sent.neigh_gbs[sent.num_gbs] = *cur;
	++sent.num_gbs;
      }
    } // complete 

    if (sent.num_gbs) {
      rp.x = dblock->particles[3 * p];
      rp.y = dblock->particles[3 * p + 1];
      rp.z = dblock->particles[3 * p + 2];
      rp.gid = DIY_Gid(0, lid);
      rp.nid = p;
      rp.dir = 0x00;
      DIY_Enqueue_item_gbs(0, lid, (void *)&rp,
			   NULL, sizeof(struct remote_particle_t),
			   sent.neigh_gbs, sent.num_gbs,
			   &transform_particle);
      sent.particle = p;
      sent_particles.push_back(sent);
    }
  } // for p
}


// --------------------------------------------------------------------------
//
//   determines connectivity of faces in complete cells
//
//   vblock: one voronoi block
//
//   side effects: allocates memory for cell_faces and cell_faces_start 
//   in voronoi block
// 
void cell_faces(struct vblock_t *vblock) {

  int cell; // current cell 
  int i;

  // todo: allocate to size of complete cells and don't store faces for
  //   incomplete cells 

  // temporary count of number of faces in each of my original cells 
  int *counts = (int *)malloc(vblock->num_orig_particles *
			      sizeof(int));
  // starting offset of faces in each of my original cells 
  vblock->cell_faces_start = (int *)malloc(vblock->num_orig_particles *
					   sizeof(int));
  memset(counts, 0, vblock->num_orig_particles * sizeof(int));

  // pass 1: traverse faces array and get number of faces in each cell
  //   use face starting offsets array temporarily to hold face counts,
  //   will convert to starting offsets (prefix sum of counts) later 
  vblock->tot_num_cell_faces = 0;
  for (i = 0; i < vblock->num_faces; i++) {
    cell = vblock->faces[i].cells[0];
    // each block retains only those cells and their faces whose particles 
    //   it originally had 
    if (cell < vblock->num_orig_particles) {
      counts[cell]++;
      vblock->tot_num_cell_faces++;
    }
    cell = vblock->faces[i].cells[1];
    if (cell < vblock->num_orig_particles) {
      counts[cell]++;
      vblock->tot_num_cell_faces++;
    }
  }

  // convert face counts to starting offsets and offset of end (used to
  //    compute face count of last cell 
  vblock->cell_faces_start[0] = 0;
  for (i = 1; i < vblock->num_orig_particles; i++)
    vblock->cell_faces_start[i] = vblock->cell_faces_start[i - 1] + 
      counts[i - 1];

  // allocate cell_faces 
  vblock->cell_faces = (int *)malloc(vblock->tot_num_cell_faces * sizeof(int));

  // pass 2: traverse faces array and save face ids for each cell 
  memset(counts, 0, vblock->num_orig_particles * sizeof(int));
  for (i = 0; i < vblock->num_faces; i++) {
    cell = vblock->faces[i].cells[0];
    // again, each block retains only those cells and their faces 
    //    whose particles it originally had 
    if (cell < vblock->num_orig_particles) {
      int id = vblock->cell_faces_start[cell] + counts[cell];
      vblock->cell_faces[id] = i;
      counts[cell]++;
    }
    cell = vblock->faces[i].cells[1];
    if (cell < vblock->num_orig_particles) {
      int id = vblock->cell_faces_start[cell] + counts[cell];
      vblock->cell_faces[id] = i;
      counts[cell]++;
    }
  }

  // cleanup 
  free(counts);

}
// --------------------------------------------------------------------------
//
//   determines complete cells: cells that don't contain an infinite
//    vertex or any other vertices outside of the block bounds
//
//   vblock: one voronoi block
//   lid: local id of block
//
//   side effects: allocates memory for complete cells in voronoi block
// 
void complete_cells(struct vblock_t *vblock, int lid) {

  struct bb_t bounds; // block bounds 
  int vid, vid1; // vertex id 
  int j, k, n, m;
  double d2_min = 0.0; // dia^2 of circumscribing sphere of volume min_vol 
  int start_n; // index into cells at start of new cell 
  int too_small; // whether cell volume is definitely below threshold 

  // allocate memory based on number of cells and average number of
  //    faces and vertices in a cell
  //   this is wasteful if filtering on volume because we will probably
  //   only need a fraction of this memory
  //   todo: fix this with my own memory manager 
  vblock->temp_complete_cells =
	  (int *)malloc(vblock->num_orig_particles * sizeof(int));
  vblock->complete_cells =
	  (int *)malloc(vblock->num_orig_particles * sizeof(int));

  DIY_Block_bounds(0, lid, &bounds);

  // minimum cell diameter squared 
  if (min_vol > 0.0)
    // d^2 = 4 * (3/4 * min_vol / pi)^ (2/3) 
    d2_min = 1.539339 * pow(min_vol, 0.66667);

  // find complete cells 

  vblock->temp_num_complete_cells = 0;
  n = 0; // index into vblock->cells 

  // for all cells up to original number of input particles (each block 
  //    retains only those cells whose particles it originally had) 
    
  for (j = 0; j < vblock->num_orig_particles; j++) {

    // init 
    if (!vblock->num_cell_verts[j])
      continue;

    int complete = 1;
    too_small = (min_vol > 0.0 ? 1 : 0);
    
    // for all vertex indices in the current cell 
    for (k = 0; k < vblock->num_cell_verts[j]; k++) {

      if (k == 0)
        start_n = n;

      vid = vblock->cells[n];

      float eps = 1.0e-6;
      if ( // vertex can fail for the following reasons 

	  // qhull's "infinite vertex" 
	  (fabs(vblock->verts[3 * vid] - vblock->verts[0]) < eps &&
	   fabs(vblock->verts[3 * vid + 1] - vblock->verts[1]) < eps &&
	   fabs(vblock->verts[3 * vid + 2] - vblock->verts[2]) < eps) ||

	  // out of overall data bounds when wrapping is off 
        (!wrap_neighbors &&
	   (vblock->verts[3 * vid]     < (data_mins[0] - eps) ||
	    vblock->verts[3 * vid]     > (data_maxs[0] + eps) ||
	    vblock->verts[3 * vid + 1] < (data_mins[1] - eps) ||
	    vblock->verts[3 * vid + 1] > (data_maxs[1] + eps) ||
	    vblock->verts[3 * vid + 2] < (data_mins[2] - eps) ||
	    vblock->verts[3 * vid + 2] > (data_maxs[2] + eps) ) ) ) {

        complete = 0;
        
        //CLP -debug 
        
        /*
        if (fabs(vblock->verts[3 * vid] - vblock->verts[0]) < eps &&
	   fabs(vblock->verts[3 * vid + 1] - vblock->verts[1]) < eps &&
	   fabs(vblock->verts[3 * vid + 2] - vblock->verts[2]) < eps)
               fprintf(stderr, "cell %d failed because failed because infinite\n",j);
           
        if   (!wrap_neighbors &&
	   (vblock->verts[3 * vid]     < (data_mins[0] - eps) ||
	    vblock->verts[3 * vid]     > (data_maxs[0] + eps) ||
	    vblock->verts[3 * vid + 1] < (data_mins[1] - eps) ||
	    vblock->verts[3 * vid + 1] > (data_maxs[1] + eps) ||
	    vblock->verts[3 * vid + 2] < (data_mins[2] - eps) ||
	    vblock->verts[3 * vid + 2] > (data_maxs[2] + eps) ) )
            {
            fprintf(stderr, "cell %d failed because vertex out of bounds, (%f, %f, %f) \n",j, vblock->verts[3 * vid],vblock->verts[3 * vid + 1], vblock->verts[3 * vid + 2] );
            }
        */

	n += (vblock->num_cell_verts[j] - k); // skip rest of this cell 
	break;
    
      } // if 

      // check minimum volume if enabled and it has not been excceded yet 
      if (too_small) {

	// for all vertices in this cell 
	for (m = start_n; m < start_n + vblock->num_cell_verts[j]; m++) {
	  vid1 = vblock->cells[m];
	  double d2 =
	    (vblock->verts[3 * vid] - vblock->verts[3 * vid1]) *
	    (vblock->verts[3 * vid] - vblock->verts[3 * vid1]) +
	    (vblock->verts[3 * vid + 1] - vblock->verts[3 * vid1 + 1]) *
	    (vblock->verts[3 * vid + 1] - vblock->verts[3 * vid1 + 1]) +
	    (vblock->verts[3 * vid + 2] - vblock->verts[3 * vid1 + 2]) *
	    (vblock->verts[3 * vid + 2] - vblock->verts[3 * vid1 + 2]);
	  if (d2 > d2_min) {
	    too_small = 0;
	    break;
	  }

	} // all vertices in this cell 

      } // small volume threshold 

      // check if volume is too small at the end of the cell 
      if (k == vblock->num_cell_verts[j] - 1 && too_small)
        complete = 0;
      n++;

    } // for all vertex indices in this cell 

    // one last check that site is within cell bounds. if so, save the cell 
    if (complete && cell_bounds(vblock, j, start_n)) {
      (vblock->temp_complete_cells)[vblock->temp_num_complete_cells++] = j;
      vblock->is_complete[j] = 1;
    }
    else
      vblock->is_complete[j] = 0;

  } // for all cells 

}
// --------------------------------------------------------------------------
//
//   generates test particles for a  block
//
//   lid: local id of block
//   particles: pointer to particle vector in this order: 
//   particle0x, particle0y, particle0z, particle1x, particle1y, particle1z, ...
//   jitter: maximum amount to randomly move particles
//
//   returns: number of particles in this block
//
//   side effects: allocates memory for particles, caller's responsibility 
//     to free
// 
int gen_particles(int lid, float **particles, float jitter) {

  int sizes[3]; // number of grid points 
  int i, j, k;
  int n = 0;
  int num_particles; // throreticl num particles with duplicates at 
		     // block boundaries 
  int act_num_particles; // actual number of particles unique across blocks 
  float jit; // random jitter amount, 0 - MAX_JITTER 

  // allocate particles 
  struct bb_t bounds;
  DIY_Block_bounds(0, lid, &bounds);
  sizes[0] = (int)(bounds.max[0] - bounds.min[0] + 1);
  sizes[1] = (int)(bounds.max[1] - bounds.min[1] + 1);
  sizes[2] = (int)(bounds.max[2] - bounds.min[2] + 1);

  num_particles = sizes[0] * sizes[1] * sizes[2];

  *particles = (float *)malloc(num_particles * 3 * sizeof(float));

  // assign particles 

  n = 0;
  for (i = 0; i < sizes[0]; i++) {
    if (bounds.min[0] > 0 && i == 0) // dedup block doundary points 
      continue;
    for (j = 0; j < sizes[1]; j++) {
      if (bounds.min[1] > 0 && j == 0) // dedup block doundary points 
	continue;
      for (k = 0; k < sizes[2]; k++) {
	if (bounds.min[2] > 0 && k == 0) // dedup block doundary points 
	  continue;

	// start with particles on a grid 
	(*particles)[3 * n] = bounds.min[0] + i;
	(*particles)[3 * n + 1] = bounds.min[1] + j;
	(*particles)[3 * n + 2] = bounds.min[2] + k;

	// and now jitter them 
	jit = rand() / (float)RAND_MAX * 2 * jitter - jitter;
	if ((*particles)[3 * n] - jit >= bounds.min[0] &&
	    (*particles)[3 * n] - jit <= bounds.max[0])
	  (*particles)[3 * n] -= jit;
	else if ((*particles)[3 * n] + jit >= bounds.min[0] &&
		 (*particles)[3 * n] + jit <= bounds.max[0])
	  (*particles)[3 * n] += jit;

	jit = rand() / (float)RAND_MAX * 2 * jitter - jitter;
	if ((*particles)[3 * n + 1] - jit >= bounds.min[1] &&
	    (*particles)[3 * n + 1] - jit <= bounds.max[1])
	  (*particles)[3 * n + 1] -= jit;
	else if ((*particles)[3 * n + 1] + jit >= bounds.min[1] &&
		 (*particles)[3 * n + 1] + jit <= bounds.max[1])
	  (*particles)[3 * n + 1] += jit;

	jit = rand() / (float)RAND_MAX * 2 * jitter - jitter;
	if ((*particles)[3 * n + 2] - jit >= bounds.min[2] &&
	    (*particles)[3 * n + 2] - jit <= bounds.max[2])
	  (*particles)[3 * n + 2] -= jit;
	else if ((*particles)[3 * n + 2] + jit >= bounds.min[2] &&
		 (*particles)[3 * n + 2] + jit <= bounds.max[2])
	  (*particles)[3 * n + 2] += jit;
  
	n++;

      }

    }

  }

  act_num_particles = n;

  return act_num_particles;

}
// --------------------------------------------------------------------------
//
//   prints a block
//
//   vblock: current voronoi block
//   gid: global block id
// 
void print_block(struct vblock_t *vblock, int gid) {

  int i;

  fprintf(stderr, "block gid = %d, %d complete cells: ", 
	  gid, vblock->num_complete_cells);
  for (i = 0; i < vblock->num_complete_cells; i++)
    fprintf(stderr, "%d ", vblock->complete_cells[i]);
  fprintf(stderr, "\n");

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
//   adds an int to a c-style vector of ints
//
//   val: value to be added
//   vals: pointer to dynamic array of values
//   numvals: pointer to number of values currently stored, updated by add_int
//   maxvals: pointer to number of values currently allocated
//   chunk_size: number of values to allocate at a time
//
// 
void add_int(int val, int **vals, int *numvals, int *maxvals, int chunk_size) {

  // first time 
  if (*maxvals == 0) {
    *vals = (int *)malloc(chunk_size * sizeof(int));
    *numvals = 0;
    *maxvals = chunk_size;
  }

  // grow memory 
  else if (*numvals >= *maxvals) {
    *vals = (int *)realloc(*vals, 
			       (chunk_size + *maxvals) * sizeof(int));
    *maxvals += chunk_size;
  }

  // add the element 
  (*vals)[*numvals] = val;
  (*numvals)++;

}
// --------------------------------------------------------------------------
//
//   adds a float to a c-style vector of floats
//
//   val: value to be added
//   vals: pointer to dynamic array of values
//   numvals: pointer to number of values currently stored, updated by add_int
//   maxvals: pointer to number of values currently allocated
//   chunk_size: number of values to allocate at a time
//
// 
void add_float(float val, float **vals, int *numvals, int *maxvals, 
	       int chunk_size) {

  // first time 
  if (*maxvals == 0) {
    *vals = (float *)malloc(chunk_size * sizeof(float));
    *numvals = 0;
    *maxvals = chunk_size;
  }

  // grow memory 
  else if (*numvals >= *maxvals) {
    *vals = (float *)realloc(*vals, 
			       (chunk_size + *maxvals) * sizeof(float));
    *maxvals += chunk_size;
  }

  // add the element 
  (*vals)[*numvals] = val;
  (*numvals)++;

}
// --------------------------------------------------------------------------
//
//   adds a point (array of 3 floats) to a c-style vector of x,y,z,x,y,z points
//
//   val: value to be added
//   vals: pointer to dynamic array of values
//   numvals: pointer to number of points currently stored, updated by add_int
//   maxvals: pointer to number of points currently allocated
//   chunk_size: number of points to allocate at a time
//
// 
void add_pt(float *val, float **vals, int *numvals, int *maxvals, 
	    int chunk_size) {

  // first time 
  if (*maxvals == 0) {
    *vals = (float *)malloc(chunk_size * 3 * sizeof(float));
    *numvals = 0;
    *maxvals = chunk_size;
  }

  // grow memory 
  else if (*numvals >= *maxvals) {
    *vals = (float *)realloc(*vals, 
			       (chunk_size + *maxvals) * 3 * sizeof(float));
    *maxvals += chunk_size;
  }

  // add the element 
  (*vals)[3 * *numvals] = val[0];
  (*vals)[3 * *numvals + 1] = val[1];
  (*vals)[3 * *numvals + 2] = val[2];
  (*numvals)++;

}
// --------------------------------------------------------------------------
//
//   adds a sent particle to a c-style vector of sent particles
//
//   val: sent particle to be added
//   vals: pointer to dynamic array of sent particles
//   numvals: pointer to number of values currently stored, updated by add_int
//   maxvals: pointer to number of values currently allocated
//   chunk_size: number of values to allocate at a time
//
// 
void add_sent(struct sent_t val, struct sent_t **vals, int *numvals, 
	      int *maxvals, int chunk_size) {

  int i;

  // first time 
  if (*maxvals == 0) {
    *vals = (struct sent_t *)malloc(chunk_size * sizeof(struct sent_t));
    *numvals = 0;
    *maxvals = chunk_size;
  }

  // grow memory 
  else if (*numvals >= *maxvals) {
    *vals = 
      (struct sent_t *)realloc(*vals, 
			       (chunk_size + *maxvals) * 
			       sizeof(struct sent_t));
    *maxvals += chunk_size;
  }

  // add the element 
  (*vals)[*numvals].particle = val.particle;
  (*vals)[*numvals].num_gbs = val.num_gbs;
  for (i = 0; i < MAX_NEIGHBORS; i++) {
    (*vals)[*numvals].neigh_gbs[i].gid = val.neigh_gbs[i].gid;
    (*vals)[*numvals].neigh_gbs[i].neigh_dir = val.neigh_gbs[i].neigh_dir;
  }
  (*numvals)++;

}
// --------------------------------------------------------------------------
//
//   checks if an array of ints has been allocated large enough to access
//   a given index. If not, grows the array and initializes the empty ints
//
//   vals: pointer to dynamic array of ints
//   index: desired index to be accessed
//   numitems: pointer to number of items currently stored, ie, 
//     last subscript accessed + 1, updated by this function
//   maxitems: pointer to number of items currently allocated
//   chunk_size: minimum number of items to allocate at a time
//   init_val: initalization value for newly allocated items
//
// 
void add_empty_int(int **vals, int index, int *numitems, int *maxitems, 
		   int chunk_size, int init_val) {

  int i;
  int alloc_chunk; // max of chunk_size and chunk needed to get to the index 

  // first time 
  if (*maxitems == 0) {

    // allocate 
    alloc_chunk = (index < chunk_size ? chunk_size : index + 1);
    *vals = (int *)malloc(alloc_chunk * sizeof(int));
    // init empty vals 
    for (i = 0; i < alloc_chunk; i++)
      (*vals)[i] = init_val;

    *numitems = 0;
    *maxitems = alloc_chunk;

  }

  // grow memory 
  else if (index >= *maxitems) {

    // realloc 
    alloc_chunk = 
      (index < *maxitems + chunk_size ? chunk_size : index + 1 - *maxitems);
    *vals = (int *)realloc(*vals, (alloc_chunk + *maxitems) * sizeof(int));

    // init empty buckets 
    for (i = *maxitems; i < *maxitems + alloc_chunk; i++)
      (*vals)[i] = init_val;

    *maxitems += alloc_chunk;

  }

  (*numitems)++;

}
// --------------------------------------------------------------------------
//
//    checks whether the site of the cell is inside the bounds of
//    temporary cell (uses the original cells, not the complete cells, and
//    the double version of verts, not save_verts)
//
//    vblock: one voronoi block
//    cell: current cell counter
//    vert: current vertex counter
//
//    returns: 1 = site is inside cell bounds, 0 = site is outside cell bounds
//  
 int cell_bounds(struct vblock_t *vblock, int cell, int vert) {

  float cell_min[3], cell_max[3];
  int k;

  // get cell bounds 
  for (k = 0; k < vblock->num_cell_verts[cell]; k++) { // vertices 

    int v = vblock->cells[vert];
	  
    if ((k == 0) || vblock->verts[3 * v] < cell_min[0])
      cell_min[0] = vblock->verts[3 * v];
    if ((k == 0) || vblock->verts[3 * v] > cell_max[0])
      cell_max[0] = vblock->verts[3 * v];

    if ((k == 0) || vblock->verts[3 * v + 1] < cell_min[1])
      cell_min[1] = vblock->verts[3 * v + 1];
    if ((k == 0) || vblock->verts[3 * v + 1] > cell_max[1])
      cell_max[1] = vblock->verts[3 * v + 1];

    if ((k == 0) || vblock->verts[3 * v + 2] < cell_min[2])
      cell_min[2] = vblock->verts[3 * v + 2];
    if ((k == 0) || vblock->verts[3 * v + 2] > cell_max[2])
      cell_max[2] = vblock->verts[3 * v + 2];

    vert++;

  } // vertices 

  // check that site of cell is in the interior of the bounds (sanity) 
  if (vblock->sites[3 * cell] < cell_min[0] ||
      vblock->sites[3 * cell] > cell_max[0] ||
      vblock->sites[3 * cell + 1] < cell_min[1] ||
      vblock->sites[3 * cell + 1] > cell_max[1] ||
      vblock->sites[3 * cell + 2] < cell_min[2] ||
      vblock->sites[3 * cell + 2] > cell_max[2]) {
    fprintf(stderr, "warning: the site for cell %d "
	    "[%.3f %.3f %.3f] is not "
	    "inside the cell bounds min [%.3f %.3f %.3f] "
	    "max [%.3f %.3f %.3f]; skipping this cell\n",
	    cell, vblock->sites[3 * cell], vblock->sites[3 * cell + 1], 
	    vblock->sites[3 * cell + 2],
	    cell_min[0], cell_min[1], cell_min[2],
	    cell_max[0], cell_max[1], cell_max[2]);

    return 0;
  }

  else
    return 1;

} 
// --------------------------------------------------------------------------
//
//    for all tets, determines local, and records the necessary 
//    information in vblock
//  
//    tet_verts: vertices of the tetrahedra in the block
//    num_tets: number of tets in this block
//    vblock: local block
//    gids: global block ids of owners of received particles in each of 
//      my blocks
//    nids: native particle ids of received particles in each of my blocks
//    dirs: wrapping directions of received particles in each of my blocks
//    rics: is complete status of received particles
//    lid: local id of this block
//    num_recv: number of received particles
// 
void gen_tets(int *tet_verts, int num_tets, struct vblock_t *vblock,
	      int *gids, int *nids, unsigned char *dirs,
	      struct remote_ic_t *rics, int lid, int num_recvd) {

  int v; // vertex in current tet (0, 1, 2, 3) 
  int t; // current tet 
  int n = 0; // number of vertices in strictly local final tets 
  int m = 0; // number of vertices in non strictly local final tets 
  int i;

  // todo: static allocation wasteful; we don't know how many tets
  // are local and how many are remote; use add_int to groew arrays instead 
  vblock->loc_tets = (int *)malloc(num_tets * 4 * sizeof(int));
  vblock->rem_tet_gids = (int *)malloc(num_tets * 4 * sizeof(int));
  vblock->rem_tet_nids = (int *)malloc(num_tets * 4 * sizeof(int));
  vblock->rem_tet_wrap_dirs = (unsigned char *)malloc(num_tets * 4);

  // for all tets 
  for (t = 0; t < num_tets; t++) {

    // test whether tet is strictly local (all vertices are local) or not 
    for (v = 0; v < 4; v++) {
      if (tet_verts[t * 4 + v] >= vblock->num_orig_particles)
        break;
    }

    if (v == 4) { // local, store it 
      // filter out tets that touch local incomplete voronoi cells 
      int v1;
      for (v1 = 0; v1 < 4; v1++) {
	if (!vblock->is_complete[tet_verts[t * 4 + v1]])
	  break;
      }
      if (v1 == 4) {
	int v2;
	for (v2 = 0; v2 < 4; v2++)
	  vblock->loc_tets[n++] = tet_verts[t * 4 + v2];
      }
    }

    // not strictly local, at least one vertex is remote, and at least one
    //    vertex is local 
    else if (tet_verts[t * 4 + 0] < vblock->num_orig_particles ||
	     tet_verts[t * 4 + 1] < vblock->num_orig_particles ||
	     tet_verts[t * 4 + 2] < vblock->num_orig_particles ||
	     tet_verts[t * 4 + 3] < vblock->num_orig_particles) {

      // decide whether I should own this tet, owner will be minimum
      // block gid of all contributors to this tet 
      int sort_gids[4]; // gids of 4 vertices 
      for (v = 0; v < 4; v++) {
	if (tet_verts[t * 4 + v] < vblock->num_orig_particles)
	  sort_gids[v] = DIY_Gid(0, lid);
	else
	  sort_gids[v] = gids[tet_verts[t * 4 + v] -
			      vblock->num_orig_particles];
      }
      qsort(sort_gids, 4, sizeof(int), &compare);

      // I will own the tet 
      if (sort_gids[0] == DIY_Gid(0, lid)) {

	// filter out tets that touch local incomplete voronoi cells 
	for (v = 0; v < 4; v++) {

	  // if this vertex is local, check its completion status 
	  if (tet_verts[t * 4 + v] < vblock->num_orig_particles &&
	      !vblock->is_complete[tet_verts[t * 4 + v]])
	    break;

	  // if this vertex is remote, check its completion status 
	  if (tet_verts[t * 4 + v] >= vblock->num_orig_particles) {

	    // find the correct entry in the completion status
	    //    todo: linear search for now, accelerate later 
	    for (i = 0; i < num_recvd; i++) {
	      if (rics[i].gid ==
		  gids[tet_verts[t * 4 + v] - vblock->num_orig_particles] &&
		  rics[i].nid ==
		  nids[tet_verts[t * 4 + v] - vblock->num_orig_particles])
		break;
	    }
	    assert(i < num_recvd); // sanity 
	    if (!rics[i].is_complete)
	      break;

	  } // if vertex is remote 

	} // for four vertices 

	if (v == 4) { // complete 

	  int v1;
	  // save four remote verts 
	  for (v1 = 0; v1 < 4; v1++) {

	    // this vertex is local 
	    if (tet_verts[t * 4 + v1] < vblock->num_orig_particles) {
	      vblock->rem_tet_gids[m] = DIY_Gid(0, lid);
	      vblock->rem_tet_nids[m] = tet_verts[t * 4 + v1];
	      vblock->rem_tet_wrap_dirs[m] = 0x00;
	    }
	    // this vertex is remote 
	    else {
	      // need to subtract number of original (local) particles
	      // from vertex to index into gids and nids;
	      // they are only for remote particles but tet verts
	      // are for all particles, local + remote 
	      vblock->rem_tet_gids[m] =
		gids[tet_verts[t * 4 + v1] - vblock->num_orig_particles];
	      vblock->rem_tet_nids[m] =
		nids[tet_verts[t * 4 + v1] - vblock->num_orig_particles];
	      vblock->rem_tet_wrap_dirs[m] =
		dirs[tet_verts[t * 4 + v1] - vblock->num_orig_particles];
	    }

	    m++;

	  } // same four remote verts 

	} // complete 

      } // I will own this tet 

    } // not strictly local 

  } // for all tets 

  // store quantities of local and nonlocal tets 
  vblock->num_loc_tets = n / 4;
  vblock->num_rem_tets = m / 4;

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

  struct rusage r_usage;
  getrusage(RUSAGE_SELF, &r_usage);

#ifdef __APPLE__
  const int to_mb = 1048576;
#else
  const int to_mb = 1024;
#endif

  fprintf(stderr, "%d: max memory = %ld MB, current memory in dashboard\n", 
	  breakpoint, r_usage.ru_maxrss / to_mb);
  sleep(dwell);
  fprintf(stderr, "%d: done\n", breakpoint);

}
// ---------------------------------------------------------------------------
