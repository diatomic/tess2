//---------------------------------------------------------------------------
//
// density field regular grid computation from voronoi tessellation
//
// Tom Peterka
// Argonne National Laboratory
// 9700 S. Cass Ave.
// Argonne, IL 60439
// tpeterka@mcs.anl.gov
//
// (C) 2013 by Argonne National Laboratory.
// See COPYRIGHT in top-level directory.
//
//--------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <vector>
#include "voronoi.h"
#include <math.h>
#include "mpi.h"
#include "diy.h"
#include "io.h"

using namespace std;

// voronoi blocks
vblock_t **vblocks;

// grid point
struct grid_pt_t {
  int idx[3]; // global grid point index
  float mass; // mass
};

// debug
float max_dense = 0.0;
float tot_mass = 0.0; // total output mass
float check_mass = 0.0; // ground truth total mass
int max_cell_grid_pts = 0; // maximum number of grid points covered by a cell
int64_t tot_interior_evals = 0; // total number of cell interior evaluations

// globals
static float data_mins[3], data_maxs[3]; // data global bounds
static float grid_phys_mins[3], grid_phys_maxs[3]; // grid physical bounds
static float grid_step_size[3]; // physical size of one grid space
static int glo_num_idx[3]; // global grid number of points
static int swap_bytes; // swap bytes flag
static char infile[256]; // input file name
static float mass; // particle mass
static char outfile[] = "dense.raw"; // output file name
static float eps = 0.0001; // epsilon for floating point values to be equal
static bool project; // whether to project to 2D
static float proj_plane[3]; // normal to projection plane

// timing
#define DENSE_MAX_TIMES 4
#define INPUT_TIME 0
#define COMP_TIME 1
#define OUTPUT_TIME 2
#define TOT_TIME 3

// function prototypes
void ParseArgs(int argc, char **argv, int *num_given_bounds,
	       float *given_mins, float *given_maxs);
void BlockGridParams(int lid, int *block_min_idx, int *block_max_idx,
		     int *block_num_idx);
void IterateCells(int block, int *block_min_idx, int *block_num_idx, 
		  float **density);
void IterateCellsCIC(int block, int *block_min_idx, int *block_num_idx, 
		     float **density);
void CellBounds(vblock_t *vblock, int c_cell, float *cell_min, float *cell_max, 
		float *centroid, vector<float> &normals);
void ComputeNormal(vblock_t *vblock, int face, float *normal);
bool PtInCell(float *pt, vblock_t *vblock, int cell, float *normals);
void Global2LocalIdx(int *global_idx, int *local_idx, int *block_min_idx);
int CellGridPts(vblock_t *vblock, int cell, float *cell_mins, float *cell_maxs, 
		grid_pt_t* &grid_pts, int * &border, int& alloc_grid_pts,
		float *normals);
int CellInteriorGridPts(vblock_t *vblock, int cell, int *cell_grid_pts, 
			int *cell_min_grid_idx, float *cell_min_grid_pos, 
			grid_pt_t *grid_pts, int *border, float *normals);
void GridStepParams(int num_given_bounds, 
		    float *given_mins, float *given_maxs);
void ItemDtype(DIY_Datatype *dtype);
void WriteGrid(float **density, MPI_Comm comm, int nblocks, 
	       int mblocks);
void handle_error(int errcode, char *str, MPI_Comm comm);
int index(int *block_grid_idx, int *block_num_idx);
void idx2phys(int *grid_idx, float *pos);
void phys2idx(float *pos, int *grid_idx);
void DataBounds(int nblocks, MPI_Comm comm);
void SummaryStats(float max_dense, float tot_mass,
		  float check_mass, int tot_cells, int max_cell_grid_pts,
		  double *times, MPI_Comm comm);
void DistributeScalarCIC(float *pt, float scalar,
			 vector <int> &grid_idxs, vector <float> &grid_scalars);
//--------------------------------------------------------------------------

int main(int argc, char** argv) {

  int dim = 3;
  int tot_blocks; // global number of blocks
  int nblocks; // my local number of blocks
  int num_threads = 1; // number of threads DIY can use
  int rank, groupsize; // MPI usual
  int did; // domain id
  MPI_Comm comm = MPI_COMM_WORLD; // MPI communicator
  int tot_cells; // total number of cells in this process

  // grid given bounds
  int num_given_bounds; // number of given bounds
  float given_mins[3], given_maxs[3]; // the given bounds

  // local block grid parameters
  int block_min_idx[3]; // global grid index of block minimum grid point
  int block_max_idx[3]; // global grid index of block maximum grid point
  int block_num_idx[3]; // number of grid points in local block

  // parse args
  ParseArgs(argc, argv, &num_given_bounds, given_mins, given_maxs);

  // ensure projection plane normal vector is unit length
  float length = sqrt(proj_plane[0] * proj_plane[0] + 
		      proj_plane[1] * proj_plane[1] +
		      proj_plane[2] * proj_plane[2]);
  proj_plane[0] /= length;
  proj_plane[1] /= length;
  proj_plane[2] /= length;

  // init
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &groupsize);

  // timing
  double times[DENSE_MAX_TIMES]; // timing
  MPI_Barrier(comm);
  times[TOT_TIME] = MPI_Wtime();
  times[INPUT_TIME] = MPI_Wtime();

  // read the tessellation
  // pnetcdf is the only version for the density estimator (no diy version)

  int *gids; // block global ids
  int *num_neighbors; // number of neighbors for each local block
  int **neighbors; // neighbors of each local block
  int **neigh_procs; // processes of neighbors of each local block
  gb_t **diy_neighs; // neighbors in diy global block format

  // read tessellation
  pnetcdf_read(&nblocks, &tot_blocks, &vblocks, argv[1], MPI_COMM_WORLD,
	       &gids, &num_neighbors, &neighbors, &neigh_procs);
  int wrap = 0; // todo: make wrap an input program argument
  bb_t bounds[nblocks]; // block bounds
  for (int i = 0; i < nblocks; i++) {
    for (int j = 0; j < dim; j++) {
      bounds[i].min[j] = vblocks[i]->mins[j];
      bounds[i].max[j] = vblocks[i]->maxs[j];
    }
  }
  int maxblocks; // max blocks in any process
  MPI_Allreduce(&nblocks, &maxblocks, 1, MPI_INT, MPI_MAX, comm);

  MPI_Barrier(comm);
  times[INPUT_TIME] = MPI_Wtime() - times[INPUT_TIME];
  times[COMP_TIME] = MPI_Wtime();

  // init diy
  diy_neighs = new gb_t*[nblocks];
  for (int i = 0; i < nblocks; i++) {
    if (num_neighbors[i])
      diy_neighs[i] = new gb_t[num_neighbors[i]];
    for (int j = 0; j < num_neighbors[i]; j++) {
      diy_neighs[i][j].gid = neighbors[i][j];
      diy_neighs[i][j].proc = neigh_procs[i][j];
    }
  }
  DIY_Init(dim, num_threads, comm);
  did = DIY_Decomposed(nblocks, gids, bounds, NULL, NULL, NULL, NULL, 
		       diy_neighs, num_neighbors, wrap);

  // cleanup temporary data
  for (int i = 0; i < nblocks; i++) {
    if (num_neighbors[i]) {
      delete[] diy_neighs[i];
      free(neighbors[i]);
      free(neigh_procs[i]);
    }
  }
  delete[] diy_neighs;
  free(neighbors);
  free(neigh_procs);
  free(num_neighbors);
  free(gids);

  // find global data bounds and grid bounds, step size
  DataBounds(nblocks, comm);
  GridStepParams(num_given_bounds, given_mins, given_maxs);

  // allocate density field
  float *density[nblocks];
  for (int block = 0; block < nblocks; block++) {
    BlockGridParams(block, block_min_idx, block_max_idx, block_num_idx);
    int npts; // total number of points in the block
    if (project)
      npts = block_num_idx[0] * block_num_idx[1];
    else
      npts = block_num_idx[0] * block_num_idx[1] * block_num_idx[2];
    density[block] = new float[npts];

    // init density
    memset(density[block], 0 , npts * sizeof(float));

    // debug
//     float min_pos[3], max_pos[3];
//     idx2phys(block_min_idx, min_pos);
//     idx2phys(block_max_idx, max_pos);
//     fprintf(stderr, "The grid in gid %d extends from [%d %d %d] "
// 	    "(%.3f %.3f %.3f) to "
// 	    "[%d %d %d] (%.3f %.3f %.3f) and is [%d %d %d] in size.\n",
// 	    DIY_Gid(did,block), 
// 	    block_min_idx[0], block_min_idx[1], block_min_idx[2],
// 	    min_pos[0], min_pos[1], min_pos[2],
// 	    block_max_idx[0], block_max_idx[1], block_max_idx[2],
// 	    max_pos[0], max_pos[1], max_pos[2],
// 	    block_num_idx[0], block_num_idx[1], block_num_idx[2]);

  }

  // divisor for volume (3d density) or area (2d density)
  // assumes projection is to x-y plane
  float div = (project ? grid_step_size[0] * grid_step_size[1] :
	       grid_step_size[0] * grid_step_size[1] * grid_step_size[2]);

  // sample the density

  for (int block = 0; block < nblocks; block++) { // blocks

    // get local block grid parameters
    BlockGridParams(block, block_min_idx, block_max_idx, block_num_idx);

    // iterate over cells, distributing density onto grid points

    // voronoi
    IterateCells(block, block_min_idx, block_num_idx, density);

    // for comarison, CIC
//     IterateCellsCIC(block, block_min_idx, block_num_idx, density);

    tot_cells += vblocks[block]->num_complete_cells;

  }

  // received items from neighbors
  void ***items = new void**[nblocks]; // received items
  int *num_items = new int[nblocks]; // number of received items in each block

  // exchange neighbors
  DIY_Exchange_neighbors(did, items, num_items, 1.0, &ItemDtype);

  // save received items
  for (int block = 0; block < nblocks; block++) {

    BlockGridParams(block, block_min_idx, block_max_idx, block_num_idx);

    for (int j = 0; j < num_items[block]; j++) {

      grid_pt_t *grid_pt = (grid_pt_t*)(items[block][j]);

      // make sure this point really goes to this block, points right on
      // the max block boundary may get sent inadvertently sent here
      if (grid_pt->idx[0] <  block_min_idx[0] ||
	  grid_pt->idx[0] >= block_max_idx[0] ||
	  grid_pt->idx[1] <  block_min_idx[1] ||
	  grid_pt->idx[1] >= block_max_idx[1] ||
	  grid_pt->idx[2] <  block_min_idx[2] ||
	  grid_pt->idx[2] >= block_max_idx[2])
	continue;

      // assign the density in the local block array
      int block_grid_idx[3]; // indices in local block array
      Global2LocalIdx(grid_pt->idx, block_grid_idx, block_min_idx);
      int idx = index(block_grid_idx, block_num_idx);
      density[block][idx] += (grid_pt->mass / div);

      // debug
      tot_mass += grid_pt->mass;
      if (density[block][idx] > max_dense)
	max_dense = density[block][idx];

    }

  }

  MPI_Barrier(comm);
  times[COMP_TIME] = MPI_Wtime() - times[COMP_TIME];
  times[OUTPUT_TIME] = MPI_Wtime();

  // write file
  WriteGrid(density, comm, nblocks, maxblocks);

  MPI_Barrier(comm);
  times[OUTPUT_TIME] = MPI_Wtime() - times[OUTPUT_TIME];

  // cleanup
  for (int i = 0; i < nblocks; i++)
    delete[] density[i];
  DIY_Flush_neighbors(did, items, num_items, &ItemDtype);
  delete[] num_items;
  delete[] items;

  MPI_Barrier(comm);
  times[TOT_TIME] = MPI_Wtime() - times[TOT_TIME];

  SummaryStats(max_dense, tot_mass, check_mass, tot_cells,
	       max_cell_grid_pts, times, comm);

  DIY_Finalize();
  MPI_Finalize();

}
//--------------------------------------------------------------------------
//
// iterate over cells and assign single density to grid point
//
// block: local block number
// block_min_idx: minimum (i,j,k) grid point index in block
// block_num_idx: number of grid points in block (output) (x,y,z)
// density: density field
//
// side effects: writes density or sends to neighbors
//
void IterateCells(int block, int *block_min_idx, int *block_num_idx, 
		  float **density) {

  float cell_min[3], cell_max[3], cell_centroid[3]; // cell bounds
  float grid_pos[3]; // physical position of grid point
  int alloc_grid_pts = 0; // number of grid points allocated
  grid_pt_t *grid_pts = NULL; // grid points covered by the cell
  int *border = NULL; // cell border, min and max x index for each y, z index
  int num_grid_pts; // number of grid points    

  // debug
//   fprintf(stderr, "gid %d has %d complete cells\n", DIY_Gid(0, block),
// 	  vblocks[block]->num_complete_cells);

  // divisor for volume (3d density) or area (2d density)
  // assumes projection is to x-y plane
  float div = (project ? grid_step_size[0] * grid_step_size[1] :
	       grid_step_size[0] * grid_step_size[1] * grid_step_size[2]);

  // cells
  for (int cell = 0; cell < vblocks[block]->num_complete_cells; cell++) {

    vector <float> normals; // cell normals

    // cell bounds
    CellBounds(vblocks[block], cell, cell_min, cell_max,
	       cell_centroid, normals);

    // debug
//     if (cell_min[0] < data_mins[0] || cell_min[1] < data_mins[1] || 
// 	cell_min[2] < data_mins[2] ||
// 	cell_max[0] > data_maxs[0] || cell_max[1] > data_maxs[1] || 
// 	cell_max[2] > data_maxs[2])
//       fprintf(stderr, "1: block = %d cell = %d cell_mins = [%.1f %.1f %.1f] "
// 	      "cell_maxs = [%.1f %.1f %.1f]\n", block, cell,
// 	      cell_min[0], cell_min[1], cell_min[2],
// 	      cell_max[0], cell_max[1], cell_max[2]);

    // debug
    check_mass++;

    // grid points covered by this cell
    num_grid_pts = CellGridPts(vblocks[block], cell, cell_min, cell_max, 
			       grid_pts,  border, alloc_grid_pts, &normals[0]);

    // debug
    if (num_grid_pts > max_cell_grid_pts)
      max_cell_grid_pts = num_grid_pts;

    // iterate over grid points covered by cell
    for (int i = 0; i < num_grid_pts; i++) {

      idx2phys(grid_pts[i].idx, grid_pos);

      // assign density to grid points in the block
      if (grid_pos[0] >= vblocks[block]->mins[0] &&
	  (grid_pos[0] < vblocks[block]->maxs[0]  ||
	   fabs(grid_pos[0] - data_maxs[0]) < eps) &&

	  grid_pos[1] >= vblocks[block]->mins[1] &&
	  (grid_pos[1] < vblocks[block]->maxs[1]  ||
	   fabs(grid_pos[1] - data_maxs[1]) < eps) &&

	  grid_pos[2] >= vblocks[block]->mins[2] &&
	  (grid_pos[2] < vblocks[block]->maxs[2]  ||
	   fabs(grid_pos[2] - data_maxs[2]) < eps) ) {

	// assign the density to the local block density array
	int block_grid_idx[3]; // local block idx of grid point
	Global2LocalIdx(grid_pts[i].idx, block_grid_idx, block_min_idx);
	int idx = index(block_grid_idx, block_num_idx);
	density[block][idx] += (grid_pts[i].mass / div);

	// debug
	tot_mass += grid_pts[i].mass;
	if (density[block][idx] > max_dense)
	  max_dense = density[block][idx];

      }

      // or send grid points to neighboring blocks
      else
	DIY_Enqueue_item_points(0, block, (void *)&grid_pts[i], NULL,
				sizeof(grid_pt_t), grid_pos, 1, NULL);

    } // grid points covered by cell

  } // cells

  if (grid_pts)
    free(grid_pts);
  if (border)
    free(border);

}
//--------------------------------------------------------------------------
// // DEPRECATED
// //
// // allocates a new block and creates DIY datatype for it
// //
// // did: domain id
// // lid: local block number
// // hdr: block header
// // dtype: pointer to datatype
// //
// // side effects: commits datatype, DIY will cleanup
// //
// // returns: address of new block
// //
// void *CreateReadType(int did, int lid, int *hdr, DIY_Datatype *dtype) {

//   did = did; // quiet compiler warning
//   lid = lid; 

//   // allocate space

//   struct vblock_t *v = new vblock_t;

//   v->num_verts = hdr[NUM_VERTS];
//   v->tot_num_cell_verts = hdr[TOT_NUM_CELL_VERTS];
//   v->num_complete_cells = hdr[NUM_COMPLETE_CELLS];
//   v->tot_num_cell_faces = hdr[TOT_NUM_CELL_FACES];
//   v->tot_num_face_verts = hdr[TOT_NUM_FACE_VERTS];
//   v->num_orig_particles = hdr[NUM_ORIG_PARTICLES];

//   if (v->num_verts > 0)
//     v->save_verts = new float[3 * v->num_verts];
//   if (v->num_orig_particles > 0) {
//     v->num_cell_verts = new int[v->num_orig_particles];
//     v->sites = new float[3 * v->num_orig_particles];
//   }
//   if (v->tot_num_cell_verts > 0)
//     v->cells = new int[v->tot_num_cell_verts];
//   if (v->num_complete_cells > 0) {
//     v->complete_cells = new int[v->num_complete_cells];
//     v->areas = new float[v->num_complete_cells];
//     v->vols = new float[v->num_complete_cells];
//     v->num_cell_faces = new int[v->num_complete_cells];
//   }
//   if (v->tot_num_cell_faces > 0)
//     v->num_face_verts = new int[v->tot_num_cell_faces];
//   if (v->tot_num_face_verts > 0)
//     v->face_verts = new int[v->tot_num_face_verts];

//   struct map_block_t map[] = {

//     { DIY_FLOAT,  OFST, 3, 
//       offsetof(struct vblock_t, mins)                 },
//     { DIY_FLOAT, ADDR, v->num_verts * 3, 
//       DIY_Addr(v->save_verts)                         },
//     { DIY_FLOAT, ADDR, v->num_orig_particles * 3,
//       DIY_Addr(v->sites)                              },
//     { DIY_INT,    ADDR, v->num_complete_cells, 
//       DIY_Addr(v->complete_cells)                     },
//     { DIY_FLOAT, ADDR, v->num_complete_cells, 
//       DIY_Addr(v->areas)                              },
//     { DIY_FLOAT, ADDR, v->num_complete_cells, 
//       DIY_Addr(v->vols)                               },
//     { DIY_INT,    ADDR, v->num_complete_cells, 
//       DIY_Addr(v->num_cell_faces)                     },
//     { DIY_INT,    ADDR, v->tot_num_cell_faces, 
//       DIY_Addr(v->num_face_verts)                     },
//     { DIY_INT,    ADDR, v->tot_num_face_verts, 
//       DIY_Addr(v->face_verts)                         },
//     { DIY_FLOAT,  OFST, 3, 
//       offsetof(struct vblock_t, maxs)                 },

//   };

//   DIY_Create_struct_datatype(DIY_Addr(v), 10, map, dtype);

//   return v;

// }
//--------------------------------------------------------------------------
//
// parse args
//
void ParseArgs(int argc, char ** argv, int *num_given_bounds,
	       float *given_mins, float *given_maxs) {

  if (argc < 9) {
    fprintf(stderr, "Usage: dense <filename> "
	    "<resample grid size x y z> <projection plane ! or x y z> <mass> <swap (0 or 1)> <cell-centered densities (0 or 1)>");
    exit(0);
  }

  strcpy(infile, argv[1]);
  glo_num_idx[0] = atoi(argv[2]);
  glo_num_idx[1] = atoi(argv[3]);
  glo_num_idx[2] = atoi(argv[4]);
  if (!strcmp(argv[5], "!")) {
    project = false;
    mass = atof(argv[6]);
    swap_bytes = atoi(argv[7]);
    *num_given_bounds = atoi(argv[8]);
    if (*num_given_bounds == 1) {
      given_mins[0] = atof(argv[9]);
      given_maxs[0] = atof(argv[10]);
    }
    else if (*num_given_bounds == 2) {
      given_mins[0] = atof(argv[9]);
      given_mins[1] = atof(argv[10]);
      given_maxs[0] = atof(argv[11]);
      given_maxs[1] = atof(argv[12]);
    }
    else if (*num_given_bounds == 3) {
      given_mins[0] = atof(argv[9]);
      given_mins[1] = atof(argv[10]);
      given_mins[2] = atof(argv[11]);
      given_maxs[0] = atof(argv[12]);
      given_maxs[1] = atof(argv[13]);
      given_maxs[2] = atof(argv[14]);
    }
  }
  if (strcmp(argv[5], "!")) {
    project = true;
    proj_plane[0] = atof(argv[5]);
    proj_plane[1] = atof(argv[6]);
    proj_plane[2] = atof(argv[7]);
    mass = atof(argv[8]);
    swap_bytes = atoi(argv[9]);
    *num_given_bounds = atoi(argv[10]);
    if (*num_given_bounds == 1) {
      given_mins[0] = atof(argv[11]);
      given_maxs[0] = atof(argv[12]);
    }
    else if (*num_given_bounds == 2) {
      given_mins[0] = atof(argv[11]);
      given_mins[1] = atof(argv[12]);
      given_maxs[0] = atof(argv[13]);
      given_maxs[1] = atof(argv[14]);
    }
    else if (*num_given_bounds == 3) {
      given_mins[0] = atof(argv[11]);
      given_mins[1] = atof(argv[12]);
      given_mins[2] = atof(argv[13]);
      given_maxs[0] = atof(argv[14]);
      given_maxs[1] = atof(argv[15]);
      given_maxs[2] = atof(argv[16]);
    }
  }

}
//--------------------------------------------------------------------------
//
// grid parameters of one local block
//
// lid: block local id
// block_min_idx: global grid idx of block minimum grid point (output) (i,j,k)
// block_max_idx: global grid idx of block maximum grid point (output) (i,j,k)
// block_num_idx: number of grid points in block (output) (i,j,k)
//
void BlockGridParams(int lid, int *block_min_idx, int *block_max_idx,
		     int *block_num_idx) {

  float pos[3]; // temporary position (x,y,z)

  // global grid index of block minimum grid point
  phys2idx(vblocks[lid]->mins, block_min_idx);
  idx2phys(block_min_idx, pos);
  if (pos[0] < vblocks[lid]->mins[0])
    block_min_idx[0]++;
  if (pos[1] < vblocks[lid]->mins[1])
    block_min_idx[1]++;
  if (pos[2] < vblocks[lid]->mins[2])
    block_min_idx[2]++;
  idx2phys(block_min_idx, pos); // double check adjusted position
  assert(pos[0] >= vblocks[lid]->mins[0] && pos[1] >= vblocks[lid]->mins[1] &&
	 pos[2] >= vblocks[lid]->mins[2]);

  // global grid index of block maximum grid point
  phys2idx(vblocks[lid]->maxs, block_max_idx);
  idx2phys(block_max_idx, pos);
  if (pos[0] + grid_step_size[0] <= vblocks[lid]->maxs[0])
    block_max_idx[0]++;
  if (pos[1] + grid_step_size[1] <= vblocks[lid]->maxs[1])
    block_max_idx[1]++;
  if (pos[2] + grid_step_size[2] <= vblocks[lid]->maxs[2])
    block_max_idx[2]++;
  idx2phys(block_max_idx, pos); // double check adjusted position
  assert(pos[0] <= vblocks[lid]->maxs[0] && pos[1] <= vblocks[lid]->maxs[1] &&
	 pos[2] <= vblocks[lid]->maxs[2]);

  // eliminate duplication at the maximum block border
  if (fabs(data_mins[0] + block_max_idx[0] * grid_step_size[0] -
	   vblocks[lid]->maxs[0]) < eps && 
      fabs(vblocks[lid]->maxs[0] - data_maxs[0]) > grid_step_size[0])
    block_max_idx[0]--;
  if (fabs(data_mins[1] + block_max_idx[1] * grid_step_size[1] -
      vblocks[lid]->maxs[1]) < eps &&
      fabs(vblocks[lid]->maxs[1] - data_maxs[1]) > grid_step_size[1])
    block_max_idx[1]--;
  if (fabs(data_mins[2] + block_max_idx[2] * grid_step_size[2] -
      vblocks[lid]->maxs[2]) < eps &&
      fabs(vblocks[lid]->maxs[2] - data_maxs[2]) > grid_step_size[2])
    block_max_idx[2]--;

  // possibly extend minimum end of blacks at the minimum end of the domain
  if (fabs(vblocks[lid]->mins[0] - data_mins[0]) < grid_step_size[0])
    block_min_idx[0] = 0;
  if (fabs(vblocks[lid]->mins[1] - data_mins[1]) < grid_step_size[1])
    block_min_idx[1] = 0;
  if (fabs(vblocks[lid]->mins[2] - data_mins[2]) < grid_step_size[2])
    block_min_idx[2] = 0;

  // possibly extend maximum end of blacks at the maximum end of the domain
  if (fabs(vblocks[lid]->maxs[0] - data_maxs[0]) < grid_step_size[0])
    block_max_idx[0] = glo_num_idx[0] - 1;
  if (fabs(vblocks[lid]->maxs[1] - data_maxs[1]) < grid_step_size[1])
    block_max_idx[1] = glo_num_idx[1] - 1;
  if (fabs(vblocks[lid]->maxs[2] - data_maxs[2]) < grid_step_size[2])
    block_max_idx[2] = glo_num_idx[2] - 1;

  // compute number of grid points in local block
  block_num_idx[0] = block_max_idx[0] - block_min_idx[0] + 1;
  block_num_idx[1] = block_max_idx[1] - block_min_idx[1] + 1;
  block_num_idx[2] = block_max_idx[2] - block_min_idx[2] + 1;

}
//--------------------------------------------------------------------------
//
// get cell bounds
//
// vblock: one voronoi block
// c_cell: current cell counter
// cell_min, cell_max: cell bounds (output)
// centroid: centroid, mean of all vertices (output)
// normals: face normals (nx_0,ny_0,nz_0,nx_1,ny_1,nz_1, ...) (output)
//
void CellBounds(vblock_t *vblock, int c_cell, float *cell_min, float *cell_max, 
		float *centroid, vector<float> &normals) {

  float n[3]; // face normal

  centroid[0] = 0.0;
  centroid[1] = 0.0;
  centroid[2] = 0.0;
  int tot_verts = 0;

  int cell = vblock->complete_cells[c_cell];
  int num_faces; // number of faces in the current cell
  int num_verts; // number of vertices in the current face

  // debug
  if (cell >= vblock->num_orig_particles)
    fprintf(stderr, "block mins [%.1f %.1f %.1f] has cell %d >= "
	    "num_orig_particles %d\n",
	    vblock->mins[0], vblock->mins[1], vblock->mins[2], cell,
	    vblock->num_orig_particles);

  // number of faces in the current cell
  if (cell < vblock->num_orig_particles - 1)
    num_faces = vblock->cell_faces_start[cell + 1] -
      vblock->cell_faces_start[cell];
  else
    num_faces = vblock->tot_num_cell_faces -
      vblock->cell_faces_start[cell];

  // grow vectors to correct size
  normals.reserve(3 * num_faces);

  // get cell bounds
  for (int k = 0; k < num_faces; k++) { // faces

    int start = vblock->cell_faces_start[cell];
    int face = vblock->cell_faces[start + k];
    num_verts = vblock->faces[face].num_verts;

    // normal
    ComputeNormal(vblock, face, n);
    // check sign of dot product of normal with vector from site 
    // to first face vertex to see if normal has correct direction
    // want outward normal
    int v0 = vblock->faces[face].verts[0];
    float v[3];
    v[0] = vblock->save_verts[3 * v0] - vblock->sites[3 * cell];
    v[1] = vblock->save_verts[3 * v0 + 1] - vblock->sites[3 * cell + 1];
    v[2] = vblock->save_verts[3 * v0 + 2] - vblock->sites[3 * cell + 2];
    if (v[0] * n[0] + v[1] * n[1] + v[2] * n[2] < 0.0) {
      n[0] *= -1.0;
      n[1] *= -1.0;
      n[2] *= -1.0;
    }
    normals.push_back(n[0]);
    normals.push_back(n[1]);
    normals.push_back(n[2]);

    // debug
//     fprintf(stderr, "1: c_cell %d cell %d k %d face %d normal [%.3f %.3f %.3f]\n",
// 	    c_cell, cell, k, face, n[0], n[1], n[2]);

    for (int l = 0; l < num_verts; l++) { // vertices

      int v = vblock->faces[face].verts[l];

      // extrema for entire cell
      if (k == 0 && l == 0 || vblock->save_verts[3 * v] < cell_min[0])
	cell_min[0] = vblock->save_verts[3 * v];
      if (k == 0 && l == 0 || vblock->save_verts[3 * v] > cell_max[0])
	cell_max[0] = vblock->save_verts[3 * v];
      centroid[0] += vblock->save_verts[3 * v];

      if (k == 0 && l == 0 || vblock->save_verts[3 * v + 1] < cell_min[1])
	cell_min[1] = vblock->save_verts[3 * v + 1];
      if (k == 0 && l == 0 || vblock->save_verts[3 * v + 1] > cell_max[1])
	cell_max[1] = vblock->save_verts[3 * v + 1];
      centroid[1] += vblock->save_verts[3 * v + 1];

      if (k == 0 && l == 0 || vblock->save_verts[3 * v + 2] < cell_min[2])
	cell_min[2] = vblock->save_verts[3 * v + 2];
      if (k == 0 && l == 0 || vblock->save_verts[3 * v + 2] > cell_max[2])
	cell_max[2] = vblock->save_verts[3 * v + 2];
      centroid[2] += vblock->save_verts[3 * v + 2];

      tot_verts++;

    } // vertices

  } // faces

  centroid[0] /= tot_verts;
  centroid[1] /= tot_verts;
  centroid[2] /= tot_verts;

} 
//--------------------------------------------------------------------------
//
// makes DIY datatype for sending and receiving one item
//
// dtype: pointer to the datatype
//
void ItemDtype(DIY_Datatype *dtype) {

  struct map_block_t map[] = {
    {DIY_INT,   OFST, 3, offsetof(grid_pt_t, idx)}, // global grid index
    {DIY_FLOAT, OFST, 1, offsetof(grid_pt_t, mass)}, // mass
  };
  DIY_Create_struct_datatype(0, 2, map, dtype);

}
//--------------------------------------------------------------------------
//
// write density grid
//
// density: density field
// comm: MPI communicator
// nblocks: local number of blocks
// mblocks: max number of blocks in any process
//
void WriteGrid(float **density, MPI_Comm comm, int nblocks, int mblocks) {

  MPI_Status status;
  int pts_written;
  MPI_File fd; 
  int block_min_idx[3]; // global grid index of block minimum grid point
  int block_max_idx[3]; // global grid index of block maximum grid point
  int block_num_idx[3]; // number of grid points in local block
  int sizes[3]; // sizes of global array
  int subsizes[3]; // sizes of subarrays
  int starts[3]; // starting offsets of subarrays
  MPI_Datatype dtype; // subarray datatype

  // open
  int retval = MPI_File_open(comm, (char *)outfile,
			     MPI_MODE_WRONLY | MPI_MODE_CREATE,
			     MPI_INFO_NULL, &fd);
  assert(retval == MPI_SUCCESS);
  MPI_File_set_size(fd, 0); // start with an empty file every time

  // write
  for (int block = 0; block < mblocks; block++) {

    if (block < nblocks) { // non-null block

      int num_pts; // total number of points per block

      // get local block grid parameters
      BlockGridParams(block, block_min_idx, block_max_idx, block_num_idx);

      if (project) {

	// reversed order intentional
	sizes[0] = glo_num_idx[1];
	sizes[1] = glo_num_idx[0];
	starts[0] = block_min_idx[1];
	starts[1] = block_min_idx[0];
	subsizes[0] = block_num_idx[1];
	subsizes[1] = block_num_idx[0];

	MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C,
				 MPI_FLOAT, &dtype);
	MPI_Type_commit(&dtype);
	MPI_File_set_view(fd, 0, MPI_FLOAT, dtype, (char *)"native", 
			  MPI_INFO_NULL);

	num_pts = block_num_idx[0] * block_num_idx[1];

      } else {

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

	MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
				 MPI_FLOAT, &dtype);
	MPI_Type_commit(&dtype);
	MPI_File_set_view(fd, 0, MPI_FLOAT, dtype, (char *)"native", 
			  MPI_INFO_NULL);

	num_pts = block_num_idx[0] * block_num_idx[1] * block_num_idx[2];

      }

      // write block
      int errcode = MPI_File_write_all(fd, density[block], num_pts, 
					  MPI_FLOAT, &status);
      if (errcode != MPI_SUCCESS)
	handle_error(errcode, (char *)"MPI_File_write_all nonempty datatype", 
		     comm);
      MPI_Get_count(&status, MPI_FLOAT, &pts_written);
      assert(pts_written == num_pts);

      MPI_Type_free(&dtype);

    }

    else { // null block
      float unused;
      MPI_File_set_view(fd, 0, MPI_FLOAT, MPI_FLOAT, (char *)"native", 
			MPI_INFO_NULL);
      MPI_File_write_all(fd, &unused, 0, MPI_FLOAT, &status);
    }

  }

  // close
  MPI_File_close(&fd);

}
//--------------------------------------------------------------------------
//
// MPI error handler
// decodes and prints MPI error messages
//
void handle_error(int errcode, char *str, MPI_Comm comm) {

  char msg[MPI_MAX_ERROR_STRING];
  int resultlen;
  MPI_Error_string(errcode, msg, &resultlen);
  fprintf(stderr, "%s: %s\n", str, msg);
  MPI_Abort(comm, 1);

}
//-----------------------------------------------------------------------
//
// compute 1-d index in a block
// points in a block are listed in row major order
//
// block_grid_idx: 3d index in this block (x,y,z)
// block_num_idx: number of pts in each dimension in this block (x,y,z)
//
//
// returns: 1-d index
//
int index(int *block_grid_idx, int *block_num_idx) {

  int idx[3]; // index after possible projection
  float proj_length; // projected length of idx onto normal vector
  float shift[3]; // shift in idx due to projection

  // project index into plane
  if (project) {

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
  else {

    idx[0] = block_grid_idx[0];
    idx[1] = block_grid_idx[1];
    idx[2] = block_grid_idx[2];

  }

  return (idx[2] * block_num_idx[1] * block_num_idx[0] +
	  idx[1] * block_num_idx[0] + 
	  idx[0]);

}
//-----------------------------------------------------------------------
//
// physical position (x,y,z) of a global grid index (i,j,k)
//
// grid_idx: global grid index (x,y,z)
// pos: physical position (x,y,z) (output)
//
void idx2phys(int *grid_idx, float *pos) {

  pos[0] = grid_idx[0] * grid_step_size[0] + grid_phys_mins[0];
  pos[1] = grid_idx[1] * grid_step_size[1] + grid_phys_mins[1];
  pos[2] = grid_idx[2] * grid_step_size[2] + grid_phys_mins[2];

}
//-----------------------------------------------------------------------
//
// global grid index (i,j,k) of physical position (x,y,z)
// grid index found by integer division (truncating, not rounding)
// so grid index will be at or just before the physical position
//
// pos: physical position (x,y,z)
// grid_idx: global grid index (x,y,z) (output)
//
void phys2idx(float *pos, int *grid_idx) {

  grid_idx[0] = (pos[0] - grid_phys_mins[0]) / grid_step_size[0];
  grid_idx[1] = (pos[1] - grid_phys_mins[1]) / grid_step_size[1];
  grid_idx[2] = (pos[2] - grid_phys_mins[2]) / grid_step_size[2];

}
//-----------------------------------------------------------------------
//
// compute normal of a face using Newell's method
//
// Newell's method is more robust than simply computing the cross product of
//   three points when the points are colinear or slightly nonplanar. 
//
// vblock: one voronoi block
// face: face id
// normal: (output) normal, allocated by caller
//
void ComputeNormal(vblock_t *vblock, int face, float *normal) {

  int v; // index of vertex
  float cur[3], next[3]; // current and next vertex

  normal[0] = 0.0;
  normal[1] = 0.0;
  normal[2] = 0.0;

  for (int i = 0; i < vblock->faces[face].num_verts; i++) {

    // get current and next vertex going around the face
    v = vblock->faces[face].verts[i];
    cur[0] = vblock->save_verts[3 * v];
    cur[1] = vblock->save_verts[3 * v + 1];
    cur[2] = vblock->save_verts[3 * v + 2];
    if (i + 1 < vblock->faces[face].num_verts)
      v = vblock->faces[face].verts[i + 1]; // next vertex 
    else
      v = vblock->faces[face].verts[0]; // next = last = first vertex
    next[0] = vblock->save_verts[3 * v];
    next[1] = vblock->save_verts[3 * v + 1];
    next[2] = vblock->save_verts[3 * v + 2];

    normal[0] += (cur[1] - next[1]) * (cur[2] + next[2]);
    normal[1] += (cur[2] - next[2]) * (cur[0] + next[0]);
    normal[2] += (cur[0] - next[0]) * (cur[1] + next[1]);

  }

  // normalize
  float mag = sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
		   normal[2] * normal[2]);
  normal[0] /= mag;
  normal[1] /= mag;
  normal[2] /= mag;

}
//--------------------------------------------------------------------------
//
// whether a point lies inside a cell
//
// pt: point
// vblock: one voronoi block
// c_cell: current complete cell counter
// normals: face normals (nx_0,ny_0,nz_0,nx_1,ny_1,nz_1, ...)
//
// returns whether point is in cell (true) or not (false)
//
bool PtInCell(float *pt, vblock_t *vblock, int c_cell, float *normals) {

  int sign; // sign of distance (1 or -1)
  int old_sign = 0; // previous sign, 0 = uninitialized
  int v; // vertex id
  float v0[3]; // first vertex
  float dist = 0.0; // signed distance from point to plane

  int cell = vblock->complete_cells[c_cell]; // current cell
  int num_faces; // number of faces in the current cell
  if (cell < vblock->num_orig_particles - 1)
    num_faces = vblock->cell_faces_start[cell + 1] -
      vblock->cell_faces_start[cell];
  else
    num_faces = vblock->tot_num_cell_faces -
      vblock->cell_faces_start[cell];

  for (int k = 0; k < num_faces; k++) { // faces

    int start = vblock->cell_faces_start[cell];
    int face = vblock->cell_faces[start + k];

    dist = 0.0;

    // compute distance from point to face

    // first vertex in the face
    v = vblock->faces[face].verts[0];
    v0[0] = vblock->save_verts[3 * v];
    v0[1] = vblock->save_verts[3 * v + 1];
    v0[2] = vblock->save_verts[3 * v + 2];

    float *n = &(normals[3 * k]); // current normal
    dist = n[0] * (pt[0] - v0[0]) + n[1] * (pt[1] - v0[1]) + 
      n[2] * (pt[2] - v0[2]);

    // check sign of distance only if non-zero
    if (fabs(dist) > eps) {
      sign = (dist >= 0.0 ? 1 : -1);
      if (old_sign == 0)
	old_sign = sign;
      if (old_sign != sign)
	return false;
    }

  } // faces

  return true;

}
//--------------------------------------------------------------------------
//
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
// vblock: one voronoi block
// cell: current cell counter
// cell_mins: minimum cell vertex (x,y,z)
// cell_maxs: maximum cell vertex (x,y,z)
// grid_pts: (output) grid points covered by this cell, allocated by this
//   function, caller's responsibility to free
// border: cell border, min and max x index for each y and z index
// alloc_grid_pts: number of grid points currently allocated, this function
//   will realloc to the new size if needed, otherwise will leave old size
// normals: face normals (nx_0,ny_0,nz_0,nx_1,ny_1,nz_1, ...)
//
// returns: number of grid points covered by this cell
//
int CellGridPts(vblock_t *vblock, int cell, float *cell_mins, 
		float *cell_maxs,  grid_pt_t* &grid_pts, int* &border, 
		int &alloc_grid_pts, float *normals) {

  float center[3]; // cell center
  int num_grid_pts; // number of grid points covered by this cell

  // global grid index of cell minimum grid point
  int cell_min_grid_idx[3];
  phys2idx(cell_mins, cell_min_grid_idx);

  // global grid index of cell maximum grid point
  int cell_max_grid_idx[3];
  phys2idx(cell_maxs, cell_max_grid_idx);

  // cell minimum grid point physical position
  float cell_min_grid_pos[3];
  idx2phys(cell_min_grid_idx, cell_min_grid_pos);

  // number of grid points covered by cell bounding box
  int cell_grid_pts[3];
  cell_grid_pts[0] = cell_max_grid_idx[0] - cell_min_grid_idx[0] + 1;
  cell_grid_pts[1] = cell_max_grid_idx[1] - cell_min_grid_idx[1] + 1;
  cell_grid_pts[2] = cell_max_grid_idx[2] - cell_min_grid_idx[2] + 1;

  // grid_pts and border memory allocation
  int npts = cell_grid_pts[0] * cell_grid_pts[1] * cell_grid_pts[2];

  if (!alloc_grid_pts) {
    grid_pts = (grid_pt_t *)malloc(npts * sizeof(grid_pt_t));
    border = (int *)malloc(npts * 2 * sizeof(int)); // more than large enough
    alloc_grid_pts = npts;
  }  else if (npts > alloc_grid_pts) {
    grid_pts = (grid_pt_t *)realloc(grid_pts, npts * sizeof(grid_pt_t));
    border = (int *)realloc(border, npts * sizeof(grid_pt_t));
    alloc_grid_pts = npts;
  }

  num_grid_pts = CellInteriorGridPts(vblock, cell, cell_grid_pts, 
				     cell_min_grid_idx, cell_min_grid_pos, 
				     grid_pts, border, normals);

  // debug
//   fprintf(stderr, "2: cell = %d num_grid_pts = %d\n", cell, num_grid_pts);

  // if no grid points covered by cell, pick a single grid point near to the
  // cell centroid
  if (!num_grid_pts) {

    center[0] = (cell_mins[0] + cell_maxs[0]) / 2.0f;
    center[1] = (cell_mins[1] + cell_maxs[1]) / 2.0f;
    center[2] = (cell_mins[2] + cell_maxs[2]) / 2.0f;
    phys2idx(center, grid_pts[num_grid_pts].idx);

    // deposit mass onto one grid point, density will be computed from this later
    grid_pts[num_grid_pts].mass = mass;
    num_grid_pts = 1;

  }

  return num_grid_pts;

}
//--------------------------------------------------------------------------
//
// convert global grid index to local block grid index
//
// global_idx: global grid index (i,j,k)
// local_idx: (output) local grid index in block (i,j,k)
// block_min_dx: minimum index (i,j,k) in local block
//
void Global2LocalIdx(int *global_idx, int *local_idx, int *block_min_idx) {

  // compute local index of current grid point in this block
  local_idx[0] = global_idx[0] - block_min_idx[0];
  local_idx[1] = global_idx[1] - block_min_idx[1];
  local_idx[2] = global_idx[2] - block_min_idx[2];

}
//--------------------------------------------------------------------------
//
// finds global data bounds
// 
// nblocks: local number of blocks
// comm: MPI communicator
//
void DataBounds(int nblocks, MPI_Comm comm) {

  float block_mins[3]; // mins of all local blocks
  float block_maxs[3]; // maxs of all local blocks
  int rank;

  MPI_Comm_rank(comm, &rank);

  for (int i = 0; i < nblocks; i++) {
    bb_t bb; // block bounds
    DIY_Block_bounds(0, i, &bb);
    if (i == 0) {
      block_mins[0] = bb.min[0];
      block_mins[1] = bb.min[1];
      block_mins[2] = bb.min[2];
      block_maxs[0] = bb.max[0];
      block_maxs[1] = bb.max[1];
      block_maxs[2] = bb.max[2];
    } else {
      if (bb.min[0] < block_mins[0])
	block_mins[0] = bb.min[0];
      if (bb.min[1] < block_mins[1])
	block_mins[1] = bb.min[1];
      if (bb.min[2] < block_mins[2])
	block_mins[2] = bb.min[2];
      if (bb.max[0] > block_maxs[0])
	block_maxs[0] = bb.max[0];
      if (bb.max[1] > block_maxs[1])
	block_maxs[1] = bb.max[1];
      if (bb.max[2] > block_maxs[2])
	block_maxs[2] = bb.max[2];
    }

  }

  MPI_Allreduce(block_mins, data_mins, 3, MPI_FLOAT, MPI_MIN, comm);
  MPI_Allreduce(block_maxs, data_maxs, 3, MPI_FLOAT, MPI_MAX, comm);

  if (rank == 0)
    fprintf(stderr, "data bounds: min = [%.3f %.3f %.3f] max[%.3f %.3f %.3f]\n",
	    data_mins[0], data_mins[1], data_mins[2],
	    data_maxs[0], data_maxs[1], data_maxs[2]);

}
//--------------------------------------------------------------------------
//
// print summary stats
//
// max_dense: local maximum density
// tot_mass: local total mass
// check_mass: local ground truth total mass
// tot_cells: local total number of cells
// max_cell_grid_pts: maximum number of grid points covered by a cell
// times: timing info
// comm: MPI cmmunicator
//
void SummaryStats(float max_dense, float tot_mass,
		  float check_mass, int tot_cells, int max_cell_grid_pts,
		  double *times, MPI_Comm comm) {

  int rank;
  MPI_Comm_rank(comm, &rank);

  float glo_max_dense = 0; // global max density
  float glo_tot_mass = 0; // global total mass
  float glo_check_mass = 0; // global reference total mass
  int glo_tot_cells = 0; // global number of cells
  int glo_max_cell_grid_pts = 0; // global maximum number of grid points
                             // covered by a cell

  MPI_Reduce(&max_dense, &glo_max_dense, 1, MPI_FLOAT, MPI_MAX, 0, comm);
  MPI_Reduce(&tot_mass, &glo_tot_mass, 1, MPI_FLOAT, MPI_SUM, 0, comm);
  MPI_Reduce(&check_mass, &glo_check_mass, 1, MPI_FLOAT, MPI_SUM, 0, comm);
  MPI_Reduce(&tot_cells, &glo_tot_cells, 1, MPI_INT, MPI_SUM, 0, comm);
  MPI_Reduce(&max_cell_grid_pts, &glo_max_cell_grid_pts, 1, MPI_INT, 
	     MPI_MAX, 0, comm);

  // physical positions of global grid extents
  float grid_min_pos[3], grid_max_pos[3];
  int idx[3];  // grid index
  idx[0] = 0;
  idx[1] = 0;
  idx[2] = 0;
  idx2phys(idx, grid_min_pos);
  idx[0] = glo_num_idx[0] - 1;
  idx[1] = glo_num_idx[1] - 1;
  idx[2] = glo_num_idx[2] - 1;
  idx2phys(idx, grid_max_pos);

  if (rank == 0) {
    fprintf(stderr, "--------------Summary--------------\n");
    fprintf(stderr, "Total number of voronoi cells = %d\n", glo_tot_cells);
    fprintf(stderr, "Global 3D grid size = %d x %d x %d\n",
	    glo_num_idx[0], glo_num_idx[1], glo_num_idx[2]);
    fprintf(stderr, "Physical grid extends from min [%.4e %.4e %.4e]\n"
	    "to max [%.4e %.4e %.4e]\n"
	    "by steps of [%.4e %.4e %.4e]\n",
	    grid_min_pos[0], grid_min_pos[1], grid_min_pos[2],
	    grid_max_pos[0], grid_max_pos[1], grid_max_pos[2],
	    grid_step_size[0], grid_step_size[1], grid_step_size[2]);
    fprintf(stderr, "Global max number of grid points in largest cell = %d\n",
	    glo_max_cell_grid_pts);
    fprintf(stderr, "max_dense = %.3e tot_mass = %.3e (should be %.3e)\n",
	    glo_max_dense, glo_tot_mass, glo_check_mass);
    fprintf(stderr, "Total number of cell interior evaluations = %.lld\n", 
	    tot_interior_evals);
    fprintf(stderr, "Total time = %.3lf s = \n", times[TOT_TIME]);
    fprintf(stderr, "%.3lf s input + %.3lf s density computation + "
	    "%.3lf s output\n",
	    times[INPUT_TIME], times[COMP_TIME], times[OUTPUT_TIME]);
    fprintf(stderr, "-----------------------------------\n");
  }

}
//--------------------------------------------------------------------------
#if 0
// DEPRECATED naive O(n^3) version
//
// finds interior grid points in cell and sets density at them
//
// vblock: one voronoi block
// cell: current cell counter
// cell_grid_pts: number of grid points covered by cell bounding box
// cell_min_grid_idx: cell minimum grid point global index
// cell_min_grid_pos: cell minimum grid point physical position
// grid_pts: (output) grid points covered by this cell, allocated by caller
// border: cell border, min and max x index for each y and z index
// normals: face normals (nx_0,ny_0,nz_0,nx_1,ny_1,nz_1, ...)
//
// returns: number of interior grid points
//
int CellInteriorGridPts(vblock_t *vblock, int cell, int *cell_grid_pts, 
			int *cell_min_grid_idx, float *cell_min_grid_pos, 
			grid_pt_t *grid_pts, int *border, float *normals) {

  int num_grid_pts = 0; // current number of grid points interior to cell
  int tot_num_grid_pts = 0; // total number of grid points interior to cell
  float grid_pos[3]; // physical position of current grid point

  // find the interior of the cell
  for (int zi = 0; zi < cell_grid_pts[2]; zi++) { // z
    for (int yi = 0; yi < cell_grid_pts[1]; yi++) { // y
      for (int xi = 0; xi < cell_grid_pts[0]; xi++) { // x
	grid_pos[0] = cell_min_grid_pos[0] + xi * grid_step_size[0];
	grid_pos[1] = cell_min_grid_pos[1] + yi * grid_step_size[1];
	grid_pos[2] = cell_min_grid_pos[2] + zi * grid_step_size[2];
	tot_interior_evals++;
	if (PtInCell(grid_pos, vblock, cell, normals)) {
	  grid_pts[num_grid_pts].idx[0] = cell_min_grid_idx[0] + xi;
	  grid_pts[num_grid_pts].idx[1] = cell_min_grid_idx[1] + yi;
	  grid_pts[num_grid_pts].idx[2] = cell_min_grid_idx[2] + zi;
	  grid_pts[num_grid_pts].mass = mass;
	  tot_num_grid_pts++;
	}
	else
	  grid_pts[num_grid_pts].mass = 0.0f;
	num_grid_pts++;

      }
    }
  }

  // divide the mass at each grid point by the total number of grid points
  num_grid_pts = 0;
  for (int zi = 0; zi < cell_grid_pts[2]; zi++) { // z
    for (int yi = 0; yi < cell_grid_pts[1]; yi++) { // y
      for (int xi = 0; xi < cell_grid_pts[0]; xi++) { // x

	if (grid_pts[num_grid_pts].mass)
	  grid_pts[num_grid_pts].mass = mass / (float)tot_num_grid_pts;
	num_grid_pts++;

      }
    }
  }

  return num_grid_pts;

}
#endif
//--------------------------------------------------------------------------
#if 0
// DEPRECATED version
// improved over naive version by limiting x scan
//
// finds interior grid points in cell and sets density at them
//
// vblock: one voronoi block
// cell: current cell counter
// cell_grid_pts: number of grid points covered by cell bounding box
// cell_min_grid_idx: cell minimum grid point global index
// cell_min_grid_pos: cell minimum grid point physical position
// grid_pts: (output) grid points covered by this cell, allocated by caller
// border: cell border, min and max x index for each y and z index
// normals: face normals (nx_0,ny_0,nz_0,nx_1,ny_1,nz_1, ...)
//
// returns: number of interior grid points
//
int CellInteriorGridPts(vblock_t *vblock, int cell, int *cell_grid_pts, 
			int *cell_min_grid_idx, float *cell_min_grid_pos, 
			grid_pt_t *grid_pts, int *border, float *normals) {

  int num_grid_pts = 0; // current number of grid points interior to cell
  int tot_num_grid_pts = 0; // total number of grid points interior to cell
  float grid_pos[3]; // physical position of current grid point
  int old_x_left = cell_grid_pts[0] / 2; // old x index stepping left
  int old_x_right = cell_grid_pts[0] / 2; // old x index stepping right
  bool x_in_left, x_in_right; // pt inside cell while stepping leftk right
  int min_xi, max_xi; // min, max x index of border crossing

  // find the interior of the cell

  for (int zi = 0; zi < cell_grid_pts[2]; zi++) { // z
    for (int yi = 0; yi < cell_grid_pts[1]; yi++) { // y

      // init the stepping
      grid_pos[0] = cell_min_grid_pos[0] + old_x_left * grid_step_size[0];
      grid_pos[1] = cell_min_grid_pos[1] + yi * grid_step_size[1];
      grid_pos[2] = cell_min_grid_pos[2] + zi * grid_step_size[2];
      tot_interior_evals++;
      if (PtInCell(grid_pos, vblock, cell, face, vert, normals)) {
	x_in_left = true;
	x_in_right = true;
      }
      else {
	x_in_left = false;
	x_in_right = false;
      }
      min_xi = cell_grid_pts[0] - 1;
      max_xi = 0;

      // x step left
      for (int xi = old_x_left; xi >= 0 && xi < cell_grid_pts[0];) {

	grid_pos[0] = cell_min_grid_pos[0] + xi * grid_step_size[0];
	grid_pos[1] = cell_min_grid_pos[1] + yi * grid_step_size[1];
	grid_pos[2] = cell_min_grid_pos[2] + zi * grid_step_size[2];

	tot_interior_evals++;
	if (PtInCell(grid_pos, vblock, cell, normals)) {
	  if (x_in_left) { // remains interior, keep stepping
	    if (xi < min_xi)
	      min_xi = xi;
	    if (xi > max_xi)
	      max_xi = xi;
	    xi--;
	  }
	  else { // edge crossing from interior to exterior
	    if (xi < min_xi)
	      min_xi = xi;
	    if (xi > max_xi)
	      max_xi = xi;
	    old_x_left = xi;
	    break;
	  }
	} // pt is inside the cell

	// pt is outside the cell
	else {
	  if (!x_in_left) // remains exterior, keep stepping
	    xi++;
	  else { // edge crossing from exterior to interior
	    old_x_left = xi;
	    break;
	  }
	} // pt is outside the cell

      } // x step left

      // x step right
      for (int xi = old_x_right; xi >= 0 && xi < cell_grid_pts[0];) {

	grid_pos[0] = cell_min_grid_pos[0] + xi * grid_step_size[0];
	grid_pos[1] = cell_min_grid_pos[1] + yi * grid_step_size[1];
	grid_pos[2] = cell_min_grid_pos[2] + zi * grid_step_size[2];

	tot_interior_evals++;
	if (PtInCell(grid_pos, vblock, cell, face, vert, normals)) {
	  if (x_in_right) {// remains interior, keep stepping
	    if (xi < min_xi) 
	      min_xi = xi;
	    if (xi > max_xi)
	      max_xi = xi;
	    xi++;
	  } 
	  else { // edge crossing from interior to exterior
	    if (xi < min_xi)
	      min_xi = xi;
	    if (xi > max_xi)
	      max_xi = xi;
	    old_x_right = xi;
	    break;
	  }
	} // pt is inside the cell

	// pt is outside the cell
	else {
	  if (!x_in_right) // remains exterior, keep stepping
	    xi--;
	  else { // edge crossing from exterior to interior
	    old_x_right = xi;
	    break;
	  }
	} // pt is outside the cell

      } // x step right

      border[2 * (zi * cell_grid_pts[1] + yi)]     = min_xi;
      border[2 * (zi * cell_grid_pts[1] + yi) + 1] = max_xi;
      // if min_xi > max_xi, then no points were found
      if (min_xi <= max_xi)
	tot_num_grid_pts += (max_xi - min_xi + 1);

    } // y

  } // z

  // deposit the density in the interior of the cell
  for (int zi = 0; zi < cell_grid_pts[2]; zi++) { // z
    for (int yi = 0; yi < cell_grid_pts[1]; yi++) { // y

      min_xi = border[2 * (zi * cell_grid_pts[1] + yi)];
      max_xi = border[2 * (zi * cell_grid_pts[1] + yi) + 1];

      for (int xi = min_xi; xi <= max_xi; xi++) { // x

	grid_pts[num_grid_pts].idx[0] = cell_min_grid_idx[0] + xi;
	grid_pts[num_grid_pts].idx[1] = cell_min_grid_idx[1] + yi;
	grid_pts[num_grid_pts].idx[2] = cell_min_grid_idx[2] + zi;
	// deposit mass onto grid points, density to be computed later
	grid_pts[num_grid_pts].mass = mass / (float)tot_num_grid_pts;
	num_grid_pts++;

      }
    }
  }
  // debug
//   fprintf(stderr, "cell = %d tot_num_grid_pts = %d num_grid_pts = %d\n",
// 	  cell, tot_num_grid_pts, num_grid_pts);

  // cleanup
  assert(tot_num_grid_pts == num_grid_pts); // sanity
  return num_grid_pts;

}
#endif
//--------------------------------------------------------------------------
#if 1
//
// finds interior grid points in cell and sets density at them
// current version further improved by limiting y scan
//
// vblock: one voronoi block
// cell: current cell counter
// cell_grid_pts: number of grid points covered by cell bounding box
// cell_min_grid_idx: cell minimum grid point global index
// cell_min_grid_pos: cell minimum grid point physical position
// grid_pts: (output) grid points covered by this cell, allocated by caller
// border: cell border, min and max x index for each y and z index
// normals: face normals (nx_0,ny_0,nz_0,nx_1,ny_1,nz_1, ...)
//
// returns: number of interior grid points
//
int CellInteriorGridPts(vblock_t *vblock, int cell, int *cell_grid_pts, 
			int *cell_min_grid_idx, float *cell_min_grid_pos, 
			grid_pt_t *grid_pts, int *border, float *normals) {

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
  int first_x; // x index of border crossing at first y line in each z

  int y_steps = 0; // see how many y_steps we ended up making

  // find the border points of the cell

  // z step
  for (zi = 0; zi < cell_grid_pts[2]; zi++) {

    grid_pos[2] = cell_min_grid_pos[2] + zi * grid_step_size[2];
    border_found = false; // init
    z_step_done = false;

    // initialize (with min > max) unused y-scan lines
    for (yj = 0; yj < y_start; yj++) { // prior to start
      border[2 * (zi * cell_grid_pts[1] + yj)]     = 1; // min
      border[2 * (zi * cell_grid_pts[1] + yj) + 1] = 0; // max
    }

    // y step
    // even though the upper loop bound is the full number of grid points,
    // early termination will occur deep in the body of the loop when the
    // number of x-intersections becomes 0
    for (yi = y_start; yi < cell_grid_pts[1]; yi++) {

      // debug
      y_steps++;

      grid_pos[1] = cell_min_grid_pos[1] + yi * grid_step_size[1];

      // init the x stepping
      grid_pos[0] = cell_min_grid_pos[0] + x_left * grid_step_size[0];
      grid_pos[1] = cell_min_grid_pos[1] + yi * grid_step_size[1];
      grid_pos[2] = cell_min_grid_pos[2] + zi * grid_step_size[2];
      tot_interior_evals++;
      if (PtInCell(grid_pos, vblock, cell, normals)) {
	x_in_left = true;
	x_in_right = true;
      }
      else {
	x_in_left = false;
	x_in_right = false;
      }
      min_xi = cell_grid_pts[0] - 1;
      max_xi = 0;

      // x step left
      for (xi = x_left; xi >= 0 && xi < cell_grid_pts[0];) {

	grid_pos[0] = cell_min_grid_pos[0] + xi * grid_step_size[0];

	tot_interior_evals++;
	if (PtInCell(grid_pos, vblock, cell, normals)) {
	  if (x_in_left) { // remains interior, keep stepping
	    if (xi < min_xi)
	      min_xi = xi;
	    if (xi > max_xi)
	      max_xi = xi;
	    xi--;
	  }
	  else { // edge crossing from interior to exterior
	    if (xi < min_xi)
	      min_xi = xi;
	    if (xi > max_xi)
	      max_xi = xi;
	    x_left = xi;
	    break;
	  }
	} // pt is inside the cell

	// pt is outside the cell
	else {
	  if (!x_in_left) // remains exterior, keep stepping
	    xi++;
	  else { // edge crossing from exterior to interior
	    x_left = xi;
	    break;
	  }
	} // pt is outside the cell

      } // x step left

      // x step right
      for (xi = x_right; xi >= 0 && xi < cell_grid_pts[0];) {

	grid_pos[0] = cell_min_grid_pos[0] + xi * grid_step_size[0];

	tot_interior_evals++;
	if (PtInCell(grid_pos, vblock, cell, normals)) {
	  if (x_in_right) {// remains interior, keep stepping
	    if (xi < min_xi) 
	      min_xi = xi;
	    if (xi > max_xi)
	      max_xi = xi;
	    xi++;
	  } 
	  else { // edge crossing from interior to exterior
	    if (xi < min_xi)
	      min_xi = xi;
	    if (xi > max_xi)
	      max_xi = xi;
	    x_right = xi;
	    break;
	  }
	} // pt is inside the cell

	// pt is outside the cell
	else {
	  if (!x_in_right) // remains exterior, keep stepping
	    xi--;
	  else { // edge crossing from exterior to interior
	    x_right = xi;
	    break;
	  }
	} // pt is outside the cell

      } // x step right

      border[2 * (zi * cell_grid_pts[1] + yi)]     = min_xi;
      border[2 * (zi * cell_grid_pts[1] + yi) + 1] = max_xi;

      // min_xi > max_xi is the signal that no points were found

      // intersection found in this y-scan line
      if (min_xi <= max_xi) {
	tot_num_grid_pts += (max_xi - min_xi + 1);
	if (yi == y_start)
	  first_x = (min_xi + max_xi) / 2;
      }

      // note the first y line with an intersection in this z step
      int first_y; // y line with first border points
      first_y = y_start; // initial values change nothing unless reset below
      if (min_xi <= max_xi && !border_found) {
	first_y = yi;
	border_found = true;
      }

      if (min_xi > max_xi && border_found)
	z_step_done = true;

      // when this z step is done and if there are more z steps to do, 
      // check if y_start needs to be loosened for next z step
      if ((yi == cell_grid_pts[1] - 1 || z_step_done)
	  && zi + 1 < cell_grid_pts[2]) {

	grid_pos[2] = cell_min_grid_pos[2] + (zi + 1) * grid_step_size[2];
	grid_pos[0] = cell_min_grid_pos[0] + first_x * grid_step_size[0];

	for (yj = first_y; yj > 0; yj--) {
	  grid_pos[1] = cell_min_grid_pos[1] + yj * grid_step_size[1];
	  tot_interior_evals++;
	  if (!PtInCell(grid_pos, vblock, cell, normals))
	    break;
	}
	y_start = yj;

      } // loosen y start

      // terminate the yi loop (this z step) early if it is done
      // also initialize the borders of unused y-scan lines skipped at the end
      if (z_step_done) {
	for (yj = yi + 1; yj < cell_grid_pts[1]; yj++) { // after end
	  border[2 * (zi * cell_grid_pts[1] + yj)]     = 1; // min
	  border[2 * (zi * cell_grid_pts[1] + yj) + 1] = 0; // max
	}

	break; // y step

      }

    } // y step

  } // z step

  // deposit the mass in the interior of the cell
  num_grid_pts = 0;
  for (zi = 0; zi < cell_grid_pts[2]; zi++) { // z
    for (yi = 0; yi < cell_grid_pts[1]; yi++) { // y

      min_xi = border[2 * (zi * cell_grid_pts[1] + yi)];
      max_xi = border[2 * (zi * cell_grid_pts[1] + yi) + 1];

      for (xi = min_xi; xi <= max_xi; xi++) { // x

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
#endif
//--------------------------------------------------------------------------
//
// computes grid step parameters
//
// int num_given_bounds: 
//  0 = none, 1 = x bounds given, 2 = x,y bounds given, 3 = x,y,z bounds given
// given_mins, given_maxs: given bounds (x,y,z)
//
void GridStepParams(int num_given_bounds, 
		    float *given_mins, float *given_maxs) {

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
  if (num_given_bounds >= 1) {
    grid_phys_mins[0] = given_mins[0];
    grid_phys_maxs[0] = given_maxs[0];
  }
  if (num_given_bounds >= 2) {
    grid_phys_mins[1] = given_mins[1];
    grid_phys_maxs[1] = given_maxs[1];
  }
  if (num_given_bounds >= 3) {
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
//--------------------------------------------------------------------------
//
// iterate over cells and assigns density to grid points
//  to grid points within a window size of one grid space, ie, CIC for the 
//  8 grid points of a cell (vertex centered), 
//  equivalent to CIC for 8 neighboring cells (cell-centered)
//
//  Note that we are only using the site (original particle position) 
//   from the cell, ignoring rest of voronoi cell for CIC
//
// block: local block number
// block_min_idx: global grid idx of block minimum grid point (x,y,z)
// block_num_idx: number of grid points in block (x,y,z)
// density: density field
//
// side effects: writes density or sends to neighbors
//
void IterateCellsCIC(int block, int *block_min_idx, int *block_num_idx,
		     float **density) {

  float grid_pos[3]; // physical position of grid point

  // debug
  fprintf(stderr, "gid %d has %d cell sites (particles)\n", DIY_Gid(0, block),
	  vblocks[block]->num_orig_particles);

  // divisor for volume (3d density) or area (2d density)
  // assumes projection is to x-y plane
  float div = (project ? grid_step_size[0] * grid_step_size[1] :
	       grid_step_size[0] * grid_step_size[1] * grid_step_size[2]);

  // cells
  for (int cell = 0; cell < vblocks[block]->num_orig_particles; cell++) {

    // debug
    check_mass++;

    // distribute mass at cell site to neighboring grid points
    vector<int> grid_idxs; // grid idxs that get a fraction of the mass
    vector<float> grid_masses; // mass given to each grid_idx
    float *pt = &(vblocks[block]->sites[3 * cell]); // x,y,z of particle
    DistributeScalarCIC(pt, mass, grid_idxs, grid_masses);

    assert((int)grid_idxs.size() / 3 == 8); // sanity

    // (8) grid points for this cell site
    for (int i = 0; i < (int)grid_idxs.size() / 3; i++) {

      idx2phys(&(grid_idxs[3 * i]), grid_pos);

      // assign density to grid points in the block
      if (grid_pos[0] >= vblocks[block]->mins[0] &&
	  (grid_pos[0] < vblocks[block]->maxs[0]  ||
	   fabs(grid_pos[0] - data_maxs[0]) < eps) &&

	  grid_pos[1] >= vblocks[block]->mins[1] &&
	  (grid_pos[1] < vblocks[block]->maxs[1]  ||
	   fabs(grid_pos[1] - data_maxs[1]) < eps) &&

	  grid_pos[2] >= vblocks[block]->mins[2] &&
	  (grid_pos[2] < vblocks[block]->maxs[2]  ||
	   fabs(grid_pos[2] - data_maxs[2]) < eps) ) {

	// assign the density to the local block density array
	int block_grid_idx[3]; // local block idx of grid point
	Global2LocalIdx(&(grid_idxs[3 * i]), block_grid_idx, block_min_idx);
	int idx = index(block_grid_idx, block_num_idx);
	density[block][idx] += (grid_masses[i] / div);

	// debug
	tot_mass += grid_masses[i];
	if (density[block][idx] > max_dense)
	  max_dense = density[block][idx];

      }

      // or send grid points to neighboring blocks
      else {
	grid_pt_t grid_pt;
	grid_pt.idx[0] = grid_idxs[3 * i];
	grid_pt.idx[1] = grid_idxs[3 * i + 1];
	grid_pt.idx[2] = grid_idxs[3 * i + 2];
	grid_pt.mass = grid_masses[3 * i];
	DIY_Enqueue_item_points(0, block, (void *)&grid_pt, NULL,
				sizeof(grid_pt_t), grid_pos, 1, NULL);
      }

    } // (8) grid points for this cell site

  } // cells

}
//--------------------------------------------------------------------------
//
// distributes scalar value to grid points within a window size of one
//  grid space, ie, CIC for the 8 grid points of a cell (vertex centered), 
//  equivalent to CIC for 8 neighboring cells (cell-centered)
//
// pt: input point
// scalar: scalar value at input point
// win_min, win_max: physical extents of desired window (x,y,z)
// grid_idxs: global grid indices of grid points within window size of
//  input point (i,j,k,i,j,k,...) (output)
// grid_scalars: distributed scalars at each grid_idx (output)
//
// The implementation below is more complicated than plain CIC because
// it can handle larger window sizes one grid space,
// hence the computation of vol / v0 and eventually normalizing by
// tot_weight. This would be uneccessary in ordinary CIC, but the
// result is the same as CIC when the window is one grid space.
//
void DistributeScalarCIC(float *pt, float scalar,
		      vector <int> &grid_idxs, vector <float> &grid_scalars) {

  // global grid indices of window min and max grid points
  //
  // This is the only part simplified for CIC, min and max window points
  // are just one step apart. After this, the rest of the computation is
  // completely general for any window size.
  int min_win_idx[3];
  int max_win_idx[3];
  phys2idx(pt, min_win_idx);
  max_win_idx[0] = min_win_idx[0] + 1;
  max_win_idx[1] = min_win_idx[1] + 1;
  max_win_idx[2] = min_win_idx[2] + 1;

  float tot_weight = 0.0f; // total of weights in the window, should be 1.0

  // distribute fractional densities onto grid points in the window
  float v0 = 0.0; // volume of first box computed
  vector <float> weights; // weights accociated with grid points in the windonw
  int ijk[3]; // grid index
  for (ijk[2] = min_win_idx[2]; ijk[2] <= max_win_idx[2]; ijk[2]++) {
    for (ijk[1] = min_win_idx[1]; ijk[1] <= max_win_idx[1]; ijk[1]++) {
      for (ijk[0] = min_win_idx[0]; ijk[0] <= max_win_idx[0]; ijk[0]++) {

	grid_idxs.push_back(ijk[0]);
	grid_idxs.push_back(ijk[1]);
	grid_idxs.push_back(ijk[2]);

	// move point a little if it lies on a grid line
	float grid_pos[3]; // physical position of grid point
	float p[3] = {pt[0], pt[1], pt[2]}; // temp copy of pt
	idx2phys(ijk, grid_pos);
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
  for (int i = 0; i < (int)weights.size(); i++) {
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
//--------------------------------------------------------------------------
