//------------------------------------------------------------------------------
//
// driver for tessllating generic io hacc data in postprocessing
//
// Tom Peterka
// Argonne National Laboratory
// 9700 S. Cass Ave.
// Argonne, IL 60439
// tpeterka@mcs.anl.gov
//
// All rights reserved. May not be used, modified, or copied
// without permission
//
//--------------------------------------------------------------------------
#include "mpi.h"
#include <assert.h>
#include "tess.h"
#include <vector>
#include <stdio.h>
#include "GenericIODefinitions.hpp"
#include "GenericIOReader.h"
#include "GenericIOMPIReader.h"
#include "GenericIOPosixReader.h"
#include <math.h>

#define MAX_NUM_NEIGHBORS 27

using namespace std;

void GetArgs(int argc, char **argv, char *infile, char *outfile,
	     float *minvol, float *maxvol, int *wrap);
void ReadGIO(gio::GenericIOReader *reader, int rank, int groupsize,
	     int* &gids, float** &particles, int* &num_particles, bb_t* &bb, 
	     int&tot_blocks, int& nblocks, float *data_mins, float *data_maxs,
	     int *block_dims);
void GetNeighbors(gio::GenericIOReader *reader, bool wrap, int nblocks, 
		  int *gids, int *block_dims,
		  gb_t** &neighbors, int* &num_neighbors, int rank,
		  int grooupsize);
void ijk2gid(int *ijk, int& gid, int *block_dims, bool wrap);

// DEPRECATED
// void gid2ijk(int gid, int *ijk, int *block_dims);

int main(int argc, char *argv[]) {

  int tot_blocks; // total number of blocks in the domain
  char infile[256]; // input file name
  char outfile[256]; // output file name
  float minvol, maxvol; // volume range, -1.0 = unused
  double times[MAX_TIMES]; // timing
  int nblocks; // my local number of blocks
  int num_threads = 1; // number of threads diy can use
  int dim = 3; // 3d always
  int wrap; // whether wraparound neighbors are used
  int rank, groupsize; // MPI usual

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &groupsize);

  GetArgs(argc, argv, infile, outfile, &minvol, &maxvol, &wrap);

  // following are allocated by ReagGIO
  int *gids; // local block gids
  bb_t *bb; // block bounds
  gb_t **neighbors; // block neighbors
  int *num_neighbors; // number of neighbors in each block
  float **particles; // particles[block_num][particle] 
		     //  where each particle is 3 values, px, py, pz
  int *num_particles; // number of particles in each block
  float data_mins[3]; // physical global data minimum
  float data_maxs[3]; // physical global data maximum
  int block_dims[3]; // global number of blocks in each dimension

  // intialize reader
  gio::GenericIOReader *reader = new gio::GenericIOMPIReader();
  reader->SetFileName(infile);
  reader->SetCommunicator(MPI_COMM_WORLD);
  reader->OpenAndReadHeader();

  // read generic I/O data
  ReadGIO(reader, rank, groupsize, gids, particles, 
	  num_particles, bb, tot_blocks, nblocks, data_mins, data_maxs,
	  block_dims);

  // debug
  if (rank == 0)
    fprintf(stderr, "data_mins[%.1f %.1f %.1f] data_maxs[%.1f %.1f %.1f] "
	    "block_dims[%d %d %d]\n",
	    data_mins[0], data_mins[1], data_mins[2], 
	    data_maxs[0], data_maxs[1], data_maxs[2], 
	    block_dims[0], block_dims[1], block_dims[2]);

  // find neighboring blocks
  GetNeighbors(reader, wrap, nblocks, gids, block_dims, neighbors, 
	       num_neighbors, rank, groupsize);

  // initialize, run, cleanup tess
  tess_init(nblocks, gids, bb, neighbors, num_neighbors, data_mins, data_maxs, 
	    wrap, 0, minvol, maxvol, MPI_COMM_WORLD, times);
  tess(particles, num_particles, outfile);
  tess_finalize();

  // cleanup
  for (int b = 0; b < nblocks; b++) {
    free(particles[b]);
    delete[] neighbors[b];
  }
  free(particles);
  free(num_particles);
  delete[] gids;
  delete[] bb;
  delete[] num_neighbors;
  delete[] neighbors;
  reader->Close();
  delete reader;

  MPI_Finalize();

  return 0;

}
//----------------------------------------------------------------------------
//
// gets command line args
//
void GetArgs(int argc, char **argv, char *infile, char *outfile,
	     float *minvol, float *maxvol, int *wrap) {

  assert(argc >= 6);

  strcpy(infile, argv[1]);
  strcpy(outfile, argv[2]);
  *minvol = atof(argv[3]);
  *maxvol = atof(argv[4]);
  *wrap = atoi(argv[5]);

}
//----------------------------------------------------------------------------
//
// reads generic I/O data in one block
//
// reader: opened instance of generic io reader
// rank, groupsize: MPI usual
// gids: local block gids (output) allocated by this function
// particles: particles for my local blocks (output) allocated by this function
// num_particles: number of particles in local blocks (output) allocated by this
//   function
// bb: block bounds of local blocks (output) allocated by this function
// tot_blocks: total number of blocks (output)
// nblocks: local number of blocks (output)
// data_mins: physical global data minimum (output)
// data_maxs: physical global data maximum (output)
// block_dims: global number of blocks in each cimension (output)
//
void ReadGIO(gio::GenericIOReader *reader, int rank, int groupsize,
	     int* &gids, float** &particles, int* &num_particles, bb_t* &bb, 
	     int& tot_blocks, int& nblocks, float *data_mins, float *data_maxs,
	     int *block_dims) {

  double min[3], max[3]; // local block bounds

  // total number of blocks and local blocks
  tot_blocks = reader->GetTotalNumberOfBlocks();
  int max_blocks = ceilf((float)tot_blocks / groupsize); // max in any process
  gids = new int[max_blocks];
  nblocks = 0;
  for (int b = 0; b < tot_blocks; ++b) {
    if (b % groupsize == rank) // round robin assignment
      gids[nblocks++] = b;
  }

  // global data
  double origin[3], scale[3]; // global min and max as doubles
  uint64_t dims[3]; // global block dims as unint64
  reader->GetPhysOrigin(origin);
  reader->GetPhysScale(scale);
  reader->GetGlobalDimensions(dims);
  data_mins[0] = origin[0];
  data_mins[1] = origin[1];
  data_mins[2] = origin[2];
  data_maxs[0] = scale[0];
  data_maxs[1] = scale[1];
  data_maxs[2] = scale[2];
  block_dims[0] = dims[0];
  block_dims[1] = dims[1];
  block_dims[2] = dims[2];

  // allocate particle arrays
  // malloc used for particles and num_particles becuase tess uses realloc
  bb = new bb_t[nblocks];
  num_particles = (int *)malloc(nblocks * sizeof(int));
  particles = (float **)malloc(nblocks * sizeof(float *));

  // read local blocks
  for (int b = 0; b < nblocks; b++) {

    // clear reader variables
    reader->ClearVariables();

    //  block bounds
    // note the reader wants lid, not gid
    reader->GetBlockBounds(b, min, max);
    bb[b].min[0] = min[0];
    bb[b].min[1] = min[1];
    bb[b].min[2] = min[2];
    bb[b].max[0] = max[0];
    bb[b].max[1] = max[1];
    bb[b].max[2] = max[2];
    
    // number of particles in this block
    // note the reader wants gid, not lid
    num_particles[b] = reader->GetNumberOfElements(gids[b]);

    // debug
//     fprintf(stderr, "gid %d num_particles = %d min [%.1f %.1f %.1f] "
// 	    "max [%.1f %.1f %.1f]\n",
// 	    gids[b], num_particles[b], 
// 	    bb[b].min[0], bb[b].min[1], bb[b].min[2],
// 	    bb[b].max[0], bb[b].max[1], bb[b].max[2]);

    // padsize CRC for floats
    int floatpadsize = gio::CRCSize / sizeof(float);

    // allocate application arrays to store variables plus CRCs
    float* x = new float[num_particles[b] + floatpadsize]; 
    float* y = new float[num_particles[b] + floatpadsize]; 
    float* z = new float[num_particles[b] + floatpadsize];

    // clear variables and then register application arrays with the reader
    reader->AddVariable("x", x, gio::GenericIOBase::ValueHasExtraSpace); 
    reader->AddVariable("y", y, gio::GenericIOBase::ValueHasExtraSpace); 
    reader->AddVariable("z", z, gio::GenericIOBase::ValueHasExtraSpace);

    // read the particles
    // note the reader wants gid, not lid
    reader->ReadBlock(gids[b]);

    // package particles
    particles[b] = (float *)malloc(num_particles[b] * 3 * sizeof(float));
    for (int i = 0; i < num_particles[b]; i++) {
      particles[b][3 * i]     = x[i];
      particles[b][3 * i + 1] = y[i];
      particles[b][3 * i + 2] = z[i];
    }

    // cleanup temporary points
    delete[] x;
    delete[] y;
    delete[] z;

  }

}
//----------------------------------------------------------------------------
//
// gets neighbors of local blocks
//
// reader: open reader instance
// wrap: whether wrapping is used
// gids: local block gids
// block_dims: number of blocks in each dimension
// neighbors: neighbors of local blocks (output) alocated by thie function
// num_neihghbors: number of neighbors of local blocks (output) allocated by 
//   this function
// rank, groupsize: MPI usual
//
void GetNeighbors(gio::GenericIOReader *reader, bool wrap, int nblocks, 
		  int *gids, int *block_dims,
		  gb_t** &neighbors, int* &num_neighbors, int rank,
		  int groupsize) {

  int neigh_gids[MAX_NUM_NEIGHBORS]; // gids of neighbor blocks
  int ijk[3]; // block coords

  neighbors = new gb_t*[nblocks];
  num_neighbors = new int[nblocks];

  // for all local blocks
  for (int b = 0; b < nblocks; b++) {

    neighbors[b] = new gb_t[MAX_NUM_NEIGHBORS];
    num_neighbors[b] = 0;

    gio::RankHeader block_info = reader->GetBlockHeader(b);
    ijk[0] = block_info.Coords[0];
    ijk[1] = block_info.Coords[1];
    ijk[2] = block_info.Coords[2];

    // DEPRECATED
//     gid2ijk2(gids[b], ijk, block_dims);

    // debug
//     fprintf(stderr, "gid %d has ijk [%d %d %d]\n", 
// 	    gids[b], ijk[0], ijk[1], ijk[2]);

    // for all neighbor directions
    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
	for (int k = -1; k <= 1; k++) {

	  int neigh_ijk[3];
	  int neigh_gid;
	  unsigned char neigh_dir;
	  int neigh_proc;
	  neigh_ijk[0] = ijk[0] + i;
	  neigh_ijk[1] = ijk[1] + j;
	  neigh_ijk[2] = ijk[2] + k;
	  ijk2gid(neigh_ijk, neigh_gid, block_dims, wrap);

	  if (neigh_gid >= 0) {
	    neigh_dir = 0x00;
	    if (i == -1)
	      neigh_dir |= 0x01;
	    if (i == 1)
	      neigh_dir |= 0x02;
	    if (j == -1)
	      neigh_dir |= 0x04;
	    if (j == 1)
	      neigh_dir |= 0x08;
	    if (k == -1)
	      neigh_dir |= 0x10;
	    if (k == 1)
	      neigh_dir |= 0x20;
	    neigh_proc = neigh_gid % groupsize;

	    // store the neighbor
	    neighbors[b][num_neighbors[b]].gid = neigh_gid;
	    neighbors[b][num_neighbors[b]].proc = neigh_proc;
	    neighbors[b][num_neighbors[b]].neigh_dir = neigh_dir;

	    // debug
// 	    fprintf(stderr, "gid %d neig %d ijk [%d %d %d] has "
// 		    "(gid %d, proc %d, dir %0x)\n",
// 		    gids[b], num_neighbors[b], neigh_ijk[0], neigh_ijk[1], 
// 		    neigh_ijk[2], neigh_gid, neigh_proc, neigh_dir);

	    num_neighbors[b]++;

	  } // neigh_gid >= 0
	} // k
      } // j
    } // k

  } // lcoal blocks

}
//----------------------------------------------------------------------------
//
// convert i,j,k block coordinates to gid
// for column-major order, k changes fastest and i changes slowest
// this is the MPI standard MPI_Cart_create order, which HACC uses when
// it runs, and genericio reads back
//
// ijk: (i, j, k) block coords
// gid: output gid
// block_dims: number of blocks in each dimension
// whether wrapping is on
//
void ijk2gid(int *ijk, int& gid, int *block_dims, bool wrap) {

  int i = ijk[0];
  int j = ijk[1];
  int k = ijk[2];

  // i,j,k out of bounds
  if (!wrap) {
    if (i < 0 || j < 0 || k < 0 ||
	i >= block_dims[0] || j >= block_dims[1] || k >= block_dims[2]) {
      gid = -1;
      return;
    }
  }

  // wrap i,j,k
  if (wrap) {
    i = (i < 0 ? block_dims[0] - 1 : i);
    j = (j < 0 ? block_dims[1] - 1 : j);
    k = (k < 0 ? block_dims[2] - 1 : k);
    i = (i >= block_dims[0] ? 0 : i);
    j = (j >= block_dims[1] ? 0 : j);
    k = (k >= block_dims[2] ? 0 : k);
  }

  // i fastest, k slowest order
  // in case this order is ever needed (not used for generic io)
//   gid = k * block_dims[1] * block_dims[0] + j * block_dims[0] + i;

  // k fastest, i slowest order
  gid = i * block_dims[1] * block_dims[2] + j * block_dims[2] + k;

}
//----------------------------------------------------------------------------
// // DEPRECATED, replaced by a function in the generic io reader
// //
// // convert gid to i,j,k block coordinates
// // for column-major order, k changes fastest and i changes slowest
// // gid: block gid
// // ijk: output (i, j, k) block coords
// // block_dims: number of blocks in each dimension
// //
// void gid2ijk(int gid, int *ijk, int *block_dims) {

//   // i fastest, k slowest
// //   i = gid % block_dims[0];
// //   j = (gid / block_dims[0]) % block_dims[1];
// //   k = gid / (block_dims[1] * block_dims[0]);

//   // k fastest, i slowest
//   int k = gid % block_dims[2];
//   int j = (gid / block_dims[2]) % block_dims[1];
//   int i = gid / (block_dims[1] * block_dims[2]);

//   ijk[0] = i;
//   ijk[1] = j;
//   ijk[2] = k;

// }
//----------------------------------------------------------------------------
