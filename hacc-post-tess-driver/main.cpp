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
#include <set>
#include <stdio.h>
#include "GenericIODefinitions.hpp"
#include "GenericIOReader.h"
#include "GenericIOMPIReader.h"
#include "GenericIOPosixReader.h"
#include <math.h>

#define MAX_NUM_NEIGHBORS 27

using namespace std;

void GetArgs(int argc, char **argv, char *infile, char *outfile,
	     float *minvol, float *maxvol, int *wrap, int *bf, 
	     int *sample_rate);
void ReadGIO(gio::GenericIOReader *reader, int rank, int groupsize,
	     int* &gids, float** &particles, int* &num_particles, bb_t* &bb, 
	     int&tot_blocks, int& nblocks, float *data_mins, float *data_maxs,
	     int *block_dims, int sample_rate);
void Redistribute(int *bf, int* &gids, 
		  float** &particles, int* &num_particles, bb_t* &bb, 
		  int& tot_blocks, int& nblocks, MPI_Comm comm);
void Pt2Child(float *pt, int *bf, float *mins, float *maxs, int *idx);
void GetNeighbors(int *gids, bb_t *bb, bool wrap, int nblocks, int *block_dims,
		  gb_t** &neighbors, int* &num_neighbors, float *data_mins,
		  float *data_maxs, int *bf, int *old_block_dims);
void ijk2gid(int *ijk, int& gid, int *block_dims, bool wrap, int *bf,
	     int *old_block_dims);
void gid2ijk(int gid, int *ijk, int *bf, int *old_block_dims);

// DEPRECATED
// void bb2ijk(bb_t bb, int *ijk, int *block_dims, 
// 	    float *data_mins, float *data_maxs);

int main(int argc, char *argv[]) {

  int tot_blocks; // total number of blocks in the domain
  char infile[256]; // input file name
  char outfile[256]; // output file name
  float minvol, maxvol; // volume range, -1.0 = unused
  double times[TESS_MAX_TIMES]; // timing
  int nblocks; // my local number of blocks
  int num_threads = 1; // number of threads diy can use
  int dim = 3; // 3d always
  int wrap; // whether wraparound neighbors are used
  int rank, groupsize; // MPI usual
  int bf[3]; // redistribution blocking factor
  int sample_rate; // sample rate

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &groupsize);

  GetArgs(argc, argv, infile, outfile, &minvol, &maxvol, &wrap, bf, 
	  &sample_rate);

  // following are allocated by ReadGIO
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
  ReadGIO(reader, rank, groupsize, gids, particles, num_particles, bb, 
	  tot_blocks, nblocks, data_mins, data_maxs, block_dims,
	  sample_rate);

  // check that bf is valid and that bf, tot_blocks, and groupsize agree
  assert(bf[0] > 0 && bf[1] > 0 && bf[2] > 0); // 0's not allowed
  assert(tot_blocks * bf[0] * bf[1] * bf[2] >= groupsize);

  // debug
  int num_pts = num_particles[0];
  int max_pts;
  int min_pts;
  // fprintf(stderr, "rank = %d num_pts = %d\n", rank, num_pts);
  MPI_Reduce(&num_pts, &max_pts, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&num_pts, &min_pts, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
  if (rank == 0)
    fprintf(stderr, "num pts before redistribution = [%d, %d]\n", min_pts,
	    max_pts);

  // redistribute particles
  if (bf[0] > 1 || bf[1] > 1 || bf[2] > 1)
    Redistribute(bf, gids, particles, num_particles, 
		 bb, tot_blocks, nblocks, MPI_COMM_WORLD);

  // adjust block dims for redistribution and save a copy of the original
  int old_block_dims[3];
  old_block_dims[0] = block_dims[0];
  old_block_dims[1] = block_dims[1];
  old_block_dims[2] = block_dims[2];
  block_dims[0] *= bf[0];
  block_dims[1] *= bf[1];
  block_dims[2] *= bf[2];

  // debug
  if (rank == 0)
    fprintf(stderr, "data_mins[%.1f %.1f %.1f] data_maxs[%.1f %.1f %.1f] "
	    "block_dims[%d %d %d]\n",
	    data_mins[0], data_mins[1], data_mins[2], 
	    data_maxs[0], data_maxs[1], data_maxs[2], 
	    block_dims[0], block_dims[1], block_dims[2]);


  // find neighboring blocks
  GetNeighbors(gids, bb, wrap, nblocks, block_dims, neighbors, num_neighbors, 
	       data_mins, data_maxs, bf, old_block_dims);

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
	     float *minvol, float *maxvol, int *wrap, int *bf, 
	     int *sample_rate) {

  assert(argc >= 10);

  strcpy(infile, argv[1]);
  if (argv[2][0] == '!')
    strcpy(outfile, "");
  else
    strcpy(outfile, argv[2]);
  *minvol = atof(argv[3]);
  *maxvol = atof(argv[4]);
  *wrap = atoi(argv[5]);
  bf[0] = atoi(argv[6]);
  bf[1] = atoi(argv[7]);
  bf[2] = atoi(argv[8]);
  *sample_rate = atoi(argv[9]);

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
// sample_rate: 1 out of every sample_rate particles will be kept
//
void ReadGIO(gio::GenericIOReader *reader, int rank, int groupsize,
	     int* &gids, float** &particles, int* &num_particles, bb_t* &bb, 
	     int& tot_blocks, int& nblocks, float *data_mins, float *data_maxs,
	     int *block_dims, int sample_rate) {

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
    int idpadsize = gio::CRCSize / sizeof(int64_t);

    // allocate application arrays to store variables plus CRCs
    float* x = new float[num_particles[b] + floatpadsize]; 
    float* y = new float[num_particles[b] + floatpadsize]; 
    float* z = new float[num_particles[b] + floatpadsize];
    int64_t *id = new int64_t[num_particles[b] + idpadsize];

    // clear variables and then register application arrays with the reader
    reader->AddVariable("x", x, gio::GenericIOBase::ValueHasExtraSpace); 
    reader->AddVariable("y", y, gio::GenericIOBase::ValueHasExtraSpace); 
    reader->AddVariable("z", z, gio::GenericIOBase::ValueHasExtraSpace);
    reader->AddVariable("id", id, gio::GenericIOBase::ValueHasExtraSpace);

    // read the particles
    // note the reader wants gid, not lid
    reader->ReadBlock(gids[b]);

    // unique_ids is used to weed out duplicate particles, which sometimes
    // can happen in hacc
    set <int64_t> unique_ids;

    // package particles, sampling as specified and filtering out duplicates
    num_particles[b] /= sample_rate;
    particles[b] = (float *)malloc(num_particles[b] * 3 * sizeof(float));
    int nu = 0; // number of unique points
    for (int i = 0; i < num_particles[b]; i++) {
      if (unique_ids.find(id[i * sample_rate]) == unique_ids.end()) {
	particles[b][3 * nu]     = x[i * sample_rate];
	particles[b][3 * nu + 1] = y[i * sample_rate];
	particles[b][3 * nu + 2] = z[i * sample_rate];
	unique_ids.insert(id[i * sample_rate]);
	nu++;
      }
    }

    // debug
    if (nu < num_particles[b])
      fprintf(stderr, "%d duplicate particles found and removed in rank %d\n",
	      num_particles[b] - nu, rank);

    num_particles[b] = nu;

    // cleanup temporary points
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] id;

  }

}
//----------------------------------------------------------------------------
//
// redistributes particles from one block to 2, 4, or 8 blocks
//
// bf: blocking factor (number of children, including parent) in each
//  dimension, eg (2, 2, 2)
// gids: local block gids (input and output) allocated by this function for new
//  blocks
// particles: particles for my local blocks (output) allocated by this function
//  for new blocks
// num_particles: number of particles in local blocks (output) allocated by this
//   function for new blocks
// bb: block bounds of local blocks (output) allocated by this function
//   for new blocks
// tot_blocks: total number of blocks (input and output)
// nblocks: local number of blocks (input and output)
// MPI communicator
//
void Redistribute(int *bf, int* &gids, 
		  float** &particles, int* &num_particles, bb_t* &bb, 
		  int& tot_blocks, int& nblocks, MPI_Comm comm) {

  int rank, groupsize; // MPI usual
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &groupsize);

  // check bf
  if (bf[0] < 2 && bf[1] < 2 && bf[2] < 2) // nothing to do
    return;
  assert(bf[0] > 0 && bf[1] > 0 && bf[2] > 0); // sanity

  // since we are redistributing, assume that there are more procs
  // than original blocks, and the final blocks are 1 block per proc
  // ie, gids = ranks
  assert(tot_blocks * bf[0] * bf[1] * bf[2] == groupsize);
  assert(nblocks <= 1); // sanity

  // parents (nblocks = 1) sort particles and send to children
  if (nblocks) {

    int b = 0; // local block id

    // particles going to child blocks
    vector<float> *ps = new vector<float>[bf[0] * bf[1] * bf[2]];

    // for all particles
    for (int i = 0; i < num_particles[b]; i++) {
    
      // determine child block for the particle
      int cid[3]; // child index (i,j,k) 0 to bf -1 in each dimension
      Pt2Child(&particles[b][3 * i], bf, bb[b].min, bb[b].max, cid);

      // debug
      //       fprintf(stderr, "pt [%.3f %.3f %.3f] in bb min [%.1f %.1f %f] "
      // 	      "max [%.1f %.1f %.1f] belongs in cid [%d %d %d]\n",
      // 	      particles[b][3 * i], particles[b][3 * i + 1], 
      // 	      particles[b][3 * i + 2], 
      // 	      bb[b].min[0], bb[b].min[1], bb[b].min[2], 
      // 	      bb[b].max[0], bb[b].max[1], bb[b].max[2], 
      // 	      cid[0], cid[1], cid[2]);

      // child is column-major ordering (z fastest) of cid
      int child = cid[2] + cid[1] * bf[2] + cid[0] * (bf[2] * bf[1]);

      // sort particle into proper vector
      ps[child].push_back(particles[b][3 * i]);
      ps[child].push_back(particles[b][3 * i + 1]);
      ps[child].push_back(particles[b][3 * i + 2]);

    } // for all particles

    // parent appends child block bounds to particle arrays for 
    // any children receiving particles
    for (int i = 0; i < bf[0]; i++) {
      for (int j = 0; j < bf[1]; j++) {
	for (int k = 0; k < bf[2]; k++) {
	  int child = k + j * bf[2] + i * (bf[2] * bf[1]);
	  // there are some particles to send and i,j,k is not parent
	  if (ps[child].size() && (i || j || k)) {
	    // get child bounds
	    float child_min[3], child_max[3];
	    child_min[0] = bb[b].min[0] + 
	      i * (bb[b].max[0] - bb[b].min[0]) / bf[0];
	    child_min[1] = bb[b].min[1] + 
	      j * (bb[b].max[1] - bb[b].min[1]) / bf[1];
	    child_min[2] = bb[b].min[2] + 
	      k * (bb[b].max[2] - bb[b].min[2]) / bf[2];
	    child_max[0] = child_min[0] + 
	      (bb[b].max[0] - bb[b].min[0]) / bf[0];
	    child_max[1] = child_min[1] + 
	      (bb[b].max[1] - bb[b].min[1]) / bf[1];
	    child_max[2] = child_min[2] + 
	      (bb[b].max[2] - bb[b].min[2]) / bf[2];
	    ps[child].push_back(child_min[0]);
	    ps[child].push_back(child_min[1]);
	    ps[child].push_back(child_min[2]);
	    ps[child].push_back(child_max[0]);
	    ps[child].push_back(child_max[1]);
	    ps[child].push_back(child_max[2]);

	    // debug
// 	    fprintf(stderr, "&&& gid %d sending bounds min [%.1f %.1f %.1f] "
// 		    "max [%.1f %.1f %.1f] to child %d\n",
// 		    gids[b],
// 		    child_min[0], child_min[1], child_min[2],
// 		    child_max[0], child_max[1], child_max[2],
// 		    child);

	  }
	}
      }
    }


    // parent sends particles to all children
    // start at j = 1 to skip sending to self
    for (int j = 1; j < bf[0] * bf[1] * bf[2]; j++) {

      // destination proc
      int dest_proc = tot_blocks - 1 + rank * (bf[0] * bf[1] * bf[2] - 1) + j;
      // send counts
      int num_floats = (int)(ps[j].size());
      MPI_Send(&num_floats, 1, MPI_INT, dest_proc, 0, comm);
      // send particles
      if (num_floats)
	MPI_Send(&ps[j][0], num_floats, MPI_FLOAT, dest_proc, 0, comm);

      // debug
//       fprintf(stderr, "rank %d sending %d particles to rank %d\n",
// 	      rank, (num_floats - 6) / 3, dest_proc);

    } // for all children

    // shrink particles to the remaining ones, ie those in child 0
    particles[b] = (float *)realloc(particles[b], 
				    ps[0].size() * sizeof(float));
    for (int j = 0; j < (int)ps[0].size(); j++)
      particles[b][j] = ps[0][j];
    num_particles[b] = (int)(ps[0].size() / 3);

    // adjust bounds for the new smaller block (only max needs adjustment)
    bb[b].max[0] = bb[b].min[0] + (bb[b].max[0] - bb[b].min[0]) / bf[0];
    bb[b].max[1] = bb[b].min[1] + (bb[b].max[1] - bb[b].min[1]) / bf[1];
    bb[b].max[2] = bb[b].min[2] + (bb[b].max[2] - bb[b].min[2]) / bf[2];

    // debug
//     fprintf(stderr, "+++ gid %d num_particles = %d min [%.1f %.1f %.1f] "
// 	    "max [%.1f %.1f %.1f]\n",
// 	    gids[b], num_particles[b], 
// 	    bb[b].min[0], bb[b].min[1], bb[b].min[2],
// 	    bb[b].max[0], bb[b].max[1], bb[b].max[2]);

    // sanity check: each particle must be inside the bounds
    for (int k = 0; k < num_particles[b]; k++) {
      assert(particles[b][3 * k] >= bb[b].min[0] &&
	     particles[b][3 * k] <= bb[b].max[0]);
      assert(particles[b][3 * k + 1] >= bb[b].min[1] &&
	     particles[b][3 * k + 1] <= bb[b].max[1]);
      assert(particles[b][3 * k + 2] >= bb[b].min[2] &&
	     particles[b][3 * k + 2] <= bb[b].max[2]);
    }

    delete[] ps;

  } // parents

  // children receive particles
  else {

    MPI_Status status;
    // source proc
    int src_proc = (rank - tot_blocks) / (bf[0] * bf[1] * bf[2] - 1);
    // receive counts
    int num_floats;
    MPI_Recv(&num_floats, 1, MPI_INT, src_proc, 0, comm, &status);

    // children each have a block now, even if it is empty
    nblocks = 1;
    int b = 0;
    bb = new bb_t[nblocks];
    num_particles = (int *)malloc(nblocks * sizeof(int));
    particles = (float **)malloc(nblocks * sizeof(float *));
    gids = new int[nblocks];
    gids[b] = rank;

    if (num_floats) {

      // floats include 6 extents that are received but not counted
      // in num_particles
      num_particles[b] = (num_floats - 6) / 3;
      // particles allocated large enough for extents too
      particles[b] = (float *)malloc(num_floats * sizeof(float));

      // receive particles and extents into particles array
      MPI_Recv(particles[b], num_floats, MPI_FLOAT, src_proc, 0, comm, &status);

      // copy received extents into block bounds
      bb[b].min[0] = particles[b][num_particles[b] * 3];
      bb[b].min[1] = particles[b][num_particles[b] * 3 + 1];
      bb[b].min[2] = particles[b][num_particles[b] * 3 + 2];
      bb[b].max[0] = particles[b][num_particles[b] * 3 + 3];
      bb[b].max[1] = particles[b][num_particles[b] * 3 + 4];
      bb[b].max[2] = particles[b][num_particles[b] * 3 + 5];

      // shrink the particle array back to just the particles
      particles[b] = (float *)realloc(particles[b], 
				      num_particles[b] * 3 * sizeof(float));

    }

    else
      num_particles[b] = 0;

    // debug
//     fprintf(stderr, "--- gid %d num_particles %d"
// 	    "min [%.1f %.1f %.1f] max [%.1f %.1f %.1f]\n",
// 	    gids[b], num_particles[b],
// 	    bb[b].min[0], bb[b].min[1], bb[b].min[2],
// 	    bb[b].max[0], bb[b].max[1], bb[b].max[2]);

    // sanity check: each particle must be inside the bounds
    for (int k = 0; k < num_particles[b]; k++) {
      assert(particles[b][3 * k] >= bb[b].min[0] &&
	     particles[b][3 * k] <= bb[b].max[0]);
      assert(particles[b][3 * k + 1] >= bb[b].min[1] &&
	     particles[b][3 * k + 1] <= bb[b].max[1]);
      assert(particles[b][3 * k + 2] >= bb[b].min[2] &&
	     particles[b][3 * k + 2] <= bb[b].max[2]);
    }

  } // children

  // update tot_blocks
  tot_blocks += bf[0] * bf[1] * bf[2];

}
//----------------------------------------------------------------------------
//
// gets neighbors of local blocks
//
// gids: gids for local blocks
// bb: block bounds for local blocks
// wrap: whether wrapping is used
// block_dims: number of blocks in each dimension
// neighbors: neighbors of local blocks (output) alocated by thie function
// num_neihghbors: number of neighbors of local blocks (output) allocated by 
//   this function
// data_mins, data_maxs: global data extents
// bf: blocking factor of new distribution
// old_block_dims: number of blocks in each dimension in old original
//  distribution
//
void GetNeighbors(int *gids, bb_t *bb, bool wrap, int nblocks, int *block_dims,
		  gb_t** &neighbors, int* &num_neighbors, float *data_mins,
		  float *data_maxs, int *bf, int *old_block_dims) {

  int neigh_gids[MAX_NUM_NEIGHBORS]; // gids of neighbor blocks
  int ijk[3]; // block coords

  neighbors = new gb_t*[nblocks];
  num_neighbors = new int[nblocks];

  // following assumes 1 block per process, gid = MPI rank
  assert(nblocks == 1);

  // for all local blocks
  for (int b = 0; b < nblocks; b++) {

    neighbors[b] = new gb_t[MAX_NUM_NEIGHBORS];
    num_neighbors[b] = 0;

    // DEPRECATED
//     bb2ijk(bb[b], ijk, block_dims, data_mins, data_maxs);

    gid2ijk(gids[b], ijk, bf, old_block_dims);

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

	  ijk2gid(neigh_ijk, neigh_gid, block_dims, wrap, bf, old_block_dims);

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
	    neigh_proc = neigh_gid; // assumes 1 block per process

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
// wrap: whether wrapping is on
// bf: blocking factor of new distribution
// old_block_dims: number of blocks in each dimension in old original
//  distribution
//
void ijk2gid(int *ijk, int& gid, int *block_dims, bool wrap, int *bf,
	     int *old_block_dims) {

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

  // i,j,k of child among all the children of he same parent
  int ci, cj, ck;
  ci = i % bf[0];
  cj = j % bf[1];
  ck = k % bf[2];

  // i,j,k of parent
  int pi, pj, pk;
  pi = i / bf[0];
  pj = j / bf[1];
  pk = k / bf[2];

  // gid of parent
  int p_gid = pi * old_block_dims[1] * old_block_dims[2] + 
    pj * old_block_dims[2] + pk;
  
  // debug
//   if (i == 0 && j == 0 && k == 1)
//     fprintf(stderr, "ci,cj,ck = [%d %d %d] pi,pj,pk = [%d %d %d] "
// 	    "p_gid = %d c_gid = %d\n", 
// 	    ci, cj, ck, pi, pj, pk, p_gid, c_gid);

  // convert ci, cj, ck to a linearly ordered (k fastest) child gid
  // (within the children of same parent)
  int c_gid = ci * bf[1] * bf[2] + cj * bf[2] + ck;

  // child is the parent
  if (c_gid == 0)
    gid = p_gid;

  // child is offset from the parent
  else
    gid = old_block_dims[0] * old_block_dims[1] * old_block_dims[2] - 1 +
      p_gid * (bf[0] * bf[1] * bf[2] - 1) + c_gid;

}
//----------------------------------------------------------------------------
//
// convert gid to i,j,k block coordinates
// based on column-major order, k changes fastest and i changes slowest
// this is the MPI standard MPI_Cart_create order, which HACC uses when
// it runs, and genericio reads back
//
// gid: input gid
// ijk: (i, j, k) output block coords
// bf: blocking factor of new distribution
// old_block_dims: number of blocks in each dimension in old original
//  distribution
//
void gid2ijk(int gid, int *ijk, int *bf, int *old_block_dims) {

  // gid is one of the original parent blocks
  if (gid < old_block_dims[0] * old_block_dims[1] * old_block_dims[2]) {

    // k fastest order
    ijk[2] = gid % old_block_dims[2];
    ijk[1] = (gid / old_block_dims[2]) % old_block_dims[1];
    ijk[0] = gid / (old_block_dims[1] * old_block_dims[2]);

    ijk[0] *= bf[0];
    ijk[1] *= bf[1];
    ijk[2] *= bf[2];

    return;

  }

  // gid is one of the child blocks
  else {

    // gid of parent
    int p_gid = 
      (gid - old_block_dims[0] * old_block_dims[1] * old_block_dims[2]) / 
      (bf[0] * bf[1] * bf[2] - 1);

    // i,j,k of parent
    int pi, pj, pk;
    pk = p_gid % old_block_dims[2];
    pj = (p_gid / old_block_dims[2]) % old_block_dims[1];
    pi = p_gid / (old_block_dims[1] * old_block_dims[2]);
    pi *= bf[0];
    pj *= bf[1];
    pk *= bf[2];

    // gid of child within its family
    int c_gid = 
      gid - p_gid * (bf[0] * bf[1] * bf[2] - 1) -
      old_block_dims[0] * old_block_dims[1] * old_block_dims[2] + 1;

    // i,j,k of child within its family
    int ci, cj, ck;
    ck = c_gid % bf[2];
    cj = (c_gid / bf[2]) % bf[1];
    ci = c_gid / (bf[1] * bf[2]);

    // final i,j,k is parent i,i,k plus offset in the family
    ijk[0] = pi + ci;
    ijk[1] = pj + cj;
    ijk[2] = pk + ck;

    // debug
//     fprintf(stderr, "gid = %d p_gid = %d p_ijk = [%d %d %d] c_gid = %d "
// 	    "c_ijk = [%d %d %d]\n", gid, p_gid, pi, pj, pk, c_gid, ci, cj, ck);

  }

}
//----------------------------------------------------------------------------
// DEPRECATED
// //
// // convert block bounds to i,j,k block coordinates
// //
// // bb: block bounds
// // ijk: output (i, j, k) block coords
// // block_dims: number of blocks in each dimension
// // data_mins, data_maxs: global data extents
// //
// void bb2ijk(bb_t bb, int *ijk, int *block_dims, 
// 	     float *data_mins, float *data_maxs) {

//   float block_size[3]; // physical block size
//   block_size[0] = (data_maxs[0] - data_mins[0]) / block_dims[0];
//   block_size[1] = (data_maxs[1] - data_mins[1]) / block_dims[1];
//   block_size[2] = (data_maxs[2] - data_mins[2]) / block_dims[2];

//   ijk[0] = (bb.min[0] - data_mins[0]) / block_size[0];
//   ijk[1] = (bb.min[1] - data_mins[1]) / block_size[1];
//   ijk[2] = (bb.min[2] - data_mins[2]) / block_size[2];

// }
//----------------------------------------------------------------------------
//
// convert point to child block that contains it
//
// pt: point x,y,z
// bf: blocking factor (number of children, including parent) in each
//  dimension, eg (2, 2, 2)
// mins, maxs: global data extents
// child index (i,j,k) (0 to bf -1 in each dimension) (output)
//
void Pt2Child(float *pt, int *bf, float *mins, float *maxs, int *idx) {

  float block_size[3]; // size of block in each dimension

  // block size
  block_size[0] = (maxs[0] - mins[0]) / bf[0];
  block_size[1] = (maxs[1] - mins[1]) / bf[1];
  block_size[2] = (maxs[2] - mins[2]) / bf[2];

  // block index
  idx[0] = (pt[0] - mins[0]) / block_size[0];
  idx[1] = (pt[1] - mins[1]) / block_size[1];
  idx[2] = (pt[2] - mins[2]) / block_size[2];

  // clamp max point to last block, not next block
  idx[0] = (idx[0] >= bf[0] ? bf[0] - 1 : idx[0]);
  idx[1] = (idx[1] >= bf[1] ? bf[1] - 1 : idx[1]);
  idx[2] = (idx[2] >= bf[2] ? bf[2] - 1 : idx[2]);

}
//----------------------------------------------------------------------------
