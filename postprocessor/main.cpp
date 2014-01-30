#include "mpi.h"
#include <assert.h>
#include "tess.h"
#include <vector>
#include <stdio.h>

using namespace std;

void GetArgs(int argc, char **argv, int &tb, char *infile, char *outfile,
	     float *minvol, float *maxvol, int *intype, int *wrap);
void read_text_particles(char *infile, vector <float> &particles, float *mins,
			 float *maxs);
void read_double_particles(char *infile, vector <float> &particles, float *mins,
			   float *maxs);
void read_float_interleaved_particles(char *infile, vector <float> &particles, 
				      float *mins, float *maxs);
void SortParticles(vector <float> p, float *mins, float *maxs, 
		   float **particles, int *num_particles);
int Pt2Gid(float *pt, float *mins, float *maxs);

int main(int argc, char *argv[]) {

  // input file type 
  // 0 = text
  // 1 = float x's followed by y's followed by z's
  // 2 = float interleaved x y z
  // 3 = double x's followed by y's followed by z's
  // 4 = double interleaved x y z
  int intype;

  int tot_blocks; // total number of blocks in the domain
  char infile[256]; // input file name
  char outfile[256]; // output file name
  float mins[3], maxs[3]; // data global extents
  float minvol, maxvol; // volume range, -1.0 = unused
  double times[MAX_TIMES]; // timing
  float **particles; // particles[block_num][particle] 
		     //  where each particle is 3 values, px, py, pz
  int *num_particles; // number of particles in each block
  int nblocks; // my local number of blocks; todo
  int num_threads = 1; // number of threads diy can use
  int dim = 3; // 3d always
  int block_given[3] = {0, 0, 0}; // constraints on blocking (none)
  float block_ghost[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // ghost block overlap
  int wrap; // whether wraparound neighbors are used
  int rank; // MPI usual
  vector <float> p; // temporary particles

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  GetArgs(argc, argv, tot_blocks, infile, outfile, &minvol, &maxvol, 
	  &intype, &wrap);

  // root read points in serial
  if (rank == 0) {

    switch (intype) {
    case 0:
      read_text_particles(infile, p, mins, maxs);
      break;
    case 2:
      read_float_interleaved_particles(infile, p, mins, maxs);
      break;
    case 3:
      read_double_particles(infile, p, mins, maxs);
      break;
    default:
      fprintf(stderr, "file type %d not supported yet\n", intype);
      break;
    }

    // debug
    fprintf(stderr, "mins = [%.3f %.3f %.3f] maxs = [%.3f %.3f %.3f]\n",
	    mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2]);

  }
  // broadcast global extents (todo: combine into one broadcast)
  MPI_Bcast(mins, 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(maxs, 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // initialize DIY and decompose domain
  DIY_Init(dim, num_threads, MPI_COMM_WORLD);
  DIY_Decompose_cont(ROUND_ROBIN_ORDER, tot_blocks, &nblocks,
		     mins, maxs, block_ghost, block_given, wrap);

  // must use malloc instead of new because tess will realloc during its
  // particle exchange
  particles = (float **)malloc(nblocks * sizeof(float *));
  num_particles = (int *)malloc(nblocks * sizeof(int));

  // sort and distribute particles to all blocks
  SortParticles(p, mins, maxs, particles, num_particles);

  // run tess in post processing mode
  tess_init_diy_exist(nblocks, mins, maxs, wrap, minvol, maxvol, 
		      MPI_COMM_WORLD, times);
  tess(particles, num_particles, outfile);

  // cleanup
  for (int b = 0; b < nblocks; b++)
    free(particles[b]);
  free(particles);
  free(num_particles);

  DIY_Finalize();
  MPI_Finalize();

  return 0;

}
//----------------------------------------------------------------------------
//
// gets command line args
//
void GetArgs(int argc, char **argv, int &tb, char *infile, char *outfile,
	     float *minvol, float *maxvol, int *intype, int *wrap) {

  assert(argc >= 8);

  tb = atoi(argv[1]);
  strcpy(infile, argv[2]);
  strcpy(outfile, argv[3]);
  *minvol = atof(argv[4]);
  *maxvol = atof(argv[5]);
  *intype = atoi(argv[6]);
  *wrap = atoi(argv[7]);

}
//----------------------------------------------------------------------------
//
// reads text particles from a file
//
// infile: input file name
// particles: (output) particles
// mins, maxs: (output) global data extents
//
// crrently reads all particles into one block
//
void read_text_particles(char *infile, vector <float> &particles, float *mins,
			 float *maxs) {

  FILE *fd;
  char line[256];
  float x, y, z;

  fd = fopen(infile, "r");
  assert(fd != NULL);

  while (fgets(line, sizeof(line), fd)) {

    int n = sscanf(line, "%f %f %f", &x, &y, &z);
    assert(n == 3);

    if ((int)particles.size() == 0) {
      mins[0] = x;
      mins[1] = y;
      mins[2] = z;
      maxs[0] = x;
      maxs[1] = y;
      maxs[2] = z;
    }
    else {
      if (x < mins[0])
	mins[0] = x;
      if (y < mins[1])
	mins[1] = y;
      if (z < mins[2])
	mins[2] = z;
      if (x > maxs[0])
	maxs[0] = x;
      if (y > maxs[1])
	maxs[1] = y;
      if (z > maxs[2])
	maxs[2] = z;
    }

    particles.push_back(x);
    particles.push_back(y);
    particles.push_back(z);

  }

  fclose(fd);

}
//----------------------------------------------------------------------------
//
// reads double precision particles from a raw binary file organized by
// all x's followed by all y'z followed by all z's
//
// infile: input file name
// particles: (output) particles
// mins, maxs: (output) global data extents
//
// crrently reads all particles into one block
//
void read_double_particles(char *infile, vector <float> &particles, float *mins,
			   float *maxs) {

  FILE *fd;

  fd = fopen(infile, "r");
  assert(fd != NULL);

  // get number of particles from file size
  fseek(fd, 0, SEEK_END);
  int64_t file_size = ftell(fd); // file size in bytes
  int64_t num_particles = file_size / sizeof(double) / 3; // number of particles
  fseek(fd, 0, SEEK_SET);

  // debug
  fprintf(stderr, "number of particles = %d\n", (int)num_particles);

  // read individual arrays
  double *x = new double[num_particles];
  double *y = new double[num_particles];
  double *z = new double[num_particles];
  fread(x, sizeof(double), num_particles, fd);
  fread(y, sizeof(double), num_particles, fd);
  fread(z, sizeof(double), num_particles, fd);

  // interleave particle coordinates, convert to single precision, store in
  // output vector
  for (int i = 0; i < num_particles; i++) {

    if (i == 0) {
      mins[0] = x[i];
      mins[1] = y[i];
      mins[2] = z[i];
      maxs[0] = x[i];
      maxs[1] = y[i];
      maxs[2] = z[i];
    }
    else {
      if (x[i] < mins[0])
	mins[0] = x[i];
      if (y[i] < mins[1])
	mins[1] = y[i];
      if (z[i] < mins[2])
	mins[2] = z[i];
      if (x[i] > maxs[0])
	maxs[0] = x[i];
      if (y[i] > maxs[1])
	maxs[1] = y[i];
      if (z[i] > maxs[2])
	maxs[2] = z[i];
    }

    particles.push_back(x[i]);
    particles.push_back(y[i]);
    particles.push_back(z[i]);

    // debug
//     fprintf(stderr, "%.3f %.3f %.3f\n", 
// 	    particles[particles.size() - 3],
// 	    particles[particles.size() - 2],
// 	    particles[particles.size() - 1]);

  }

  // cleanup
  delete[] x;
  delete[] y;
  delete[] z;
  fclose(fd);

}
//----------------------------------------------------------------------------
//
// reads single precision particles from a raw binary file organized by
// interleaved x,y,z,x,y,z,...
//
// infile: input file name
// particles: (output) particles
// mins, maxs: (output) global data extents
//
// crrently reads all particles into one block
//
void read_float_interleaved_particles(char *infile, vector <float> &particles, 
				      float *mins, float *maxs) {

  FILE *fd;

  fd = fopen(infile, "r");
  assert(fd != NULL);

  // get number of particles from file size
  fseek(fd, 0, SEEK_END);
  int64_t file_size = ftell(fd); // file size in bytes
  int64_t num_particles = file_size / sizeof(float) / 3; // number of particles
  fseek(fd, 0, SEEK_SET);

  // debug
  fprintf(stderr, "number of particles = %d\n", (int)num_particles);

  // read particles
  particles.resize(num_particles * 3);
  fread(&particles[0], sizeof(float), num_particles * 3, fd);

  // find extents
  for (int i = 0; i < num_particles; i++) {

    if (i == 0) {
      mins[0] = particles[3 * i];
      mins[1] = particles[3 * i + 1];
      mins[2] = particles[3 * i + 2];
      maxs[0] = particles[3 * i];
      maxs[1] = particles[3 * i + 1];
      maxs[2] = particles[3 * i + 2];
    }
    else {
      if (particles[3 * i] < mins[0])
	mins[0] = particles[3 * i];
      if (particles[3 * i + 1] < mins[1])
	mins[1] = particles[3 * i + 1];
      if (particles[3 * i + 2] < mins[2])
	mins[2] = particles[3 * i + 2];
      if (particles[3 * i] > maxs[0])
	maxs[0] = particles[3 * i];
      if (particles[3 * i + 1] > maxs[1])
	maxs[1] = particles[3 * i + 1];
      if (particles[3 * i + 2] > maxs[2])
	maxs[2] = particles[3 * i + 2];
    }

    // debug
//     fprintf(stderr, "%.3f %.3f %.3f\n", 
// 	    particles[particles.size() - 3],
// 	    particles[particles.size() - 2],
// 	    particles[particles.size() - 1]);

  }

  // cleanup
  fclose(fd);

}
//----------------------------------------------------------------------------
//
// sort and distribute particles to all blocks
//
// p: input particles (x,y,z,x,y,z,...)
// mins, maxs: global data extents
// particles: particles in each of my local blocks (output)
// num_particles: number of particles in each of my local blocks (output)
//
void SortParticles(vector <float> p, float *mins, float *maxs,
		   float **particles, int *num_particles) {

  struct bb_t bb; // block bounds
  int nblocks = DIY_Num_lids(0);
  vector<float> *ps; // particle vectors for each block
  int *c; // oounts of number of particls being sent from root to each block

  // root block only, sort and send particles
  if (DIY_Gid(0, 0) == 0) {

    // particle vectors for each block
    ps = new vector<float>[DIY_Num_gids(0)]; 

    // sort particles into individual vectors for each block

    // for all particles
    for (int i = 0; i < (int)p.size() / 3; i++) {
    
      int gid = Pt2Gid(&p[3 * i], mins, maxs);
      // debug
//       fprintf(stderr, "pt [%.3f %.3f %.3f] belongs in gid %d\n",
// 	      p[3 * i], p[3 * i + 1], p[3 * i + 2], gid);
      ps[gid].push_back(p[3 * i]);
      ps[gid].push_back(p[3 * i + 1]);
      ps[gid].push_back(p[3 * i + 2]);

    } // for all particles

    // send particles to all blocks
    for (int j = 0; j < DIY_Num_gids(0); j++)
      DIY_Send(0, 0, (void *)&(ps[j][0]), (int)ps[j].size(), DIY_FLOAT, j);

  } // root block only

  // receive particles for my local blocks
  void *recv_pts[1]; // each block receives one item (an array of floats)
  int count; // number of items received for a block, 1 array
  int src_gids[1]; // sender of received item
  int sizes[1]; // size of received item in datatype units
  for (int i = 0; i < nblocks; i++) {

    DIY_Recv(0, i, recv_pts, &count, 1, DIY_FLOAT, src_gids, sizes);
    assert(count == 1); // sanity
    num_particles[i] = sizes[0] / 3;
    // must use malloc instead of new because tess will realloc during its
    // particle exchange
    particles[i] = (float *)malloc(num_particles[i] * 3 * sizeof(float));

    // debug
    fprintf(stderr, "gid %d received %d particles from gid %d\n",
	    DIY_Gid(0, i), num_particles[i], src_gids[0]);

    // copy received particles
    for (int j = 0; j < num_particles[i]; j++) {

      particles[i][3 * j] = DIY_Recvd_item(float, recv_pts, 0)[3 * j];
      particles[i][3 * j + 1] = DIY_Recvd_item(float, recv_pts, 0)[3 * j + 1];
      particles[i][3 * j + 2] = DIY_Recvd_item(float, recv_pts, 0)[3 * j + 2];
      // debug
//       fprintf(stderr, "gid %d received [%.3f %.3f %.3f]\n",
// 	      DIY_Gid(0, i), DIY_Recvd_item(float, recv_pts, 0)[3 * j],
// 	      DIY_Recvd_item(float, recv_pts, 0)[3 * j + 1],
// 	      DIY_Recvd_item(float, recv_pts, 0)[3 * j + 2]);

    }

  }
  DIY_Flush_send_recv(0);

  // root block only, cleanup vectors used for sending after send/recv
  // is completed
  if (DIY_Gid(0, 0) == 0) {
    for (int j = 1; j < DIY_Num_gids(0); j++)
      ps[j].clear();
    delete[] ps;
  }

}
//----------------------------------------------------------------------------
//
// convert point to block global id that contains it
//
// pt: point x,y,z
// mins, maxs: global data extents
//
// returns: block gid, < 0 on error
//
int Pt2Gid(float *pt, float *mins, float *maxs) {

  int idx[3]; // block index
  int dim_nblocks[3]; // number of blocks in each dimension
  float block_size[3]; // size of block in each dimension

  // block size
  DIY_Num_gids_dim(0, dim_nblocks);
  block_size[0] = (maxs[0] - mins[0]) / dim_nblocks[0];
  block_size[1] = (maxs[1] - mins[1]) / dim_nblocks[1];
  block_size[2] = (maxs[2] - mins[2]) / dim_nblocks[2];

  // block index
  idx[0] = (pt[0] - mins[0]) / block_size[0];
  idx[1] = (pt[1] - mins[1]) / block_size[1];
  idx[2] = (pt[2] - mins[2]) / block_size[2];

  // clamp max point to last block, not next block
  idx[0] = (idx[0] >= dim_nblocks[0] ? dim_nblocks[0] - 1 : idx[0]);
  idx[1] = (idx[1] >= dim_nblocks[1] ? dim_nblocks[1] - 1 : idx[1]);
  idx[2] = (idx[2] >= dim_nblocks[2] ? dim_nblocks[2] - 1 : idx[2]);

  // gid is row-major ordering of idx
  return(idx[0] + idx[1] * dim_nblocks[0] + 
	 idx[2] * (dim_nblocks[0] * dim_nblocks[1]));

}
//----------------------------------------------------------------------------
