#include "mpi.h"
#include <assert.h>
#include "tess.h"
#include <vector>
#include <stdio.h>
#include "pread.h"

using namespace std;

void GetArgs(int argc, char **argv, int &tb, char *infile, char *outfile,
	     std::vector<std::string>& coordinates,
	     float *minvol, float *maxvol, int *swap, int *wrap);

void AllReduceMinsMaxs(vector<float>& particles, float* mins, float* maxs);

void ExchangeParticles(vector <float> p, float *mins, float *maxs, 
		       float **particles, int *num_particles,
		       int rank, int size);
int Pt2Gid(float *pt, float *mins, float *maxs);

int main(int argc, char *argv[]) {

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
  int swap; // whether byte swapping
  int wrap; // whether wraparound neighbors are used
  int rank,size; // MPI usual
  vector <float> p; // temporary particles
  std::vector<std::string>  coordinates; // coordinates to read

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  GetArgs(argc, argv, tot_blocks, infile, outfile,
          coordinates,
          &minvol, &maxvol, 
	  &swap, &wrap);

  read_particles(infile, rank, size, p, coordinates, swap);

  AllReduceMinsMaxs(p, mins, maxs);

  if (rank == 0) {
    fprintf(stderr, "mins = [%.3f %.3f %.3f] maxs = [%.3f %.3f %.3f]\n",
	    mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2]);
  }
  //std::cout << rank << ": " << p.size()/3 << std::endl;
  //std::cout << "Mins: " << mins[0] << "," << mins[1] << "," << mins[2] << std::endl;
  //std::cout << "Maxs: " << maxs[0] << "," << maxs[1] << "," << maxs[2] << std::endl;

  // initialize DIY and decompose domain
  DIY_Init(dim, num_threads, MPI_COMM_WORLD);
  DIY_Decompose_cont(ROUND_ROBIN_ORDER, tot_blocks, &nblocks,
		     mins, maxs, block_ghost, block_given, wrap);

  // must use malloc instead of new because tess will realloc during its
  // particle exchange
  particles	= (float **)malloc(nblocks * sizeof(float *));
  num_particles = (int *)   malloc(nblocks * sizeof(int));

  // sort and distribute particles to all blocks
  ExchangeParticles(p, mins, maxs, particles, num_particles, rank, size);
  
  printf("%d: %d\n", rank, num_particles[0]);
  
  //for (int i = 0; i < nblocks; ++i) {
  //  std::cout << rank << ": " << num_particles[i] << std::endl;
  //}

  // run tess in post processing mode
  tess_init_diy_exist(nblocks, mins, maxs, wrap, 0, minvol, maxvol,
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
	     std::vector<std::string>& coordinates,
	     float *minvol, float *maxvol, int *swap, int *wrap) {

  assert(argc >= 10);

  tb = atoi(argv[1]);
  strcpy(infile, argv[2]);

  if (argv[3][0] =='!')
    strcpy(outfile, "");
  else
    strcpy(outfile, argv[3]);

  coordinates.resize(3);
  coordinates[0] = argv[4];
  coordinates[1] = argv[5];
  coordinates[2] = argv[6];
  *minvol = atof(argv[7]);
  *maxvol = atof(argv[8]);
  *swap = atoi(argv[9]);
  *wrap = atoi(argv[10]);

}
//----------------------------------------------------------------------------
//
// figure out min and max coordinates across all the processes
//
void AllReduceMinsMaxs(vector<float>& particles, float* mins, float* maxs)
{
  for (int i = 0; i < 3; ++i)
  {
    mins[i] = particles[i];
    maxs[i] = particles[i];
  }
  for (size_t j = 0; j < particles.size()/3; ++j)
  {
    for (int i = 0; i < 3; ++i)
    {
      float c = particles[3*j + i];
      if (c < mins[i])
	mins[i] = c;
      if (c > maxs[i])
	maxs[i] = c;
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, mins, 3, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, maxs, 3, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
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
void ExchangeParticles(vector <float> p, float *mins, float *maxs,
		       float **particles, int *num_particles,
		       int rank, int size) {

  struct bb_t bb; // block bounds
  int nblocks = DIY_Num_lids(0);
  int *c; // counts of number of particles being sent from root to each block

  // sort and send particles
  // particle vectors for each block
  vector< vector<float> > ps(size);	    // particle vectors for each rank

  // sort particles into individual vectors for each block

  // for all particles
  for (int i = 0; i < (int)p.size() / 3; i++) {
  
    int gid	= Pt2Gid(&p[3 * i], mins, maxs);
    int to_rank = gid % size;
    // debug
//     fprintf(stderr, "pt [%.3f %.3f %.3f] belongs in gid %d\n",
//          p[3 * i], p[3 * i + 1], p[3 * i + 2], gid);
    ps[to_rank].push_back(p[3 * i]);
    ps[to_rank].push_back(p[3 * i + 1]);
    ps[to_rank].push_back(p[3 * i + 2]);

  } // for all particles
  p.clear();

  vector<int> send_counts(size);
  vector<int> send_displs(size);	    // send displacements
  vector<int> recv_counts(size);
  vector<int> recv_displs(size);	    // recv displacements
  int current = 0;
  for (int i = 0; i < size; ++i) {
    send_counts[i]   = ps[i].size();
    send_displs[i]   = current;
    current	    += send_counts[i];
  }

  // flatten particles
  vector<float> flat_particles;
  flat_particles.reserve(current);	    // current is total
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < ps[i].size(); ++j) {
      flat_particles.push_back(ps[i][j]);
    }
  }
  ps.clear();

  // exchange particle counts
  MPI_Alltoall(&send_counts[0], 1, MPI_INT,
	       &recv_counts[0], 1, MPI_INT,
	       MPI_COMM_WORLD);

  // allocate receive particles
  int num_recvd = 0;
  for (int i = 0; i < size; ++i) {
    recv_displs[i] = num_recvd;
    num_recvd     += recv_counts[i];
  }
  vector<float> flat_recv_part(num_recvd);

  MPI_Alltoallv(&flat_particles[0], &send_counts[0], &send_displs[0], MPI_FLOAT,
		&flat_recv_part[0], &recv_counts[0], &recv_displs[0], MPI_FLOAT,
		MPI_COMM_WORLD);
		
  // assign particles to their blocks
  ps.resize(nblocks);
  for (int j = 0; j < num_recvd/3; ++j) {
    int gid = Pt2Gid(&flat_recv_part[3 * j], mins, maxs);
    int i   = gid/size;		// local block
  
    struct bb_t bounds; /* block bounds */
    DIY_Block_bounds(0, i, &bounds);

    float x = flat_recv_part[3 * j];
    float y = flat_recv_part[3 * j + 1];
    float z = flat_recv_part[3 * j + 2];

    ps[i].push_back(x);
    ps[i].push_back(y);
    ps[i].push_back(z);

    // For debugging purposes
    //if (bounds.min[0] > x || x > bounds.max[0] ||
    //    bounds.min[1] > y || y > bounds.max[1] ||
    //    bounds.min[2] > z || z > bounds.max[2]) {
    //  std::cerr << "Warning: " << x << "," << y << "," << z
    //            << " not in "
    //            << bounds.min[0] << "," << bounds.min[1] << "," << bounds.min[2]
    //            << " - "
    //            << bounds.max[0] << "," << bounds.max[1] << "," << bounds.max[2]
    //            << std::endl;
    //}
  }

  // copy ps out to particles
  for (int i = 0; i < nblocks; ++i) {
    num_particles[i] = ps[i].size() / 3;
    particles[i]     = (float*) malloc(ps[i].size() * sizeof(float));
    for (int j = 0; j < ps[i].size(); ++j) {
      particles[i][j] = ps[i][j];
    }
  }
  ps.clear();
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
