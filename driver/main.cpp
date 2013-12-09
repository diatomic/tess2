#include "mpi.h"
#include <assert.h>
#include "tess.h"

void GetArgs(int argc, char **argv, int &tb, int *dsize, float *jitter,
	     float *minvol, float *maxvol, int *wrap);

int main(int argc, char *argv[]) {

  int tb; // total number of blocks in the domain
  int dsize[3]; // domain grid size
  float jitter; // max amount to randomly displace particles
  float minvol, maxvol; // volume range, -1.0 = unused
  double times[MAX_TIMES]; // timing
  int wrap; // wraparound neighbors flag

  MPI_Init(&argc, &argv);

  GetArgs(argc, argv, tb, dsize, &jitter, &minvol, &maxvol, &wrap);

  tess_test(tb, dsize, jitter, minvol, maxvol, wrap, times);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Finalize();

  return 0;

}
//----------------------------------------------------------------------------
//
// gets command line args
//
void GetArgs(int argc, char **argv, int &tb, int *dsize, float *jitter,
	     float *minvol, float *maxvol, int *wrap) {

  assert(argc >= 9);

  tb = atoi(argv[1]);
  dsize[0] = atoi(argv[2]);
  dsize[1] = atoi(argv[3]);
  dsize[2] = atoi(argv[4]);
  *jitter = atof(argv[5]);
  *minvol = atof(argv[6]);
  *maxvol = atof(argv[7]);
  *wrap = atoi(argv[8]);

}
//----------------------------------------------------------------------------
