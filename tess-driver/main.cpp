#include "mpi.h"
#include <assert.h>
#include "tess.h"

void GetArgs(int argc, char **argv, int &tb, int *dsize, float *jitter,
	     float *minvol, float *maxvol, int *wrap, int *walls, char *outfile);

int main(int argc, char *argv[]) {

  int tb; // total number of blocks in the domain
  int dsize[3]; // domain grid size
  float jitter; // max amount to randomly displace particles
  float minvol, maxvol; // volume range, -1.0 = unused
  double times[MAX_TIMES]; // timing
  int wrap; // wraparound neighbors flag
  int walls; // apply walls to simulation (wrap must be off)
  char outfile[256]; // output file name

  MPI_Init(&argc, &argv);

  GetArgs(argc, argv, tb, dsize, &jitter, &minvol, &maxvol, &wrap, &walls, outfile);

  tess_test(tb, dsize, jitter, minvol, maxvol, wrap, walls, times, outfile);

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
	     float *minvol, float *maxvol, int *wrap, int *walls, char *outfile) {

  assert(argc >= 11);

  tb = atoi(argv[1]);
  dsize[0] = atoi(argv[2]);
  dsize[1] = atoi(argv[3]);
  dsize[2] = atoi(argv[4]);
  *jitter = atof(argv[5]);
  *minvol = atof(argv[6]);
  *maxvol = atof(argv[7]);
  *wrap = atoi(argv[8]);
  *walls = atoi(argv[9]);
  if (argv[10][0] =='!')
    strcpy(outfile, "");
  else
    strcpy(outfile, argv[10]);

}
//----------------------------------------------------------------------------
