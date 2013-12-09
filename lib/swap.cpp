//---------------------------------------------------------------------------
//
// swap utilities
//
// Tom Peterka
// Argonne National Laboratory
// 9700 S. Cass Ave.
// Argonne, IL 60439
// tpeterka@mcs.anl.gov
//
// (C) 2011 by Argonne National Laboratory.
// See COPYRIGHT in top-level directory.
//
//---------------------------------------------------------------------------
#include <stdio.h>
#include "swap.hpp"

//
// swaps bytes
//
// n: address of items
// nitems: number of items
// item_size: either 2, 4, or 8 bytes
// returns quietly if item_size is 1
//
void Swap(char *n, int nitems, int item_size) {

  int i;

  switch(item_size) {
  case 1:
    break;
  case 2:
    for (i = 0; i < nitems; i++) {
      Swap2(n);
      n += 2;
    }
    break;
  case 4:
    for (i = 0; i < nitems; i++) {
      Swap4(n);
      n += 4;
    }
    break;
  case 8:
    for (i = 0; i < nitems; i++) {
      Swap8(n);
      n += 8;
    }
    break;
  default:
    fprintf(stderr, "Error: size of data must be either 1, 2, 4, or 8 bytes per item\n");
    //MPI_Abort(MPI_COMM_WORLD, 0);

  }

}
//-----------------------------------------------------------------------
//
// Swaps 8  bytes from 1-2-3-4-5-6-7-8 to 8-7-6-5-4-3-2-1 order.
// cast the input as a char and use on any 8 byte variable
//
void Swap8(char *n) {

  char *n1;
  char c;

  n1 = n + 7;
  c = *n;
  *n = *n1;
  *n1 = c;

  n++;
  n1--;
  c = *n;
  *n = *n1;
  *n1 = c;

  n++;
  n1--;
  c = *n;
  *n = *n1;
  *n1 = c;

  n++;
  n1--;
  c = *n;
  *n = *n1;
  *n1 = c;

}
//-----------------------------------------------------------------------------
//
// Swaps 4 bytes from 1-2-3-4 to 4-3-2-1 order.
// cast the input as a char and use on any 4 byte variable
//
void Swap4(char *n) {

  char *n1;
  char c;

  n1 = n + 3;
  c = *n;
  *n = *n1;
  *n1 = c;

  n++;
  n1--;
  c = *n;
  *n = *n1;
  *n1 = c;

}
//----------------------------------------------------------------------------
//
// Swaps 2 bytes from 1-2 to 2-1 order.
// cast the input as a char and use on any 2 byte variable
//
void Swap2(char *n){

  char c;

  c = *n;
  *n = n[1];
  n[1] = c;

}
//----------------------------------------------------------------------------

