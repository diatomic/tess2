/* ------------------------------------------------------------------------- */
/*
  c utilities
 
  Tom Peterka 
  Argonne National Laboratory
  9700 S. Cass Ave.
  Argonne, IL 60439
  tpeterka@mcs.anl.gov
 
  (C) 2013 by Argonne National Laboratory.
  See COPYRIGHT in top-level directory.
*/
 
/* -------------------------------------------------------------------------- */

#include "tess/tess.h"
#include "tess/utils.h"

/* -------------------------------------------------------------------------- */
/*
  binary search
  tbl: lookup table
  key: search key
  size: number of table elements

  returns: index of key, -1 if not found
*/
int bin_search(int *tbl, int key, int size) {

  int max = size - 1;
  int min = 0;
  int mid;

  while (max >= min) {
    mid = (min + max) / 2;
    if (tbl[mid] < key )
      min = mid + 1;
    else if (tbl[mid] > key)
      max = mid - 1;
    else
      return mid;
  }

  return -1; /* not found */

}
/*--------------------------------------------------------------------------*/
/*
  adds an int to a c-style vector of ints

  val: value to be added
  vals: pointer to dynamic array of values
  numvals: pointer to number of values currently stored, updated by add_int
  maxvals: pointer to number of values currently allocated
  chunk_size: number of values to allocate at a time
*/
void add_int(int val, int **vals, int *numvals, int *maxvals, int chunk_size) {

  /*   first time  */
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

  /*   add the element  */
  (*vals)[*numvals] = val;
  (*numvals)++;

}
/* -------------------------------------------------------------------------- */
/*
  adds a float to a c-style vector of floats

  val: value to be added
  vals: pointer to dynamic array of values
  numvals: pointer to number of values currently stored, updated by add_int
  maxvals: pointer to number of values currently allocated
  chunk_size: number of values to allocate at a time
*/
void add_float(float val, float **vals, int *numvals, int *maxvals, 
	       int chunk_size) {

  /*   first time  */
  if (*maxvals == 0) {
    *vals = (float *)malloc(chunk_size * sizeof(float));
    *numvals = 0;
    *maxvals = chunk_size;
  }

  /*   grow memory  */
  else if (*numvals >= *maxvals) {
    *vals = (float *)realloc(*vals, 
			       (chunk_size + *maxvals) * sizeof(float));
    *maxvals += chunk_size;
  }

  /*   add the element  */
  (*vals)[*numvals] = val;
  (*numvals)++;

}
/* -------------------------------------------------------------------------- */
/*
  adds a point (array of 3 floats) to a c-style vector of x,y,z,x,y,z points

  val: value to be added
  vals: pointer to dynamic array of values
  numvals: pointer to number of points currently stored, updated by add_int
  maxvals: pointer to number of points currently allocated
  chunk_size: number of points to allocate at a time
*/
void add_pt(float *val, float **vals, int *numvals, int *maxvals, 
	    int chunk_size) {

  /*   first time  */
  if (*maxvals == 0) {
    *vals = (float *)malloc(chunk_size * 3 * sizeof(float));
    *numvals = 0;
    *maxvals = chunk_size;
  }

  /*   grow memory  */
  else if (*numvals >= *maxvals) {
    *vals = (float *)realloc(*vals, 
			       (chunk_size + *maxvals) * 3 * sizeof(float));
    *maxvals += chunk_size;
  }

  /*   add the element  */
  (*vals)[3 * *numvals] = val[0];
  (*vals)[3 * *numvals + 1] = val[1];
  (*vals)[3 * *numvals + 2] = val[2];
  (*numvals)++;

}
/* -------------------------------------------------------------------------- */
/*
  checks if an array of ints has been allocated large enough to access
  a given index. If not, grows the array and initializes the empty ints

  vals: pointer to dynamic array of ints
  index: desired index to be accessed
  numitems: pointer to number of items currently stored, ie, 
    last subscript accessed + 1, updated by this function
  maxitems: pointer to number of items currently allocated
  chunk_size: minimum number of items to allocate at a time
  init_val: initalization value for newly allocated items
*/
void add_empty_int(int **vals, int index, int *numitems, int *maxitems, 
		   int chunk_size, int init_val) {

  int i;
  int alloc_chunk; // max of chunk_size and chunk needed to get to the index 

  /*   first time  */
  if (*maxitems == 0) {

    /*     allocate  */
    alloc_chunk = (index < chunk_size ? chunk_size : index + 1);
    *vals = (int *)malloc(alloc_chunk * sizeof(int));
    /*     init empty vals  */
    for (i = 0; i < alloc_chunk; i++)
      (*vals)[i] = init_val;

    *numitems = 0;
    *maxitems = alloc_chunk;

  }

  /*   grow memory  */
  else if (index >= *maxitems) {

    /*     realloc  */
    alloc_chunk = 
      (index < *maxitems + chunk_size ? chunk_size : index + 1 - *maxitems);
    *vals = (int *)realloc(*vals, (alloc_chunk + *maxitems) * sizeof(int));

    /*     init empty buckets  */
    for (i = *maxitems; i < *maxitems + alloc_chunk; i++)
      (*vals)[i] = init_val;

    *maxitems += alloc_chunk;

  }

  (*numitems)++;

}
/* -------------------------------------------------------------------------- */
