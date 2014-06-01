/* --------------------------------------------------------------------------- */
 
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

#ifdef __cplusplus
extern "C"
#endif
int bin_search(int *tbl, int key, int size);
#ifdef __cplusplus
extern "C"
#endif
void add_int(int val, int **vals, int *numvals, int *maxvals, int chunk_size);
#ifdef __cplusplus
extern "C"
#endif
void add_float(float val, float **vals, int *numvals, int *maxvals, 
	       int chunk_size);
#ifdef __cplusplus
extern "C"
#endif
void add_pt(float *val, float **vals, int *numvals, int *maxvals, 
	    int chunk_size);
#ifdef __cplusplus
extern "C"
#endif
void add_sent(struct sent_t val, struct sent_t **vals, int *numvals, 
	      int *maxvals, int chunk_size);
#ifdef __cplusplus
extern "C"
#endif
void add_empty_int(int **vals, int index, int *numitems, int *maxitems, 
		   int chunk_size, int init_val);
