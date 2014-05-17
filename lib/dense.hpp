//---------------------------------------------------------------------------
//
// density estimator header file
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
#ifndef _DENSE_H
#define _DENSE_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <vector>
#include "voronoi.h"
#include "delaunay.h"
#include <math.h>
#include "mpi.h"
#include "diy.h"
#include "io.h"
#include "tet.h"
#include "tet-neighbors.h"

using namespace std;

// grid point
struct grid_pt_t {
  int idx[3]; // global grid point index
  double mass; // mass
};

// timing
#define DENSE_MAX_TIMES 4
#define INPUT_TIME 0
#define COMP_TIME 1
#define OUTPUT_TIME 2
#define TOTAL_TIME 3

// function prototypes
void dense(float **density, int nblocks, double times[DENSE_MAX_TIMES], 
	   MPI_Comm comm, int num_given_bounds, float *given_mins, 
	   float *given_maxs, bool project, float *proj_plane, 
	   float mass, float *data_mins, float *data_maxs, 
	   dblock_t **dblocks, float *grid_phys_mins, float *grid_phys_maxs,
	   float *grid_step_size, float eps, int *glo_num_idx);
void BlockGridParams(int lid, int *block_min_idx, int *block_max_idx,
		     int *block_num_idx, dblock_t **dblocks,
		     float *grid_phys_mins, float *grid_step_size, float eps,
		     float *data_mins, float *data_maxs, int *glo_num_idx);
void IterateCells(int block, int *block_min_idx, int *block_num_idx, 
		  float **density, bool project, float *proj_plane,
		  float *grid_phys_mins, float *grid_step_size, 
		  dblock_t **dblocks, float *data_mins, float *data_maxs, 
		  float eps, float mass);
void IterateCellsOMP(int block, int *block_min_idx, int *block_num_idx, 
		     float **density, bool project, float *proj_plane,
		     float *grid_phys_mins, float *grid_step_size, 
		     dblock_t **dblocks, float *data_mins, float *data_maxs, 
		     float eps, float mass);
void IterateCellsCic(int block, int *block_min_idx, int *block_num_idx, 
		     float **density, bool project, float *proj_plane,
		     float *grid_phys_mins, float *grid_step_size, 
		     dblock_t **dblocks, float *data_maxs, 
		     float eps, float mass);
void CellBounds(dblock_t *dblock, int cell, float *cell_min, float *cell_max, 
		vector<float> &normals, vector <vector <float> > &face_verts);
int CellGridPts(float *cell_mins, float *cell_maxs, grid_pt_t* &grid_pts, 
		int * &border, int& alloc_grid_pts, vector<float> &normals, 
		vector <vector <float> > &face_verts, float *data_mins,
		float *data_maxs, float *grid_phys_mins, float *grid_step_size,
		float mass, float eps);
int CellInteriorGridPts(int *cell_grid_pts, int *cell_min_grid_idx, 
			float *cell_min_grid_pos, grid_pt_t *grid_pts, 
			int *border, vector<float> &normals, 
			vector <vector <float> > &face_verts,
			float *grid_step_size, float eps, float mass);
bool PtInCell(float *pt, vector<float> &normals, 
	      vector <vector <float> > &face_verts, float eps);
void Normal(float *verts, float *normal);
void Global2LocalIdx(int *global_idx, int *local_idx, int *block_min_idx);
void GridStepParams(int num_given_bounds, float *given_mins, 
		    float *given_maxs, float *data_mins, float *data_maxs, 
		    float *grid_phys_mins, float *grid_phys_maxs, 
		    float *grid_step_size, int *glo_num_idx);
void ItemDtype(DIY_Datatype *dtype);
void WriteGrid(float **density, MPI_Comm comm, int nblocks, 
	       int mblocks, char *outfile, bool project, int *glo_num_idx,
	       dblock_t **dblocks, float eps, float *data_mins,
	       float *data_maxs, int num_given_bounds, float *fiven_mins,
	       float *given_maxs);
void handle_error(int errcode, char *str, MPI_Comm comm);
int index(int *block_grid_idx, int *block_num_idx, bool project, 
	  float *proj_plane);
void idx2phys(int *grid_idx, float *pos, float *grid_step_size,
	      float *grid_phys_mins);
void phys2idx(float *pos, int *grid_idx, float *grid_step_size,
	      float *grid_phys_mins);
void DataBounds(int nblocks, MPI_Comm comm, float *data_mins, float *data_maxs);
void SummaryStats(double *times, MPI_Comm comm, float *grid_step_size,
		  float *grid_phys_mins, int *glo_num_idx);
void DistributeScalarCIC(float *pt, float scalar,
			 vector <int> &grid_idxs, vector <float> &grid_scalars,
			 float *grid_step_size, float *grid_phys_mins,
			 float eps);

#endif
