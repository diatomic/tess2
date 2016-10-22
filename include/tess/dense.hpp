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
//--------------------------------------------------------------------------
#ifndef _DENSE_H
#define _DENSE_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <vector>
#include "delaunay.h"
#include "mpi.h"
// DEPRECATED
// #include "tess/io.h"
#include "tess/tet.h"
#include "tess/tet-neighbors.h"
#include "tess/tess.h"
#include "tess/tess.hpp"

using namespace std;

// estimator algorithm
enum alg
{
    DENSE_TESS,
    DENSE_CIC,
    DENSE_NUM_ALGS,
};

// grid point
struct grid_pt_t
{
    int idx[3]; // global grid point index
    double mass; // mass
};

// auxiliary arguments for foreach block functions
struct args_t
{
    alg   alg_type;
    bool  project;
    float proj_plane[3];
    float mass;
    float data_mins[3];
    float data_maxs[3];
    float grid_phys_mins[3];
    float grid_step_size[3];
    float eps;
    int   glo_num_idx[3];
    float div;
};

// timing
enum
{
    INPUT_TIME,
    COMP_TIME,
    OUTPUT_TIME,
    TOTAL_TIME,
    TESS_TIME,
    DENSE_MAX_TIMES
};

// function prototypes
void dense(alg alg_type,
           int num_given_bounds,
           float *given_mins,
	   float *given_maxs,
           bool project,
           float *proj_plane,
           float mass,
           float *data_mins,
           float *data_maxs,
           float *grid_phys_mins,
           float *grid_phys_maxs,
	   float *grid_step_size,
           float eps,
           int *glo_num_idx,
           diy::Master& master);
void init_dense(dblock_t*                         b,
                const diy::Master::ProxyWithLink& cp,
                args_t*                           a);
void est_dense(dblock_t*                         b,
               const diy::Master::ProxyWithLink& cp,
               args_t*                           a);
void recvd_pts(dblock_t*                         b,
               const diy::Master::ProxyWithLink& cp,
               args_t*                           a);
void BlockGridParams(dblock_t *dblock,
                     int *block_min_idx,
                     int *block_max_idx,
                     int *block_num_idx,
		     float *grid_phys_mins,
                     float *grid_step_size,
                     float eps,
                     float *data_mins,
                     float *data_maxs,
                     int *glo_num_idx);
void IterateCells(dblock_t *dblock,
                  int *block_min_idx,
                  int *block_num_idx,
                  bool project,
                  float *proj_plane,
                  float *grid_phys_mins,
                  float *grid_step_size,
		  float *data_mins,
                  float *data_maxs,
                  float eps,
                  float mass,
                  const diy::Master::ProxyWithLink& cp);
#ifndef TESS_NO_OPENMP
void IterateCellsOMP(dblock_t *dblock,
                     int *block_min_idx,
                     int *block_num_idx,
                     bool project,
                     float *proj_plane,
                     float *grid_phys_mins,
                     float *grid_step_size,
		     float *data_mins,
                     float *data_maxs,
                     float eps,
                     float mass,
                     const diy::Master::ProxyWithLink& cp);
#endif
void IterateCellsCic(dblock_t *dblock,
                     int *block_min_idx,
                     int *block_num_idx,
                     bool project,
                     float *proj_plane,
                     float *grid_phys_mins,
                     float *grid_step_size,
		     float *data_maxs,
                     float eps,
                     float mass,
                     const diy::Master::ProxyWithLink& cp);
void CellBounds(dblock_t *dblock,
                int cell,
                float *cell_min,
                float *cell_max,
		vector<float> &normals,
                vector <vector <float> > &face_verts);
int CellGridPts(float *cell_mins,
                float *cell_maxs,
                grid_pt_t* &grid_pts,
                int * &border,
                int& alloc_grid_pts,
                vector<float> &normals,
                vector <vector <float> > &face_verts,
                float *data_mins,
                float *data_maxs,
                float *grid_phys_mins,
                float *grid_step_size,
		float mass,
                float eps,
                float* site);
int CellInteriorGridPts(int *cell_grid_pts,
                        int *cell_min_grid_idx,
                        float *cell_min_grid_pos,
                        grid_pt_t *grid_pts,
                        int *border,
                        vector<float> &normals,
			vector <vector <float> > &face_verts,
                        float *grid_step_size,
                        float eps,
                        float mass);
bool PtInCell(float *pt,
              vector<float> &normals,
              vector <vector <float> > &face_verts,
              float eps);
void NewellNormal(float *verts,
                  int num_verts,
                  float *normal);
void Global2LocalIdx(int *global_idx,
                     int *local_idx,
                     int *block_min_idx);
void GridStepParams(int num_given_bounds,
                    float *given_mins,
                    float *given_maxs,
                    float *data_mins,
                    float *data_maxs,
                    float *grid_phys_mins,
                    float *grid_phys_maxs,
		    float *grid_step_size,
                    int *glo_num_idx);
void WriteGrid(int mblocks,
               int tblocks,
               char *outfile,
               bool project,
               int *glo_num_idx,
               float eps,
               float *data_mins,
	       float *data_maxs,
               int num_given_bounds,
               float *given_mins,
               float *given_maxs,
               diy::Master& master,
               diy::Assigner& assigner);
void ProjectGrid(int gnblocks,
                 int *glo_num_idx,
                 float eps,
                 float *data_mins,
                 float *data_maxs,
                 float *grid_phys_mins,
                 float *grid_step_size,
                 diy::Master& master,
                 diy::Assigner& assigner);
void handle_error(int errcode,
                  char *str,
                  MPI_Comm comm);
int index(int *block_grid_idx,
          int *block_num_idx,
          bool project,
          float *proj_plane);
void idx2phys(int *grid_idx,
              float *pos,
              float *grid_step_size,
              float *grid_phys_mins);
void phys2idx(float *pos,
              int *grid_idx,
              float *grid_step_size,
              float *grid_phys_mins);
void DataBounds(float *data_mins,
                float *data_maxs,
                diy::Master& master);
void dense_stats(double *times,
                 diy::Master& master,
                 float *grid_step_size,
                 float *grid_phys_mins,
                 int *glo_num_idx);
void DistributeScalarCIC(float *pt,
                         float scalar,
                         vector <int> &grid_idxs,
                         vector <float> &grid_scalars,
                         float *grid_step_size,
                         float *grid_phys_mins,
                         float eps);

#endif
