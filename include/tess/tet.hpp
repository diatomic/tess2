#ifndef _TESS_TET_HPP
#define _TESS_TET_HPP

#include "tess/tet.h"
#include <diy/types.hpp>

int find(struct tet_t* tet,
         int v);
float dot(float* u,
          float* p,
          float* x);
void circumcenter(float* c,
                  struct tet_t* tet,
                  float* particles);
int side_of_plane(diy::Bounds<float> box,
                  struct tet_t* tet,
                  float* particles,
                  int j);
int circulate_start(struct tet_t* tets,
                    int t,
                    int x,
                    int y);
void circulate_next(int* next_t,
                    int* next_v,
                    struct tet_t* tets,
                    int t,
                    int v,
                    int x,
                    int y);
float distance(float* u,
               float* v);
void cross(float* res,
           float* u,
           float* v);
float norm(float* x);
float determinant(float* t,
                  float* u,
                  float* v);

#endif
