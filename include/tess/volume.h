#ifndef _TESS_VOLUME_H
#define _TESS_VOLUME_H

#include <vector>
#include "tet.h"

void fill_circumcenters(std::vector<float>& circumcenters, tet_t* tets, int num_tets, float* particles);

float volume(int v, int* verts_to_tets, tet_t* tets, int num_tets, float* particles, const std::vector<float>& circumcenters);

#endif
