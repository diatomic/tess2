#include <cmath>

#include "tess/volume.h"
#include "tess/tet-neighbors.h"

void fill_circumcenters(std::vector<float>& circumcenters, tet_t* tets, int num_tets, float* particles)
{
  circumcenters.resize(num_tets*3);
  for (int i = 0; i < num_tets; ++i)
    circumcenter(&circumcenters[3*i], &tets[i], particles);
}

float volume(int v, int* verts_to_tets, tet_t* tets, int num_tets, float* particles, const std::vector<float>& circumcenters)
{
  int vt = verts_to_tets[v];

  std::vector< std::pair<int, int> >	nbrs;
  bool finite = neighbor_edges(nbrs, v, tets, vt);

  if (!finite)
    return -1;	    // don't compute infinite volumes

  float vol = 0;
  for (int i = 0; i < nbrs.size(); ++i)
  {
    std::vector<int>	edge_link;
    int u  = nbrs[i].first;
    int ut = nbrs[i].second;
    fill_edge_link(edge_link,v,u,ut,tets);

    // area of the Voronoi facet dual to (u,v)
    float area = 0;
    int a = edge_link[0];
    for (int i = 1; i < edge_link.size() - 1; ++i) {
      int b = edge_link[i];
      int c = edge_link[i+1];

      float ab[3], ac[3];
      for (int j = 0; j < 3; ++j) {
	ab[j] = circumcenters[3*b + j] - circumcenters[3*a + j];
	ac[j] = circumcenters[3*c + j] - circumcenters[3*a + j];
      }
      float cp[3];
      cross(cp, ab, ac);
      area += sqrt(norm(cp))/2;
    }

    // distance between u and v
    float dist = distance(&particles[3*u], &particles[3*v]);
    vol += area*dist/6;
  }

  return vol;
}
