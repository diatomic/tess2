#include "tess/tess.h"
#include "tess/tess-cgal.h"
#include "tess/tet-neighbors.h"
#include <vector>

//----------------------------------------------------------------------------
// Initialize and destroy Delaunay data structures used by CGAL.
// We keep them persistent for later incremental insertion of additional points.
// 
void* init_delaunay_data_structures(int nblocks)
{
  return new Delaunay3D[nblocks];
}

void init_delaunay_data_structure(dblock_t* b)
{
  b->Dt = static_cast<void*>(new Delaunay3D);
}

void clean_delaunay_data_structures(void* ds)
{
  Delaunay3D* dds = (Delaunay3D*) ds;
  delete[] dds;
}
void clean_delaunay_data_structure(dblock_t* b)
{
  delete static_cast<Delaunay3D*>(b->Dt);
}
//----------------------------------------------------------------------------
//
//  creates local delaunay cells in one block
//
//  b: local block
//
void local_cells(struct dblock_t *b)
{
  Delaunay3D* d = (Delaunay3D*)b->Dt;
  construct_delaunay(*d, b->num_particles, b->particles);
  int ntets =  d->number_of_finite_cells();
  b->num_tets = ntets;
  b->tets = (struct tet_t*)malloc(ntets * sizeof(struct tet_t));
  gen_tets(*d, b->tets);
  fill_vert_to_tet(b);
}
//----------------------------------------------------------------------------
//
// generates delaunay output from qhull
//
// facetlist: qhull list of convex hull facets
// tet_verts: pointer to array of tet vertex indeices for this block
//
// returns: number of tets found
//
int gen_delaunay_output(Delaunay3D &Dt, int** tet_verts) {

  int numfacets = Dt.number_of_finite_cells();
  int v = 0; // index in tets

  *tet_verts = (int *)malloc(numfacets * 4 * sizeof(int));

  // process the tets
  for(Cell_iterator cit = Dt.finite_cells_begin(); 
      cit != Dt.finite_cells_end(); ++cit) {

    for (int i = 0; i < 4; ++i)
      (*tet_verts)[v++] = cit->vertex(i)->info();

  }
  
  assert(numfacets == v / 4); // sanity

  return numfacets;

}
//----------------------------------------------------------------------------
//
//    compute Delaunay
//
void construct_delaunay(Delaunay3D &Dt, int num_particles, float *particles)
{
  int n = Dt.number_of_vertices();

#ifdef TESS_CGAL_ALLOW_SPATIAL_SORT
  std::vector< std::pair<Point,unsigned> > points; points.reserve(num_particles);
  for (unsigned j = n; j < (unsigned)num_particles; j++)
  {
    Point p(particles[3*j],
	    particles[3*j+1],
	    particles[3*j+2]);
    points.push_back(std::make_pair(p,j));
  }
  Dt.insert(points.begin(), points.end());
#else
  for (unsigned j = n; j < (unsigned)num_particles; j++)
  {
    Point p(particles[3*j],
	    particles[3*j+1],
	    particles[3*j+2]);
    Dt.insert(p)->info() = j;
  }
#endif
}
//----------------------------------------------------------------------------
//
// Convert Delaunay3D to a vector of tets
//
void gen_tets(Delaunay3D& Dt, tet_t* tets)
{
  // Initialize all cell info to -1 (infinite cells will keep -1)
  for(All_cell_iterator cit = Dt.all_cells_begin();
			cit != Dt.all_cells_end(); ++cit)
    cit->info() = -1;
  
  // Record tet vertices (and indices in info())
  int idx = 0;
  for(Cell_iterator cit = Dt.finite_cells_begin();
		    cit != Dt.finite_cells_end(); ++cit)
  {
    cit->info() = idx;		// record indices in cells' info()
    for (int i = 0; i < 4; ++i)
      tets[idx].verts[i] = cit->vertex(i)->info();
    ++idx;
  }
  
  // Record tet adjacency information
  idx = 0;
  for(Cell_iterator cit = Dt.finite_cells_begin();
		    cit != Dt.finite_cells_end(); ++cit)
  {
    for (int i = 0; i < 4; ++i)
      tets[idx].tets[i] = cit->neighbor(i)->info();
    ++idx;
  }
}
//----------------------------------------------------------------------------
