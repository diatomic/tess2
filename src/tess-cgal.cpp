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

void clean_delaunay_data_structures(void* ds)
{
  Delaunay3D* dds = (Delaunay3D*) ds;
  delete[] dds;
}

//----------------------------------------------------------------------------
//
//  creates local delaunay cells
//
//  nblocks: number of blocks
//  dblocks: pointer to array of dblocks
//  dim: number of dimensions (eg. 3)
//  ds: the delaunay data structures
//
void local_cells(int nblocks, struct dblock_t *dblocks, void *ds) {

  Delaunay3D* Dts = (Delaunay3D*) ds;

  // for all blocks
  for (int i = 0; i < nblocks; i++) {

    // Explicit check for adjacent duplicates (debug only)
    //for (int j = 0; j < dblocks[i].num_particles - 1; ++j) {
    //  int k = j + 1;
    //  if (dblocks[i].particles[3*j]     == dblocks[i].particles[3*k] &&
    //      dblocks[i].particles[3*j + 1] == dblocks[i].particles[3*k + 1] &&
    //      dblocks[i].particles[3*j + 2] == dblocks[i].particles[3*k + 2]) {
    //    fprintf(stderr, "Warning in local_dcells(): identical particles %d and %d\n", j, k);
    //  }
    //}

    Delaunay3D& Dt = Dts[i];
    construct_delaunay(Dt, dblocks[i].num_particles, dblocks[i].particles);

    // fill the tets
    int ntets =  Dt.number_of_finite_cells();
    dblocks[i].num_tets = ntets;
    dblocks[i].tets = (struct tet_t*)malloc(ntets * sizeof(struct tet_t));
    gen_tets(Dt, dblocks[i].tets);
    
    fill_vert_to_tet(&dblocks[i]);

    // Explicit check for an uninitialized vertex (debug only)
    //for (int p = 0; p < dblocks[i].num_particles; ++p)
    //  if (dblocks[i].vert_to_tet[p] == -1) {
    //    fprintf(stderr, "Vertex %d (%f %f %f) uninitialized in vert_to_tet\n", p,
    //                    dblocks[i].particles[3*p],
    //                    dblocks[i].particles[3*p + 1],
    //                    dblocks[i].particles[3*p + 2]);
    //  }

  }

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
