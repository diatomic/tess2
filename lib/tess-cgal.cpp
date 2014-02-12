#include "tess.h"
#include "tess-cgal.h"
#include <vector>

//----------------------------------------------------------------------------
// Initialize and destroy Delaunay data structures used by CGAL.
// We keep them persistent for later incremental insertion of additional points.
// 
void* init_delaunay_data_structures(int nblocks)
{
  return new Delaunay3D[nblocks];
}

void clean_delaunay_data_strucutres(void* ds)
{
  Delaunay3D* dds = (Delaunay3D*) ds;
  delete[] dds;
}

//----------------------------------------------------------------------------
//
//   creates local voronoi cells
//
//   nblocks: number of blocks
//   tblocks: pointer to array of temporary vblocks
//   dim: number of dimensions (eg. 3)
//   num_particles: number of particles in each block
//   particles: particles in each block, particles[block_num][particle]
//   where each particle is 3 values, px, py, pz
//   times: timing
//   ds: the delaunay data structures
//
void local_cells(int nblocks, struct vblock_t *tblocks, int dim,
		 int *num_particles, float **particles, void* ds) {

  int i,j;

  unsigned total = 0;
  Delaunay3D* Dts = (Delaunay3D*) ds;

  /* for all blocks */
  for (i = 0; i < nblocks; i++) {
    Delaunay3D& Dt = Dts[i];
    construct_delaunay(Dt, num_particles[i], particles[i]);
    total += num_particles[i];
    // std::cout << "Num particles (local): " << num_particles[i] << std::endl;

    /* allocate number of verts for original particles */
    tblocks[i].num_cell_verts = (int *)malloc(sizeof(int) * num_particles[i]);
    memset(tblocks[i].num_cell_verts, 0, sizeof(int) * num_particles[i]);

    /* process voronoi output */
    gen_voronoi_output(Dt, &tblocks[i], num_particles[i]);

    /* allocate cell sites for original particles */
    tblocks[i].num_orig_particles = num_particles[i];
    tblocks[i].sites =
      (float *)malloc(3 * sizeof(float) * tblocks[i].num_orig_particles);
    for (j = 0; j < tblocks[i].num_orig_particles; j++) {
      tblocks[i].sites[3 * j]     = particles[i][3 * j];
      tblocks[i].sites[3 * j + 1] = particles[i][3 * j + 1];
      tblocks[i].sites[3 * j + 2] = particles[i][3 * j + 2];
    }
  } /* for all blocks */

  //std::cout << "Total particles in local_cells(): " << total << std::endl;
}
//----------------------------------------------------------------------------
//
//   creates original voronoi cells
//
//   nblocks: number of blocks
//   vblocks: pointer to array of vblocks
//   dim: number of dimensions (eg. 3)
//   num_particles: number of particles in each block
//   num_orig_particles: number of original particles in each block, before any
//   neighbor exchange
//   particles: particles in each block, particles[block_num][particle]
//   where each particle is 3 values, px, py, pz
//   gids: global block ids of owners of received particles in each of my blocks
//   nids: native particle ids of received particles in each of my blocks
//   dirs: wrapping directions of received particles in each of my blocks
//   times: timing
//   ds: the delaunay data structures
//
void orig_cells(int nblocks, struct vblock_t *vblocks, int dim,
		int *num_particles, int *num_orig_particles, 
		float **particles, int **gids, int **nids, 
		unsigned char **dirs, double *times,
		void* ds) {

  int num_recvd; // number of received particles in current block
  int i,j;
  unsigned total = 0;
  Delaunay3D* Dts = (Delaunay3D*) ds;

  // is_complete status of received particles 
  struct remote_ic_t **rics =
    (struct remote_ic_t **)malloc(nblocks * 
				  sizeof(struct remote_ic_t *));
  // delaunay vertices
  int **tet_verts = (int **)malloc(nblocks * sizeof(int *));
  int *num_tets = (int *)malloc(nblocks * sizeof(int));
  for (i = 0; i < nblocks; i++) {
    tet_verts[i] =  NULL;
    num_tets[i] = 0;
  }

  // for all blocks 
  for (i = 0; i < nblocks; i++) {

    // number of received particles
    num_recvd = num_particles[i] - num_orig_particles[i];

    Delaunay3D& Dt = Dts[i];
    construct_delaunay(Dt, num_particles[i], particles[i]);
    // std::cout << "Num particles (orig): " << num_particles[i] << std::endl;
    total += num_particles[i];

    /* allocate number of verts for original particles */
    vblocks[i].num_cell_verts = (int *)malloc(sizeof(int) * num_particles[i]);
    memset(vblocks[i].num_cell_verts, 0, sizeof(int) * num_particles[i]);

    // process voronoi output
    gen_voronoi_output(Dt, &vblocks[i], num_particles[i]);

    // allocate cell sites for original particles
    vblocks[i].num_orig_particles = num_orig_particles[i];
    vblocks[i].sites =
      (float *)malloc(3 * sizeof(float) * vblocks[i].num_orig_particles);
    for (j = 0; j < vblocks[i].num_orig_particles; j++) {
      vblocks[i].sites[3 * j]     = particles[i][3 * j];
      vblocks[i].sites[3 * j + 1] = particles[i][3 * j + 1];
      vblocks[i].sites[3 * j + 2] = particles[i][3 * j + 2];
    }

    // allocate lookup table for cells completion status
    vblocks[i].is_complete = 
      (unsigned char *)malloc(vblocks[i].num_orig_particles);

    // determine complete cells
    complete_cells(&vblocks[i], i);

    // process delaunay output
    num_tets[i] = gen_delaunay_output(Dt, &tet_verts[i]);

    // TODO: surely this can be optimized
    // connectivity of faces in voronoi cells 
    cell_faces(&vblocks[i]);

  } // for all blocks 

  // exchange complete cell status for exchanged particles
  neighbor_is_complete(nblocks, vblocks, rics);

  // convert delaunay output to vblock for all blocks
  for (i = 0; i < nblocks; i++)
    gen_tets(tet_verts[i], num_tets[i], &vblocks[i], gids[i], nids[i], 
	     dirs[i], rics[i], i, num_particles[i] - num_orig_particles[i]);

  // cleanup
  for (i = 0; i < nblocks; i++) {
    free(rics[i]);
    if (tet_verts[i])
      free(tet_verts[i]);
  }
  free(rics);
  free(tet_verts);
  free(num_tets);

}
//----------------------------------------------------------------------------
//
//   generates voronoi output from CGAL
//
//   Dt: CGAL's Delaunay3D structure
//   vblock: pointer to one voronoi block, allocated by caller
//   num_particles: number of particles used to generate the tessellation
//   side effects: allocates data structures inside of vblock, caller's
//   responsibility to free
//
//   returns: number of cells found (<= original number of particles)
//
int gen_voronoi_output(Delaunay3D &Dt, struct vblock_t *vblock, 
		       int num_particles) {

  int i,j;

  vblock->num_verts = Dt.number_of_finite_cells() + 1;
  int temp_num_cells = Dt.number_of_vertices();

  /* vertices */
  std::map<Cell_handle, int> tet_indices;	// TODO: perhaps, replace with something better
  vblock->verts = (double *)malloc(sizeof(double) * 3 * vblock->num_verts);
  vblock->verts[0] = vblock->verts[1] = vblock->verts[2] = std::numeric_limits<double>::infinity();
  i = 1; /* already did the infinity vertex, index 0 */
  for(Cell_iterator cit = Dt.finite_cells_begin(); cit != Dt.finite_cells_end(); ++cit)
  {
      Point center = cit->circumcenter(Dt.geom_traits());
      vblock->verts[3 * i]     = center.x();
      vblock->verts[3 * i + 1] = center.y();
      vblock->verts[3 * i + 2] = center.z();
      tet_indices[cit] = i;
      i++;
  }

  /*
     order Vertex_iterators in the order of original particles
     (CGAL switches the order of the points via a spatial sort)
  */
  std::vector< std::pair<unsigned, Vertex_handle> > vertices;
  for(Vertex_iterator vit = Dt.finite_vertices_begin(); vit != Dt.finite_vertices_end(); ++vit)
      vertices.push_back(std::make_pair(vit->info(), vit));
  std::sort(vertices.begin(), vertices.end());

  // DEPRECATED, malloc moved to calling function instead TP
//   vblock->num_cell_verts = (int *)malloc(sizeof(int) * num_particles);
//   memset(vblock->num_cell_verts, 0, sizeof(int) * num_particles);

  /* number of vertices in each cell; size is number of particles; 
     if a cell is skipped, the number of vertices will be 0 */
  int cell = 0; /* index of cell being processed */
  for(unsigned k = 0; k < vertices.size(); ++k)
  {
    Vertex_handle v = vertices[k].second;
    std::vector<Cell_handle> cell_vertices;	    // Delaunay cells are Voronoi vertices
    Dt.incident_cells(v, std::back_inserter(cell_vertices));

    int num_infinite = 0;
    for (j = 0; j < cell_vertices.size(); ++j)
      if (Dt.is_infinite(cell_vertices[j]))
	++num_infinite;
    vblock->num_cell_verts[cell] = cell_vertices.size();
    if (num_infinite > 1)
      vblock->num_cell_verts[cell] -= (num_infinite - 1);
    
    ++cell;
  }

  /* allocate the cell vertices */
  vblock->tot_num_cell_verts = 0;
/*   for (i = 0; i < temp_num_cells; i++) */
  for (i = 0; i < num_particles; i++)
    vblock->tot_num_cell_verts += vblock->num_cell_verts[i];
  vblock->cells = (int *)malloc(sizeof(int) * vblock->tot_num_cell_verts);

  /* cell vertices */
  i = 0;
  for(unsigned k = 0; k < vertices.size(); ++k)
  {
    Vertex_handle v = vertices[k].second;
    std::vector<Cell_handle> cell_vertices;	    // Delaunay cells are Voronoi vertices
    Dt.incident_cells(v, std::back_inserter(cell_vertices));

    bool seen_infinite = false;
    for (j = 0; j < cell_vertices.size(); ++j)
    {
      if (Dt.is_infinite(cell_vertices[j]))
      {
	if (!seen_infinite)
	{
	  vblock->cells[i++] = 0;
	  seen_infinite = true;
	}
      } else
        vblock->cells[i++] = tet_indices[cell_vertices[j]];
    }
  }

  /* voronoi faces */
  int tot_faces = Dt.number_of_finite_edges();
  vblock->faces = (struct vface_t*)malloc(tot_faces * sizeof(struct vface_t));
  memset(vblock->faces, 0, tot_faces * sizeof(struct vface_t));
  int num_faces = 0;

  for(Edge_iterator eit = Dt.finite_edges_begin(); eit != Dt.finite_edges_end(); ++eit)
  {
    Cell_handle c = eit->first;
    Vertex_handle v0 = c->vertex(eit->second);
    Vertex_handle v1 = c->vertex(eit->third);
    vblock->faces[num_faces].cells[0]  = v0->info();
    vblock->faces[num_faces].cells[1]  = v1->info();

    int num_verts = 0;
    Cell_circulator begin = Dt.incident_cells(*eit);
    Cell_circulator cur = begin;
    bool seen_infinite = false;
    do
    {
      if (Dt.is_infinite(cur))
      {
	if (!seen_infinite)
	{
	  vblock->faces[num_faces].verts[num_verts++] = 0;
	  seen_infinite = true;
	}
      } else
	vblock->faces[num_faces].verts[num_verts++] = tet_indices[cur];
      ++cur;
    } while (cur != begin);
    vblock->faces[num_faces].num_verts = num_verts;
    ++num_faces;
  }

  vblock->num_faces = num_faces;
  assert(vblock->num_faces == tot_faces); /* sanity */

  return temp_num_cells;

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
  for (unsigned j = n; j < num_particles; j++)
  {
    Point p(particles[3*j],
	    particles[3*j+1],
	    particles[3*j+2]);
    points.push_back(std::make_pair(p,j));
  }
  Dt.insert(points.begin(), points.end());
#else
  for (unsigned j = n; j < num_particles; j++)
  {
    Point p(particles[3*j],
	    particles[3*j+1],
	    particles[3*j+2]);
    Dt.insert(p)->info() = j;
  }
#endif
}
//----------------------------------------------------------------------------
