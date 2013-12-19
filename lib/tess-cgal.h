#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Delaunay_triangulation_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

typedef CGAL::Triangulation_vertex_base_with_info_3<unsigned, K>    Vb;
typedef CGAL::Triangulation_data_structure_3<Vb>                    Tds;
//Use the Fast_location tag. Default or Compact_location works too.
//typedef CGAL::Delaunay_triangulation_3<K, Tds, CGAL::Fast_location> Delaunay3D;
typedef CGAL::Delaunay_triangulation_3<K, Tds>	    Delaunay3D;
//typedef CGAL::Delaunay_triangulation_3<K>           Delaunay3D;
typedef Delaunay3D::Point                           Point;
typedef Delaunay3D::Vertex_handle                   Vertex_handle;
typedef Delaunay3D::Cell_handle                     Cell_handle;
typedef K::FT                                       RealValue;

typedef Delaunay3D::Finite_vertices_iterator        Vertex_iterator;
typedef Delaunay3D::Finite_edges_iterator           Edge_iterator;
typedef Delaunay3D::Finite_facets_iterator          Facet_iterator;
typedef Delaunay3D::Finite_cells_iterator           Cell_iterator;
typedef Delaunay3D::Cell_circulator                 Cell_circulator;
typedef Delaunay3D::Facet_circulator                Facet_circulator;


int gen_voronoi_output(Delaunay3D &Dt, struct vblock_t *vblock,
		       int num_particles);
int gen_delaunay_output(Delaunay3D &Dt, struct vblock_t *vblock,
                        int *gids, int *nids, unsigned char *dirs,
                        struct remote_ic_t *rics, int lid, int num_recvd);

void construct_delaunay(Delaunay3D &Dt, int num_particles, float *particles);
