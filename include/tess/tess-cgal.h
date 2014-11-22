#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/Delaunay_triangulation_3.h>

#include "tet.h"
#include <vector>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

typedef CGAL::Triangulation_vertex_base_with_info_3<unsigned, K>    Vb;
typedef CGAL::Triangulation_cell_base_with_info_3<int, K>	    Cb;
typedef CGAL::Triangulation_data_structure_3<Vb,Cb>                 Tds;
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
typedef Delaunay3D::All_cells_iterator		    All_cell_iterator;
typedef Delaunay3D::Cell_circulator                 Cell_circulator;
typedef Delaunay3D::Facet_circulator                Facet_circulator;


int gen_voronoi_output(Delaunay3D &Dt, struct vblock_t *vblock,
		       int num_particles);
int gen_delaunay_output(Delaunay3D &Dt, int **tet_verts);
void construct_delaunay(Delaunay3D &Dt, int num_particles, float *particles);

void gen_tets(Delaunay3D& Dt, tet_t* tets);


#include <diy/serialization.hpp>

namespace diy
{
  template<>
  struct Serialization<Delaunay3D>
  {
    static void	    save(BinaryBuffer& bb, const Delaunay3D& Dt)
    {
      std::ostringstream out;
      CGAL::set_mode(out, CGAL::IO::BINARY);
      out << Dt;

      size_t s = out.str().size();
      diy::save(bb, s);
      bb.save_binary(out.str().c_str(), out.str().size());
    }

    static void	    load(BinaryBuffer& bb, Delaunay3D& Dt)
    {
      size_t s;
      diy::load(bb, s);

      // This is not pretty, but portable.
      // Double copying is annoying. Perhaps, it's worth implementing an
      // iostream wrapper around BinaryBuffer.
      std::vector<char> in_vec(s);
      bb.load_binary(&in_vec[0], s);

      std::string in_str(in_vec.begin(), in_vec.end());
      std::istringstream in(in_str);
      CGAL::set_mode(in, CGAL::IO::BINARY);
      in >> Dt;

      // NB: this shouldn't be necessary, but CGAL doesn't save vertex info,
      //     so we reset it here. This works because we don't need vertex ids to
      //     be consistent across iterations.
      unsigned idx = 0;
      for(Vertex_iterator vit = Dt.finite_vertices_begin(); vit != Dt.finite_vertices_end(); ++vit)
	vit->info() = idx++;
    }
  };
}
