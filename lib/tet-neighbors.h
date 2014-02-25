// C++ only header to use vector

#include <vector>

bool neighbor_edges(std::vector< std::pair<int, int> >& nbrs,
		    int	    v,
		    tet_t*  tets,
		    int	    t
		   );

bool neighbor_tets(std::vector<int>& nbrs,
		   int	    v,
		   tet_t*  tets,
		   int	    t
		  );

bool complete(int	    v,
	      tet_t*	    tets,
	      int	    t
	     );
