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

// returns int instead of bool so that C can call complete
#ifdef __cplusplus
extern "C"
#endif
int complete(int	    v,
	      tet_t*	    tets,
	      int	    t
	     );

void fill_edge_link(std::vector<int>&	edge_link,
		    int			v,
		    int			u,
		    int			ut,
		    tet_t*		tets);

void skip_tet(struct tet_t *tet);

int is_skipped_tet(struct tet_t *tet);
