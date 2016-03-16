#include <cassert>
#include <cmath>
#include <cstdio>

#include <set>
#include <queue>

#include "tess/tet.h"
#include "tess/tet-neighbors.h"

// finds index of v in tet->verts
int find(tet_t* tet, int v)
{
  for (int i = 0; i < 4; ++i)
    if (tet->verts[i] == v)
      return i;
  assert(0);
  return -1;
}

// dot product: u * (p - x)
float dot(float* u, float* p, float* x)
{
  float res = 0;
  for (int i = 0; i < 3; ++i)
    res += u[i] * (p[i] - x[i]);
  return res;
}

/**
 * computes circumcenter of a tetrahedron
 *
 * center:	array of 3 floats that will store the result
 * tet:		the tetrahedron
 * particles:	array of particles (x1,y1,z1,x2,y2,z2,...)
 */
void circumcenter(float* center, tet_t* tet, float* particles)
{
  float *a = &particles[3*tet->verts[0]],
	*b = &particles[3*tet->verts[1]],
	*c = &particles[3*tet->verts[2]],
	*d = &particles[3*tet->verts[3]];

  // center everything at d
  float t[3], u[3], v[3];
  for (int i = 0; i < 3; ++i) {
    t[i] = a[i] - d[i];
    u[i] = b[i] - d[i];
    v[i] = c[i] - d[i];
  }
  float norm_t = norm(t),
	norm_u = norm(u),
	norm_v = norm(v);

  // God help us if den is close to 0
  // (this will happen for nearly flat tets)
  float den = 2*determinant(t,u,v);

  float uv[3], vt[3], tu[3];
  cross(uv, u, v);
  cross(vt, v, t);
  cross(tu, t, u);

  for (int i = 0; i < 3; ++i)
    center[i] = d[i] + (norm_t*uv[i] + norm_u*vt[i] + norm_v*tu[i])/den;
}

// determine if any point in the box [min, max] lies on the opposite side of the
// facet opposite to vertex j
int side_of_plane(float* min, float* max, struct tet_t* tet, float* particles, int j)
{
  float one[3] = { 1., 1., 1. };

  int idx[4] = { 0, 1, 2, 3 };
  std::swap(idx[j], idx[3]);
  
  float *x = &particles[3*tet->verts[idx[0]]],
	*y = &particles[3*tet->verts[idx[1]]],
	*z = &particles[3*tet->verts[idx[2]]],
	*p = &particles[3*tet->verts[idx[3]]];

  float u[3], v[3], w[3], n[3];
  for (int i = 0; i < 3; ++i)
  {
    u[i] = y[i] - x[i];
    v[i] = z[i] - x[i];
  }
  cross(n, u, v);

  // plane through x, normal to n
  float res = dot(n, p, x);
  int sign = 0;
  if (res > 0)
    sign = 1;
  else if (res < 0)
    sign = -1;

  if (sign == 0)
    fprintf(stderr, "Warning: got a degenerate convex hull tet in side_of_plane computation\n");

  // find the sign of the extreme-most point of the box
  res = 0;
  for (int i = 0; i < 3; ++i)
  {
    if (sign ^ (n[i] < 0))
      res += n[i] * (min[i] - x[i]);
    else
      res += n[i] * (max[i] - x[i]);
  }

  return (sign ^ (res > 0));
}

// returns |x|^2
float norm(float* x)
{
  float res = 0;
  for (int i = 0; i < 3; ++i)
    res += x[i]*x[i];
  return res;
}

// res = u x v
void cross(float* res, float* u, float* v)
{
  res[0] = u[1]*v[2] - u[2]*v[1];
  res[1] = u[2]*v[0] - u[0]*v[2];
  res[2] = u[0]*v[1] - u[1]*v[0];
}

// returns determinant of |t u v|
float determinant(float* t, float* u, float* v)
{
  return t[0]*u[1]*v[2] + u[0]*v[1]*t[2] + v[0]*t[1]*u[2] -
         v[0]*u[1]*t[2] - u[0]*t[1]*v[2] - t[0]*v[1]*u[2];
}

// returns |u - v|
float distance(float* u, float* v)
{
  float n = 0;
  for (int i = 0; i < 3; ++i) {
    n += (u[i] - v[i])*(u[i] - v[i]);
  }
  return sqrt(n);
}

/**
 * find initial vertex to circulate from, tets[t] must contain edge (x,y)
 *
 * tets:    array of all tetrahedra
 * t:	    index into tets (initial tetrahedron)
 * x,y:	    edge we are circulating around
 *
 * returns: index into tets[t]->verts (neither x, nor y)
 */
int circulate_start(tet_t* tets, int t, int x, int y)
{
  for (int i = 0; i < 4; ++i) {
    int v = tets[t].verts[i];
    if (v != x && v != y)
      return i;
  }
  assert(0);	    // can't get to this point
  return -1;
}

/**
 * find the next tetrahedron around edge (x,y) and the next vertex in it
 *
 * next_t:  will store the index of the next tetrahedron
 * next_v:  will store the index of the next vertex
 * tets:    array of all tetrahedra
 * t:	    index into tets (current tetrahedron)
 * v:	    index of the current vertex in tets->verts
 * x,y:	    edge we are circulating around
 */
void circulate_next(int* next_t, int* next_v, tet_t* tets, int t, int v, int x, int y)
{
  tet_t* cur  = tets + t;

  // Find the unused vertex (i.e., the entry in tets[t]->verts that's not x,y, or tets[t]->verts[v])
  int nv = -1;
  for (int i = 0; i < 4; ++i) {
    int inv = cur->verts[i];
    if (i   != v &&
	inv != x &&
	inv != y) {
      nv = inv;
      break;
    }
  }
  assert(nv != -1);

  // take a step
  *next_t     = cur->tets[v];
  tet_t* next = tets + *next_t;

  // find the index of nv in next->verts
  for (int i = 0; i < 4; ++i) {
    int inv = next->verts[i];
    if (inv == nv) {
      *next_v = i;
      return;
    }
  }
  assert(false);	// must have found everything
}

/**
 * fills a vector of u, ut pairs with Delaunay neighbors and tetratehedra
 * that conain edge (v,u)
 *
 * nbrs:    vector of pairs (u, ut)
 * v:	    vertex
 * tets:    array of all tetrahedra
 * t:	    tet that contains v
 *
 * returns whether the Voronoi cell is finite
 */
bool neighbor_edges(std::vector< std::pair<int, int> >& nbrs,
	       int	    v,
	       tet_t*	    tets,
	       int	    t
	      )
{
  bool finite = true;
  std::queue<int>   q;
  std::set<int>	    visited_tets,
		    visited_verts;

  q.push(t);

  // BFS in the star of v
  while (!q.empty())
  {
    int t = q.front();
    q.pop();

    // already visited, continue
    if (visited_tets.find(t) != visited_tets.end())
      continue;
    visited_tets.insert(t);

    // insert vertices, not yet inserted, and queue neighbors
    for (int i = 0; i < 4; ++i) {
      int u = tets[t].verts[i];
      if (u != v) {
	if (visited_verts.find(u) == visited_verts.end()) {
	  visited_verts.insert(u);
	  nbrs.push_back(std::make_pair(u,t));
	}
	int next = tets[t].tets[i];
	if (next == -1)
	  finite = false;
	else
	  q.push(next);
      }
    }
  }

  return finite;
}

/**
 * fills a vector with tetrahedra that contain the given vertex
 *
 * nbrs:    vector to be filled with tet indices
 * v:	    vertex
 * tets:    array of all tetrahedra
 * t:	    tet that contains v
 *
 * returns whether the Voronoi cell is finite
 */
bool neighbor_tets(std::vector<int>& nbrs,
		   int	    v,
		   tet_t*   tets,
                   int      num_tets,
		   int	    t
		  )
{
  if (t < 0)
      fprintf(stderr, "Warning in neighbor_tets(): cannot start with negative tet %d\n", t);

  bool finite = true;
  std::queue<int>   q;
  std::set<int>	    visited_tets;
  q.push(t);

  // BFS in the star of v
  while (!q.empty())
  {
    int t = q.front();
    q.pop();

    if (t >= num_tets)
        fprintf(stderr, "Warning in neighbor_tets(): tet %d exceeds total %d tets\n", t, num_tets);

    // already visited, continue
    if (visited_tets.find(t) != visited_tets.end())
      continue;
    visited_tets.insert(t);
    nbrs.push_back(t);

    // queue neighbors
    for (int i = 0; i < 4; ++i) {
      int u = tets[t].verts[i];
      if (u != v) {
	int next = tets[t].tets[i];
	if (next == -1)
	  finite = false;
	else
	  q.push(next);
      }
    }
  }

  return finite;
}

/**
 * determines if v's Voronoi cell is finite
 * (equivalently, whether v lies on the convex hull)
 * returning int instead of bool so that C can call complete
 *
 * v:	    vertex
 * tets:    array of all tetrahedra
 * t:	    tet that contains v
 */
int complete(int	    v,
	      tet_t*	    tets,
              int           num_tets,
	      int	    t
	     )
{
  if (t < 0)
      fprintf(stderr, "Warning in complete(): cannot start with a negative tet %d\n", t);

  std::queue<int>   q;
  std::set<int>	    visited_tets;
  q.push(t);

  // BFS in the star of v
  while (!q.empty())
  {
    int t = q.front();
    q.pop();

    if (t >= num_tets)
        fprintf(stderr, "Warning in complete(): tet %d exceeds total %d tets\n", t, num_tets);

    // already visited, continue
    if (visited_tets.find(t) != visited_tets.end())
      continue;
    visited_tets.insert(t);

    // queue neighbors
    for (int i = 0; i < 4; ++i) {
      int u = tets[t].verts[i];
      if (u != v) {
	int next = tets[t].tets[i];
	if (next == -1)
	  return 0; // returning int instead of bool
	else
	  q.push(next);
      }
    }
  }

  return 1; // returning int instead of bool
}

/**
 * Fills edge_link with all the tets containing edge (u,v),
 * the tets are inserted in circular order, starting from ut
 *
 * edge_link:	    output vector
 * v,u:		    vertices
 * ut:		    a tet that contains edge (v,u)
 * tets:	    array of tets
 */
void fill_edge_link(std::vector<int>&	edge_link,
		    int			v,
		    int			u,
		    int			ut,
		    tet_t*		tets)
{
  int wi = circulate_start(tets, ut, v, u);
  int t  = ut;
  while (true)
  {
    edge_link.push_back(t);

    int next_t, next_wi;
    circulate_next(&next_t, &next_wi, tets, t, wi, v, u);
    if (next_t == ut || next_t == -1)
      break;

    t  = next_t;
    wi = next_wi;
  }
}
