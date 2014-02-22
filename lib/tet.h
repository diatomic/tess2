#ifndef _TESS_TET_H
#define _TESS_TET_H

struct tet_t {
  int verts[4];		// indices of the vertices
  int tets[4];		// indices of the neighbors
			// tets[i] lies opposite verts[i]
};

// Public
#ifdef __cplusplus
extern "C"
#endif
int find(struct tet_t* tet, int v);

#ifdef __cplusplus
extern "C"
#endif
void circumcenter(float* c, struct tet_t* tet, float* particles);

#ifdef __cplusplus
extern "C"
#endif
int circulate_start(struct tet_t* tets, int t, int x, int y);

#ifdef __cplusplus
extern "C"
#endif
void circulate_next(int* next_t, int* next_v, struct tet_t* tets, int t, int v, int x, int y);

#ifdef __cplusplus
extern "C"
#endif
float distance(float* u, float* v);

// Private
#ifdef __cplusplus
extern "C"
#endif
void cross(float* res, float* u, float* v);

#ifdef __cplusplus
extern "C"
#endif
float norm(float* x);

#ifdef __cplusplus
extern "C"
#endif
float determinant(float* t, float* u, float* v);


#endif	// _TESS_TET_H
