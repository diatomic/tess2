//---------------------------------------------------------------------------
//
// voronoi rendering
//
// Tom Peterka
// Argonne National Laboratory
// 9700 S. Cass Ave.
// Argonne, IL 60439
// tpeterka@mcs.anl.gov
//
// (C) 2011 by Argonne National Laboratory.
// See COPYRIGHT in top-level directory.
//
//--------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "delaunay.h"
#include "voronoi.h"
#include "ser_io.hpp"
#include <math.h>

#if defined(MAC_OSX)
#include <GLUT/glut.h> 
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/glut.h> 
#include <GL/gl.h>
#endif

#define TET // new delaunay tetrahedra data model

// pnetcdf not implemented for tet data model yet
#ifndef TET
#define PNETCDF_IO
#endif

#define SPHERE_RAD_FACTOR .002 // used to compute sphere radius
// #define PAPER // color scheme for paper (white backgound)


#ifndef SERIAL_IO
#include "mpi.h"
#include "io.h"
#endif

using namespace std;

// 3d point or vector
struct vec3d {
  float x, y, z;
};

// 2d point or vector
struct vec2d {
  float x, y;
};

// color
struct rgba {
  float r, g, b, a;
};

// mouse button state
int press_x, press_y; 
int release_x, release_y; 

// rotate
vec2d rot = {0.0, 0.0};
float rot_rate = 0.2;

// scale
float scale = 1.0; 
float scale_rate = 0.01;
vec2d aspect; // scaling due to window aspect ratio

// near clip plane
float near = 0.1;

// window size
// vec2d win_size = {1024, 512};
vec2d win_size = {1024, 1024};
// vec2d win_size = {512, 512};

// previous window size
vec2d old_win_size;

// translate
vec2d trans = {0.0, 0.0};
float trans_rate = 0.01;

// transform mode
int xform_mode = 0; 
bool block_mode = false;
#define XFORM_NONE    0 
#define XFORM_ROTATE  1
#define XFORM_SCALE   2 
#define XFORM_TRANS   3

// rendering mode
bool draw_fancy = false;
bool draw_particle = true;
bool color_density = false;
bool draw_tess = false;
bool draw_del = false;

// volume filtering
float min_vol = 0.0; // desired min vol threshold
float max_vol = 0.0; // desired max vol threshold
float min_vol_act = 0.0; // actual min vol we have
float max_vol_act = 0.0; // actual max vol we have
float min_vol_clamp = 1.0e-6; // clamp min_vol_act to this value
float vol_step = 0.001;

vec3d sizes; // individual data sizes in each dimension
float size; // one overall number for data size, max of individual sizes
float sphere_rad; // sphere radius
float clip = 0.0; // clipping faces at this fraction of size (0.0-1.0)
float z_clip; // z value of clipping (only in z for now)

// voronoi sites
vector<vec3d> sites;
vec3d site_min;
vec3d site_max;
vec3d site_center;

// voronoi vertices, faces, cells, normals
vector<vec3d> verts;
vector<int> num_face_verts;
vector<vec3d> vor_normals;

// delaunay tet vertics, face normals
vector<vec3d> tet_verts;
vector<vec3d> tet_normals;

// volumes associated with faces
vector <float> face_vols;

// global data extents
vec3d data_min, data_max;

// local blocks
#ifdef TET
dblock_t **blocks; // newer delaunay blocks
#else
vblock_t **blocks; // older voronoi and delaunay blocks
#endif
int nblocks;

// general prupose quadrics
GLUquadricObj *q;

// point sprite texture
static GLubyte sprite_intensity[5][5] = {
  {  50,    50,   50,   50,  50,  },
  {  50,   100,  100,  100,  50,  },
  {  50,   100,  255,  100,  50,  },
  {  50,   100,  100,  100,  50,  },
  {  50,    50,   50,   50,  50,  },
};
static GLubyte sprite_rgba[5][5][4];
static GLuint tex;

// function prototypes
void display();
void init_display();
void draw_cube(float *mins, float *maxs, float r, float g, float b) ;
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void key(unsigned char key, int x, int y);
void timer(int val);
void draw_sphere(rgba &color, vec3d &pos, float rad);
void draw_spheres(vector<vec3d> &sites, float rad);
void draw_sprites(vector<vec3d> &sites, float size);
void draw_axes();
void draw_tets();
void reshape(int w, int h);
void init_model();
void init_viewport(bool reset);
void headlight();
void Centroid(vec3d *verts, int num_verts, vec3d &centroid);
void Normal(vec3d *verts, vec3d &normal);
void NewellNormal(vec3d *verts, int num_verts, vec3d &normal);
void filter_volume(float min_vol, float max_vol);
void clip_cells(float z_clip);
void CellBounds(vblock_t *vblock, int cell,
		float *cell_min, float *cell_max, float *centroid);
void PrepRenderingData(int *gid2lid);
void PrepSiteRendering(int &num_sites);
void PrepCellRendering(int &num_visible_cells);
void PrepTetRendering(int &num_loc_tets, int &num_rem_tets, int* gid2lid);
int compare(const void *a, const void *b);

//--------------------------------------------------------------------------

int main(int argc, char** argv) {

  if (argc < 3) {
    fprintf(stderr, "Usage: draw <filename> <swap (0 or 1)>"
	    " [min. volume (optional)] [max volume (optional)]\n");
    exit(0);
  }

  int swap_bytes = atoi(argv[2]);

  if (argc > 3)
    min_vol = atof(argv[3]);

  if (argc > 4)
    max_vol = atof(argv[4]);

  // read the file

#ifdef PNETCDF_IO

  int tot_blocks; // total number of blocks
  int *gids; // block global ids (unused)
  int *num_neighbors; // number of neighbors for each local block (unused)
  int **neighbors; // neighbors of each local block (unused)
  int **neigh_procs; // procs of neighbors of each local block (unused)
  swap_bytes = swap_bytes; // quiet compiler warning, unused w/ pnetcdf
  MPI_Init(&argc, &argv);
  pnetcdf_read(&nblocks, &tot_blocks, &blocks, argv[1], MPI_COMM_WORLD,
	       &gids, &num_neighbors, &neighbors, &neigh_procs);
  MPI_Finalize();
  // mapping of gid to lid
  int gid2lid[nblocks]; 
  for (int b = 0; b < nblocks; b++) {
    for (int g = 0; g < nblocks; g++) {
      if (gids[g] == b) {
	gid2lid[b] = g;
	break;
      }
      assert(g < nblocks); // sanity
    }
  }

#else

  SER_IO *io = new SER_IO(swap_bytes); // io object
  nblocks = io->ReadAllBlocks(argv[1], blocks, false);

  // mapping of gid to lid
  int gid2lid[nblocks]; 
  for (int b = 0; b < nblocks; b++)
    gid2lid[b] = b;

#endif

  // get overall data extent
  for (int i = 0; i < nblocks; i++) {
    if (i == 0) {
      data_min.x = blocks[i]->mins[0];
      data_min.y = blocks[i]->mins[1];
      data_min.z = blocks[i]->mins[2];
      data_max.x = blocks[i]->maxs[0];
      data_max.y = blocks[i]->maxs[1];
      data_max.z = blocks[i]->maxs[2];
    }
    if (blocks[i]->mins[0] < data_min.x)
      data_min.x = blocks[i]->mins[0];
    if (blocks[i]->mins[1] < data_min.y)
      data_min.y = blocks[i]->mins[1];
    if (blocks[i]->mins[2] < data_min.z)
      data_min.z = blocks[i]->mins[2];
    if (blocks[i]->maxs[0] > data_max.x)
      data_max.x = blocks[i]->maxs[0];
    if (blocks[i]->maxs[1] > data_max.y)
      data_max.y = blocks[i]->maxs[1];
    if (blocks[i]->maxs[2] > data_max.z)
      data_max.z = blocks[i]->maxs[2];
  }

  // debug
  fprintf(stderr, "data sizes mins[%.3f %.3f %.3f] maxs[%.3f %.3f %.3f]\n",
	  data_min.x, data_min.y, data_min.z, 
	  data_max.x, data_max.y, data_max.z);

  // package rendering data
  PrepRenderingData(gid2lid);

  // start glut
  glutInit(&argc, argv); 
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH); 
  glutInitWindowSize(win_size.x, win_size.y); 
  glutCreateWindow("Voronoi"); 
  glutDisplayFunc(display); 
  glutTimerFunc(10, timer, 0); 
  glutMouseFunc(mouse); 
  glutMotionFunc(motion);
  glutKeyboardFunc(key); 
  glutReshapeFunc(reshape);
  glutMainLoop(); 

}
//--------------------------------------------------------------------------
//
// rendering
//
void display() {

  static bool first = true;
  int n;

  if (first)
    init_display();
  first = false;

  // set the headlight
  headlight();

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity(); 
  gluPerspective(60.0, 1.0, near, 100.0); 

  glMatrixMode(GL_MODELVIEW); 
  glLoadIdentity(); 
  gluLookAt(0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0); 

  // mouse interactions: pan, rotate, zoom
  glTranslatef(trans.x, trans.y, 0.0);
  glRotatef(rot.x, 0.0, 1.0, 0.0); 
  glRotatef(rot.y, 1.0, 0.0, 0.0); 
  glScalef(scale, scale, scale);

  // center the data in the window
  glTranslatef(-site_center.x, -site_center.y, -site_center.z);

  glEnable(GL_COLOR_MATERIAL);

  // axes
//   draw_axes();

  // block bounds
  if (block_mode) {
    for (int i = 0; i < nblocks; i++)
      draw_cube(blocks[i]->mins, blocks[i]->maxs, 1.0, 0.0, 1.0);
  }

  // delaunay tets
  if (draw_del)
    draw_tets();

  // cell edges
  if (draw_tess) {

    glDisable(GL_LIGHTING);
    glColor4f(0.7, 0.7, 0.7, 1.0);
    if (draw_fancy)
      glLineWidth(2.0);
    else
      glLineWidth(1.0);
    n = 0;

    // for all faces
    for (int i = 0; i < (int)num_face_verts.size(); i++) {

      // scan all vertices to see if the face should be clipped or drawn
      bool draw_face = true;
      if (clip > 0.0) {
	int m = n;
	for (int j = 0; j < num_face_verts[i]; j++) {
	  if (verts[m].z > z_clip) {
	    draw_face = false;
	    break;
	  }
	  m++;
	}
      }

      if (draw_face) {
	glBegin(GL_LINE_STRIP);
	int n0 = n; // index of first vertex in this face
	for (int j = 0; j < num_face_verts[i]; j++) {
	  glVertex3f(verts[n].x, verts[n].y, verts[n].z);
	  n++;
	}
	// repeat first vertex
	glVertex3f(verts[n0].x, verts[n0].y, verts[n0].z);
	glEnd();
      }

      else { // just need to increment n
	for (int j = 0; j < num_face_verts[i]; j++)
	  n++;
      }

    } // for all faces

  } // draw tess

  // sites
  if (draw_fancy) {
    glDisable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHTING);
    GLfloat amb_mat[] = {0.6, 0.6, 0.6, 1.0};
    GLfloat spec_mat[] = {1.0, 1.0, 1.0, 1.0};
    GLfloat shine[] = {1}; // 0 - 128, 0 = shiny, 128 = dull
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, spec_mat);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shine);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, amb_mat);
    if (draw_particle)
      draw_spheres(sites, sphere_rad);
  }
  else {
    glDisable(GL_LIGHTING);
    glColor3f(0.9, 0.9, 0.9);
    glEnable(GL_POINT_SMOOTH);
    glPointSize(1.0);
    if (draw_particle) {
      glBegin(GL_POINTS);
      for (int i = 0; i < (int)sites.size(); i++) {
	if (clip == 0.0 || sites[i].z < z_clip)
	  glVertex3f(sites[i].x, sites[i].y, sites[i].z);
      }
      glEnd();
    }
    glDisable(GL_COLOR_MATERIAL);
  }

  // cell faces
  if (draw_tess) {

    if (draw_fancy) {

      float d = size / 3000.0; // face shift found by trial and error
      GLfloat spec[] = {1.0, 1.0, 1.0, 1.0};
      GLfloat shine[] = {64}; // 0 - 128, 0 = shiny, 128 = dull
      glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, spec);
      glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shine);

      if (color_density) {

	n = 0;

	// for all faces
	for (int i = 0; i < (int)num_face_verts.size(); i++) {

	  // scan all vertices to see if the face should be clipped or drawn
	  bool draw_face = true;
	  if (clip > 0.0) {
	    int m = n;
	    for (int j = 0; j < num_face_verts[i]; j++) {
	      if (verts[m].z > z_clip) {
		draw_face = false;
		break;
	      }
	      m++;
	    }
	  }

	  if (draw_face) {
	    // logartithmic face color from red = small vol to blue = big vol
	    float r, b;
	    b = (log10f(face_vols[i]) - log10f(min_vol_act)) / 
	      (log10f(max_vol_act) - log10f(min_vol_act));
	    r = 1.0 - b;
	    GLfloat mat[] = {r, 0.1, b, 1.0};
	    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, mat);
	    // shift the face to set it back from the edges
	    float dx = vor_normals[i].x * d;
	    float dy = vor_normals[i].y * d;
	    float dz = vor_normals[i].z * d;
	    if (clip > 0.0) {
	      dx = 0.0;
	      dy = 0.0;
	      dz = 0.0;
	    }
	    glBegin(GL_POLYGON);
	    // draw the face
	    for (int j = 0; j < num_face_verts[i]; j++) {
	      if (clip == 0.0 || verts[n].z - dz < z_clip) {
		glNormal3f(vor_normals[i].x, vor_normals[i].y, vor_normals[i].z);
		glVertex3f(verts[n].x - dx, verts[n].y - dy, verts[n].z - dz);
	      }
	      n++;
	    }
	    glEnd();
	  }

	  else // just need to increment n
	    n += num_face_verts[i];

	} // for all faces

      } // color density

      else { // ! color_density

	GLfloat mat[] = {0.65, 0.65, 0.85, 1.0};
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, mat);
	n = 0;

	// for all faces
	for (int i = 0; i < (int)num_face_verts.size(); i++) {

	  // scan all vertices to see if the face should be clipped or drawn
	  bool draw_face = true;
	  if (clip > 0.0) {
	    int m = n;
	    for (int j = 0; j < num_face_verts[i]; j++) {
	      if (verts[m].z > z_clip) {
		draw_face = false;
		break;
	      }
	      m++;
	    }
	  }

	  if (draw_face) {
	    // shift the face to set it back from the edges
	    float dx = vor_normals[i].x * d;
	    float dy = vor_normals[i].y * d;
	    float dz = vor_normals[i].z * d;
	    if (clip > 0.0) {
	      dx = 0.0;
	      dy = 0.0;
	      dz = 0.0;
	    }
	    // draw the face
	    glBegin(GL_POLYGON);
	    for (int j = 0; j < num_face_verts[i]; j++) {
	      glNormal3f(vor_normals[i].x, vor_normals[i].y, vor_normals[i].z);
	      glVertex3f(verts[n].x - dx, verts[n].y - dy, verts[n].z - dz);
	      n++;
	    }
	    glEnd();
	  }

	  else // just need to increment n
	    n += num_face_verts[i];

	} // for all faces

      } // ! color density

    } // draw fancy

  } // draw tess

  glutSwapBuffers();

}
//--------------------------------------------------------------------------
//
// first time drawing initialization
//
void init_display() {

  // extents
  for (int i = 0; i < (int)sites.size(); i++) {
    if (i == 0) {
      site_min.x = sites[i].x;
      site_min.y = sites[i].y;
      site_min.z = sites[i].z;
      site_max.x = sites[i].x;
      site_max.y = sites[i].y;
      site_max.z = sites[i].z;
    }
    if (sites[i].x < site_min.x)
      site_min.x = sites[i].x;
    if (sites[i].y < site_min.y)
      site_min.y = sites[i].y;
    if (sites[i].z < site_min.z)
      site_min.z = sites[i].z;
    if (sites[i].x > site_max.x)
      site_max.x = sites[i].x;
    if (sites[i].y > site_max.y)
      site_max.y = sites[i].y;
    if (sites[i].z > site_max.z)
      site_max.z = sites[i].z;
  }
  site_center.x = (site_min.x + site_max.x) / 2.0;
  site_center.y = (site_min.y + site_max.y) / 2.0;
  site_center.z = (site_min.z + site_max.z) / 2.0;
  sizes.x = site_max.x - site_min.x;
  sizes.y = site_max.y - site_min.y;
  sizes.z = site_max.z - site_min.z;
  size = sizes.x;
  if (sizes.y > size)
    size = sizes.y;
  if (sizes.z > size)
    size = sizes.z;
  fprintf(stderr, "max size = %.4f\n", size);
  sphere_rad = SPHERE_RAD_FACTOR * size;

  init_model();
  init_viewport(true);

  // background
#ifdef PAPER
  glClearColor(1.0, 1.0, 1.0, 1.0); 
#else
  glClearColor(0.0, 0.0, 0.0, 1.0); 
#endif

  // gl state
//   glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_DEPTH_TEST);
  glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
  glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
  glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHT1);
  glEnable(GL_LIGHT2);
  glEnable(GL_NORMALIZE);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glShadeModel(GL_SMOOTH);

  // initialize headlight
  headlight();

  // general purpose quadrics
  q = gluNewQuadric();

  // point sprite texture
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      sprite_rgba[i][j][0] = sprite_intensity[i][j];
      sprite_rgba[i][j][1] = sprite_intensity[i][j];
      sprite_rgba[i][j][2] = sprite_intensity[i][j];
      sprite_rgba[i][j][3] = sprite_intensity[i][j];
    }
  }
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 5, 5, 0, GL_RGBA, 
	       GL_UNSIGNED_BYTE, sprite_rgba);

}
//--------------------------------------------------------------------------
//
// draw delaunay tets
//
void draw_tets() {

  glDisable(GL_LIGHTING);
  glColor4f(0.7, 0.7, 0.7, 1.0);
  if (draw_fancy)
    glLineWidth(2.0);
  else
    glLineWidth(1.0);

  // for all tets
  for (int t = 0; t < (int)tet_verts.size() / 4; t++) {

    int n = t * 4;

    // first triangle
    glBegin(GL_LINE_STRIP);
    glVertex3f(tet_verts[n].x, tet_verts[n].y, tet_verts[n].z);
    glVertex3f(tet_verts[n + 1].x, tet_verts[n + 1].y, tet_verts[n + 1].z);
    glVertex3f(tet_verts[n + 2].x, tet_verts[n + 2].y, tet_verts[n + 2].z);
    glVertex3f(tet_verts[n].x, tet_verts[n].y, tet_verts[n].z);
    glEnd();

    // second triangle
    glBegin(GL_LINE_STRIP);
    glVertex3f(tet_verts[n + 3].x, tet_verts[n + 3].y, tet_verts[n + 3].z);
    glVertex3f(tet_verts[n + 1].x, tet_verts[n + 1].y, tet_verts[n + 1].z);
    glVertex3f(tet_verts[n + 2].x, tet_verts[n + 2].y, tet_verts[n + 2].z);
    glVertex3f(tet_verts[n + 3].x, tet_verts[n + 3].y, tet_verts[n + 3].z);
    glEnd();

    // third triangle
    glBegin(GL_LINE_STRIP);
    glVertex3f(tet_verts[n].x, tet_verts[n].y, tet_verts[n].z);
    glVertex3f(tet_verts[n + 3].x, tet_verts[n + 3].y, tet_verts[n + 3].z);
    glVertex3f(tet_verts[n + 2].x, tet_verts[n + 2].y, tet_verts[n + 2].z);
    glVertex3f(tet_verts[n].x, tet_verts[n].y, tet_verts[n].z);
    glEnd();

    // fourth triangle
    glBegin(GL_LINE_STRIP);
    glVertex3f(tet_verts[n].x, tet_verts[n].y, tet_verts[n].z);
    glVertex3f(tet_verts[n + 1].x, tet_verts[n + 1].y, tet_verts[n + 1].z);
    glVertex3f(tet_verts[n + 3].x, tet_verts[n + 3].y, tet_verts[n + 3].z);
    glVertex3f(tet_verts[n].x, tet_verts[n].y, tet_verts[n].z);
    glEnd();


  } // for all tets

  if (draw_fancy) {

    glDisable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHTING);
    GLfloat amb_mat[] = {0.65, 0.65, 0.85, 1.0};
    GLfloat spec_mat[] = {1.0, 1.0, 1.0, 1.0};
    GLfloat shine[] = {64}; // 0 - 128, 0 = shiny, 128 = dull
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, spec_mat);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shine);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, amb_mat);

    glShadeModel(GL_FLAT);

    // for all tets
    for (int t = 0; t < (int)tet_verts.size() / 4; t++) {

      int n = t * 4;

      // render triangles
      glBegin(GL_TRIANGLE_STRIP);
      glVertex3f(tet_verts[n + 2].x, tet_verts[n + 2].y, tet_verts[n + 2].z);
      glVertex3f(tet_verts[n + 1].x, tet_verts[n + 1].y, tet_verts[n + 1].z);
      glNormal3f(tet_normals[n].x, tet_normals[n].y, 
		 tet_normals[n].z);
      glVertex3f(tet_verts[n].x, tet_verts[n].y, tet_verts[n].z);
      glNormal3f(tet_normals[n + 1].x, tet_normals[n + 1].y, 
		 tet_normals[n + 1].z);
      glVertex3f(tet_verts[n + 3].x, tet_verts[n + 3].y, tet_verts[n + 3].z);
      glNormal3f(tet_normals[n + 2].x, tet_normals[n + 2].y, 
		 tet_normals[n + 2].z);
      glVertex3f(tet_verts[n + 2].x, tet_verts[n + 2].y, tet_verts[n + 2].z);
      glNormal3f(tet_normals[n + 3].x, tet_normals[n + 3].y, 
		 tet_normals[n + 3].z);
      glVertex3f(tet_verts[n + 1].x, tet_verts[n + 1].y, tet_verts[n + 1].z);
      glEnd();

    } // for all tets

  } // all tets

}
//--------------------------------------------------------------------------
//
// set a headlight
//
void headlight() {

  GLfloat light_ambient[4] = {0.1, 0.1, 0.1, 1.0};  
  GLfloat light_diffuse[4] = {0.2, 0.2, 0.2, 1.0};  
  GLfloat light_specular[4] = {0.8, 0.8, 0.8, 1.0};

  glPushMatrix();
  glMatrixMode(GL_MODELVIEW); 
  glLoadIdentity(); 
  glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
  glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
  glEnable(GL_LIGHT0);

  glPopMatrix();

}
//--------------------------------------------------------------------------
//
// cube, useful for block boundaries, etc.
//
void draw_cube(float *mins, float *maxs, float r, float g, float b) {

  glPushMatrix();
  glTranslatef((mins[0] + maxs[0]) / 2.0, (mins[1] + maxs[1]) / 2.0, 
	       (mins[2] + maxs[2]) / 2.0); 
  glScalef(maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2]);
  glColor3f(r, g, b); 
  glutWireCube(1.0);
  glPopMatrix();

}
//--------------------------------------------------------------------------
// 
// sphere for rendering voronoi sites (particles)
//
void draw_sphere(rgba &color, vec3d &pos, float rad) {

  glColor3f(color.r, color.g, color.b); 
  glPushMatrix();
  glTranslatef(pos.x, pos.y, pos.z);
  gluSphere(q, rad, 7, 7);
  glPopMatrix();

}
//--------------------------------------------------------------------------
// 
// all spheres for rendering voronoi sites (particles)
//
void draw_spheres(vector<vec3d> &sites, float rad) {

  for (int i = 0; i < (int)sites.size(); i++) {

    if (clip == 0.0 || sites[i].z < z_clip) {
      glPushMatrix();
      glTranslatef(sites[i].x, sites[i].y, sites[i].z);
     gluSphere(q, rad, 7, 7);
      glPopMatrix();
    }

  }

}
//--------------------------------------------------------------------------
// 
// point sprite for rendering voronoi sites (particles)
//
void draw_sprites(vector<vec3d> &sites, float size) {

  glPushAttrib(GL_ALL_ATTRIB_BITS);

//   glDisable(GL_DEPTH_TEST);
//   glEnable (GL_BLEND); 

  glColor3f(1.0, 1.0, 1.0);  // color doesn't matter, will be textured over

  glPointSize(size);

  glEnable(GL_TEXTURE_2D);
  glEnable(GL_POINT_SPRITE);
  glTexEnvf(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
  glBindTexture(GL_TEXTURE_2D, tex);
  glEnable(GL_POINT_SMOOTH);
  glBegin(GL_POINTS);
  for(int i = 0; i < (int)sites.size(); i++)
    glVertex3f(sites[i].x, sites[i].y, sites[i].z);
  glEnd();
  glDisable(GL_POINT_SPRITE);

//   glDisable (GL_BLEND); 
//   glEnable(GL_DEPTH_TEST);

  glPopAttrib();

}
//--------------------------------------------------------------------------
//
// axes
//
void draw_axes() {

  glPushMatrix();

  // x
  glColor3f(1.0, 0.0, 0.0); 
  glPushMatrix();
  glRotatef(90.0, 0.0, 1.0, 0.0);
  gluCylinder(q, size * 0.007, size * 0.007, sizes.x * 1.2, 15, 1);
  glTranslatef(0.0, 0.0, sizes.x * 1.2);
  gluCylinder(q, size * 0.015, 0.0, size * .015, 20, 1);
  glPopMatrix();

  // y
  glColor3f(0.0, 1.0, 0.0); 
  glPushMatrix();
  glRotatef(-90.0, 1.0, 0.0, 0.0);
  gluCylinder(q, size * 0.007, size * 0.007, sizes.y * 1.2, 15, 1);
  glTranslatef(0.0, 0.0, sizes.y * 1.2);
  gluCylinder(q, size * 0.015, 0.0, size * .015, 20, 1);
  glPopMatrix();

  // z
  glColor3f(0.0, 0.0, 1.0); 
  glPushMatrix();
  gluCylinder(q, size * 0.007, size * 0.007, sizes.z * 1.2, 15, 1);
  glTranslatef(0.0, 0.0, sizes.z * 1.2);
  gluCylinder(q, size * 0.015, 0.0, size * .015, 20, 1);
  glPopMatrix();

  glPopMatrix();

}
//--------------------------------------------------------------------------
//
// mouse button events
//
void mouse(int button, int state, int x, int y) {

  if (state == GLUT_DOWN) {

    press_x = x;
    press_y = y; 
    if (button == GLUT_LEFT_BUTTON)
      xform_mode = XFORM_ROTATE; 
    else if (button == GLUT_RIGHT_BUTTON) 
      xform_mode = XFORM_SCALE; 
    else if (button == GLUT_MIDDLE_BUTTON) 
      xform_mode = XFORM_TRANS; 

  }
  else if (state == GLUT_UP)
    xform_mode = XFORM_NONE; 

}
//--------------------------------------------------------------------------
//
// mouse motion events
//
void motion(int x, int y) {

  if (xform_mode == XFORM_ROTATE) {

    rot.x += (x - press_x) * rot_rate; 
    if (rot.x > 180)
      rot.x -= 360; 
    else if (rot.x < -180)
      rot.x += 360; 
    press_x = x; 
	   
    rot.y += (y - press_y) * rot_rate;
    if (rot.y > 180)
      rot.y -= 360; 
    else if (rot.y <-180)
      rot.y += 360; 
    press_y = y; 

  }
  else if (xform_mode == XFORM_TRANS) {

    trans.x += (x - press_x) * trans_rate; 
    trans.y -= (y - press_y) * trans_rate;  // subtract to reverse y dir.
    press_x = x;
    press_y = y; 

  }
  else if (xform_mode == XFORM_SCALE){

    float old_scale = scale;
    scale /= (1 + (y - press_y) * scale_rate);  // divided to reverse y dir.
    if (scale < 0) 
      scale = old_scale; 
    press_y = y; 

  }

  glutPostRedisplay(); 

}
//--------------------------------------------------------------------------
//
// keyboard events
//
void key(unsigned char key, int x, int y) {

  x = x; // quiet compiler warnings
  y = y;

  switch(key) {

  case 'q':  // quit
    exit(1);
    break; 
  case 't':  // show voronoi tessellation
    draw_tess = !draw_tess;
    draw_del = false;
    break;
  case 'y':  // show delaunay tets
    draw_del = !draw_del;
    draw_tess = false;
    break;
  case 'p':  // show particles
    draw_particle = !draw_particle;
    break;
  case 'd':  // color by density
    color_density = !color_density;
    break;
  case 'z':  // zoom mouse motion
    xform_mode = XFORM_SCALE; 
    break; 
  case 'a':  // panning mouse motion
    xform_mode = XFORM_TRANS; 
    break; 
  case 'r': // reset rotate, pan, zoom, viewport
    init_model();
    init_viewport(true);
    break;
  case 'b': // toggle block visibility
    block_mode = !block_mode;
    break;
  case 'f': // toggle fancy rendering
    draw_fancy = !draw_fancy;
    break;
  case 'c': // increase near clip plane
    clip += 0.1;
    if (clip >= 1.0)
      clip = 1.0;
    z_clip = site_max.z - clip * (site_max.z - site_min.z);
    fprintf(stderr, "clipping at = %.1f of z range\n", clip);
    break;
  case 'C': // decrease near clip plane
    clip -= 0.1;
    if (clip <= 0.0)
      clip = 0.0;
    z_clip = site_max.z - clip * (site_max.z - site_min.z);
    fprintf(stderr, "clipping at = %.1f of z range\n", clip);
    break;
  case 'v': // restrict (minimum) volume range
    min_vol += vol_step;
    fprintf(stderr, "Minimum volume = %.4lf\n", min_vol);
    filter_volume(min_vol, max_vol);
    break;
  case 'V': //  expand (minimum) volume range
    min_vol -= vol_step;
    if (min_vol < 0.0)
      min_vol = 0.0;
    fprintf(stderr, "Minimum volume = %.4lf\n", min_vol);
    filter_volume(min_vol, max_vol);
    break;
  case 'x': // restrict (maximum) volume range
    max_vol -= vol_step;
    if (max_vol < 0.0)
      max_vol = 0.0;
    fprintf(stderr, "Maximum volume = %.4lf\n", max_vol);
    filter_volume(min_vol, max_vol);
    break;
  case 'X': //  expand (maximum) volume range
    max_vol += vol_step;
    fprintf(stderr, "Maximum volume = %.4lf\n", max_vol);
    filter_volume(min_vol, max_vol);
    break;
  case 'R': // reset volume range
    min_vol = 0.0;
    fprintf(stderr, "Minimum volume = %.4lf\n", min_vol);
    filter_volume(min_vol, max_vol);
    break;
  case 's': // decrease volume step size
    vol_step *= 0.1;
    if (vol_step < 0.0001)
      vol_step = 0.0001;
    fprintf(stderr, "Volume step size = %.4lf\n", vol_step);
    break;
  case 'S': // increase volume step size
    vol_step *= 10.0;
    fprintf(stderr, "Volume step size = %.4lf\n", vol_step);
    break;
  default:
    break;

  }
}
//--------------------------------------------------------------------------

#ifdef TET

// dblock version of filtering and clipping

//
// filter volume
//
void filter_volume(float min_vol, float max_vol) {


}
//--------------------------------------------------------------------------
//
// clip cells
//
void clip_cells(float z_clip) {


}
//--------------------------------------------------------------------------

#else

// vblock version of filtering and clipping

//--------------------------------------------------------------------------
//
// filter volume
//
void filter_volume(float min_vol, float max_vol) {

  int num_vis_cells = 0; // number of visible cells

  num_face_verts.clear();
  verts.clear();
  face_vols.clear();

  // package rendering data
  for (int i = 0; i < nblocks; i++) { // blocks

    int  n = 0;
    int m = 0;

    for (int j = 0; j < blocks[i]->num_complete_cells; j++) { // cells

      int cell = blocks[i]->complete_cells[j]; // current cell
      int num_faces; // number of face in the current cell
      int num_verts; // number of vertices in the current face

      if (cell < blocks[i]->num_orig_particles - 1)
	num_faces = blocks[i]->cell_faces_start[cell + 1] -
	  blocks[i]->cell_faces_start[cell];
      else
	num_faces = blocks[i]->tot_num_cell_faces -
	  blocks[i]->cell_faces_start[cell];

      for (int k = 0; k < num_faces; k++) { // faces

	int start = blocks[i]->cell_faces_start[cell];
	int face = blocks[i]->cell_faces[start + k];
	num_verts = blocks[i]->faces[face].num_verts;

	if (blocks[i]->vols[j] >= min_vol &&
	    (max_vol <= 0.0 || blocks[i]->vols[j] <= max_vol)) {
	  num_face_verts.push_back(num_verts);
	  face_vols.push_back(blocks[i]->vols[j]);
	}

	for (int l = 0; l < num_verts; l++) { // vertices

	  int v = blocks[i]->faces[face].verts[l];
	  vec3d s;
	  s.x = blocks[i]->save_verts[3 * v];
	  s.y = blocks[i]->save_verts[3 * v + 1];
	  s.z = blocks[i]->save_verts[3 * v + 2];
	  m++;
	  if (blocks[i]->vols[j] >= min_vol &&
	      (max_vol <= 0.0 || blocks[i]->vols[j] <= max_vol))
	    verts.push_back(s);

	} // vertices

	n++;

      } // faces

      if (blocks[i]->vols[j] >= min_vol &&
	  (max_vol <= 0.0 || blocks[i]->vols[j] <= max_vol))
	num_vis_cells++;

    } // cells

  } // blocks

  fprintf(stderr, "Number of visible cells = %d\n", num_vis_cells);

}
//--------------------------------------------------------------------------
//
// clip cells
//
void clip_cells(float z_clip) {

  float cell_min[3], cell_max[3], cell_centroid[3]; // cell bounds

  sites.clear();

  // blocks
  for (int block = 0; block< nblocks; block++) {

    // cells
    for (int cell = 0; cell < blocks[block]->num_complete_cells; cell++) {

      // cell bounds
      CellBounds(blocks[block], cell, cell_min, cell_max, cell_centroid);

      if (cell_min[2] < z_clip) {
	vec3d s;
	int n = blocks[block]->complete_cells[cell];
	s.x = blocks[block]->sites[3 * n];
	s.y = blocks[block]->sites[3 * n + 1];
	s.z = blocks[block]->sites[3 * n + 2];
	sites.push_back(s);
      }

    } // cells

  } // blocks

}
//--------------------------------------------------------------------------

#endif

//--------------------------------------------------------------------------
//
// get cell bounds
//
// vblock: one voronoi block
// cell: current cell counter
// cell_min, cell_max: cell bounds (output)
// centroid: centroid, mean of all vertices (output)
//
void CellBounds(vblock_t *vblock, int cell,
		float *cell_min, float *cell_max, float *centroid) {

  centroid[0] = 0.0;
  centroid[1] = 0.0;
  centroid[2] = 0.0;
  int tot_verts = 0;

  int num_faces; // number of face in the current cell
  int num_verts; // number of vertices in the current face

  if (cell < vblock->num_orig_particles - 1)
    num_faces = vblock->cell_faces_start[cell + 1] -
      vblock->cell_faces_start[cell];
  else
    num_faces = vblock->tot_num_cell_faces -
      vblock->cell_faces_start[cell];

  // get cell bounds
  for (int k = 0; k < num_faces; k++) { // faces

    int start = vblock->cell_faces_start[cell];
    int face = vblock->cell_faces[start + k];
    num_verts = vblock->faces[face].num_verts;

    for (int l = 0; l < num_verts; l++) { // vertices

      int v = vblock->faces[face].verts[l];
	  
      if (k == 0 && l == 0 || vblock->save_verts[3 * v] < cell_min[0])
	cell_min[0] = vblock->save_verts[3 * v];
      if (k == 0 && l == 0 || vblock->save_verts[3 * v] > cell_max[0])
	cell_max[0] = vblock->save_verts[3 * v];
      centroid[0] += vblock->save_verts[3 * v];

      if (k == 0 && l == 0 || vblock->save_verts[3 * v + 1] < cell_min[1])
	cell_min[1] = vblock->save_verts[3 * v + 1];
      if (k == 0 && l == 0 || vblock->save_verts[3 * v + 1] > cell_max[1])
	cell_max[1] = vblock->save_verts[3 * v + 1];
      centroid[1] += vblock->save_verts[3 * v + 1];

      if (k == 0 && l == 0 || vblock->save_verts[3 * v + 2] < cell_min[2])
	cell_min[2] = vblock->save_verts[3 * v + 2];
      if (k == 0 && l == 0 || vblock->save_verts[3 * v + 2] > cell_max[2])
	cell_max[2] = vblock->save_verts[3 * v + 2];
      centroid[2] += vblock->save_verts[3 * v + 2];

      tot_verts++;

    } // vertices

  } // faces

  centroid[0] /= tot_verts;
  centroid[1] /= tot_verts;
  centroid[2] /= tot_verts;

} 
//--------------------------------------------------------------------------
//
// timer events
//
void timer(int val) {

  val = val; // quiet compiler warning

  glutPostRedisplay();
  glutTimerFunc(10, timer, 0); 

}
//--------------------------------------------------------------------------
//
// reshape events
//
void reshape(int w, int h) {

  // update window and viewport size and aspect ratio
  win_size.x = w;
  win_size.y = h;

  init_viewport(false);

  glutPostRedisplay();

}
//--------------------------------------------------------------------------
//
// initialize model
//
void init_model() {

  // rotate
  rot.x = rot.y = 0.0;

  // translate
  trans.x = trans.y = 0.0;

  // scale (initial scale 1.5 makes the model fill the screen better)
  scale = 1.5 / size;

}
//--------------------------------------------------------------------------
//
// initialize viewport
//
// reset: true = first time or reset to initial viewport
//        false = modify existing viewport
//
void init_viewport(bool reset) {

  if (win_size.x > win_size.y) {
    aspect.x = 1.0;
    aspect.y = win_size.x / win_size.y;
    if (reset)
      trans.y -= (win_size.x - win_size.y) / win_size.y;
  }
  else {
    aspect.x = win_size.y / win_size.x;
    aspect.y = 1.0;
    if (reset)
      trans.x -= (win_size.y - win_size.x) / win_size.x;
  }

  if (!reset) {
    trans.x += (win_size.x - old_win_size.x) / old_win_size.x;
    trans.y += (win_size.y - old_win_size.y) / old_win_size.y;
  }

  old_win_size.x = win_size.x;
  old_win_size.y = win_size.y;

  glViewport(0, 0, win_size.x * aspect.x, win_size.y * aspect.y);

}
//--------------------------------------------------------------------------
//
// compute the centroid (arithmetic mean) of a number of vertices
//
// verts: vertices
// num_verts: number of vertices
// centroid: (output) centroid
//
void Centroid(vec3d *verts, int num_verts, vec3d &centroid) {

  centroid.x = 0.0;
  centroid.y = 0.0;
  centroid.z = 0.0;

  for (int i = 0; i < num_verts; i++) {
    centroid.x += verts[i].x;
    centroid.y += verts[i].y;
    centroid.z += verts[i].z;
  }

  centroid.x /= num_verts;
  centroid.y /= num_verts;
  centroid.z /= num_verts;

}
//--------------------------------------------------------------------------
//
// compute normal of a face using cross product
//
// verts: 3 vertices in order around a face
// normal: (output) the normal of (verts[0] - verts[1] ) x (verts[2] - verts[1])
//
void Normal(vec3d *verts, vec3d &normal) {


  vec3d v0, v1;

  v0.x = verts[0].x - verts[1].x;
  v0.y = verts[0].y - verts[1].y;
  v0.z = verts[0].z - verts[1].z;

  v1.x = verts[2].x - verts[1].x;
  v1.y = verts[2].y - verts[1].y;
  v1.z = verts[2].z - verts[1].z;

  normal.x = v0.y * v1.z - v0.z * v1.y;
  normal.y = v0.z * v1.x - v0.x * v1.z;
  normal.z = v0.x * v1.y - v0.y * v1.x;

  float mag = sqrt(normal.x * normal.x + normal.y * normal.y +
		   normal.z * normal.z);
  // normalize
  normal.x /= mag;
  normal.y /= mag;
  normal.z /= mag;

}
//--------------------------------------------------------------------------
//
// DEPRECATED
//
// compute normal of a face using Newell's method
//
// Newell's method is more robust than simply computing the cross product of
//   three points when the points are colinear or slightly nonplanar. 
//
void NewellNormal(vec3d *verts, int num_verts, vec3d &normal) {

  normal.x = 0.0;
  normal.y = 0.0;
  normal.z = 0.0;

  for (int i = 0; i < num_verts; i++) {
    int cur = i;
    int next = (i + 1) % num_verts;
    normal.x += (verts[cur].y - verts[next].y) * (verts[cur].z + verts[next].z);
    normal.y += (verts[cur].z - verts[next].z) * (verts[cur].x + verts[next].x);
    normal.z += (verts[cur].x - verts[next].x) * (verts[cur].y + verts[next].y);
  }

  float mag = sqrt(normal.x * normal.x + normal.y * normal.y +
		   normal.z * normal.z);
  // normalize
  normal.x /= mag;
  normal.y /= mag;
  normal.z /= mag;

  // direction is inward, need to invert
  normal.x *= -1.0;
  normal.y *= -1.0;
  normal.z *= -1.0;

}
//--------------------------------------------------------------------------
// 
// comparison function for qsort (debugging)
//
int compare(const void *a, const void *b) {

  if (*((int*)a) < *((int*)b))
    return -1;
  if (*((int*)a) == *((int*)b))
    return 0;
  return 1;

}
//--------------------------------------------------------------------------
//
// package rendering data
// gid2lid: mapping of gids to lids
//
void PrepRenderingData(int *gid2lid) {

  // number of sites, cells and tets
  int num_sites;
  int num_vis_cells;
  int num_loc_tets;
  int num_rem_tets;

  // sites
  PrepSiteRendering(num_sites);

  // voronoi cells
  PrepCellRendering(num_vis_cells);

  // delauany tets
  PrepTetRendering(num_loc_tets, num_rem_tets, gid2lid);

  if (min_vol_act < min_vol_clamp)
    min_vol_act = min_vol_clamp;
  if (min_vol == 0.0)
    min_vol = min_vol_act;
  if (max_vol == 0.0)
    max_vol = max_vol_act;

  fprintf(stderr, "Number of particles = %d\n"
	  "Number of visible cells = %d\n"
	  "Number of tets = %d (%d local + %d remote)\n"
	  "Minimum volume = %.4f Maximum volume = %.4f\n",
	  (int)sites.size(), num_vis_cells, 
	  num_loc_tets + num_rem_tets, num_loc_tets, num_rem_tets,
	  min_vol, max_vol);

}
//--------------------------------------------------------------------------

#ifdef TET

// prep rendering for newer dblock model

//--------------------------------------------------------------------------
//
// package cell faces for rendering
//
// num_vis_cells: (output) number of visible cells
//
void PrepCellRendering(int &num_vis_cells) {

  num_vis_cells = 0; // numbe of visible cells

  for (int b = 0; b < nblocks; b++) { // blocks

    // local tets
    for (int t = 0; t < blocks[b]->num_tets; t++) {

      // start circulating
      int start_vert = circulate_start(blocks[b]->tets, t, 0, 1);
//       fprintf(stderr, "starting tet %d at vert %d\n", t, start_vert);

    } // local tets

  } // blocks

}
//--------------------------------------------------------------------------
//
// package tets for rendering
//
// num_loc_tets: (output) number of local tets
// num_rem_tets: (output) number of remote tets
// gid2lid: mapping of gids to lids
//
void PrepTetRendering(int &num_loc_tets, int &num_rem_tets, int *gid2lid) {

  num_loc_tets = 0;
  num_rem_tets = 0;

  for (int i = 0; i < nblocks; i++) { // blocks

    // local tets
    for (int j = 0; j < blocks[i]->num_tets; j++) {

      // check that tet has all neighbors, ie, not on convex hull
      if (blocks[i]->tets[j].tets[0] == -1 ||
	  blocks[i]->tets[j].tets[1] == -1 ||
	  blocks[i]->tets[j].tets[2] == -1 ||
	  blocks[i]->tets[j].tets[3] == -1) {
	// debug
// 	fprintf(stderr, "skipping tet %d (convex hull)\n", j);
	continue;
      }

      // site indices for tet vertices
      int s0 = blocks[i]->tets[j].verts[0];
      int s1 = blocks[i]->tets[j].verts[1];
      int s2 = blocks[i]->tets[j].verts[2];
      int s3 = blocks[i]->tets[j].verts[3];

      // coordinates for tet vertices
      vec3d p0, p1, p2, p3;
      p0.x = blocks[i]->particles[3 * s0];
      p0.y = blocks[i]->particles[3 * s0 + 1];
      p0.z = blocks[i]->particles[3 * s0 + 2];
      p1.x = blocks[i]->particles[3 * s1];
      p1.y = blocks[i]->particles[3 * s1 + 1];
      p1.z = blocks[i]->particles[3 * s1 + 2];
      p2.x = blocks[i]->particles[3 * s2];
      p2.y = blocks[i]->particles[3 * s2 + 1];
      p2.z = blocks[i]->particles[3 * s2 + 2];
      p3.x = blocks[i]->particles[3 * s3];
      p3.y = blocks[i]->particles[3 * s3 + 1];
      p3.z = blocks[i]->particles[3 * s3 + 2];

      // add the vertices
      tet_verts.push_back(p0);
      tet_verts.push_back(p1);
      tet_verts.push_back(p2);
      tet_verts.push_back(p3);

      num_loc_tets++;

    } // local tets

#if 0

    // remote tets
    for (int j = 0; j < blocks[i]->num_rem_tets; j++) {

      // gids for tet vertices
      int g0 = blocks[i]->rem_tet_gids[4 * j];
      int g1 = blocks[i]->rem_tet_gids[4 * j + 1];
      int g2 = blocks[i]->rem_tet_gids[4 * j + 2];
      int g3 = blocks[i]->rem_tet_gids[4 * j + 3];

      // site indices for tet vertices
      int s0 = blocks[i]->rem_tet_nids[4 * j];
      int s1 = blocks[i]->rem_tet_nids[4 * j + 1];
      int s2 = blocks[i]->rem_tet_nids[4 * j + 2];
      int s3 = blocks[i]->rem_tet_nids[4 * j + 3];

      // coordinates for tet vertices
      // assuming that gid = lid, ie, blocks were written and read in gid order
      vec3d p0, p1, p2, p3;

      // p0
      p0.x = blocks[gid2lid[g0]]->particles[3 * s0];
      p0.y = blocks[gid2lid[g0]]->particles[3 * s0 + 1];
      p0.z = blocks[gid2lid[g0]]->particles[3 * s0 + 2];

      // wraparound transform
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j] & DIY_X0) == DIY_X0)
	p0.x += (data_max.x - data_min.x);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j] & DIY_X1) == DIY_X1)
	p0.x -= (data_max.x - data_min.x);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j] & DIY_Y0) == DIY_Y0)
	p0.y += (data_max.y - data_min.y);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j] & DIY_Y1) == DIY_Y1)
	p0.y -= (data_max.y - data_min.y);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j] & DIY_Z0) == DIY_Z0)
	p0.z += (data_max.z - data_min.z);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j] & DIY_Z1) == DIY_Z1)
	p0.z -= (data_max.z - data_min.z);

      // p1
      p1.x = blocks[gid2lid[g1]]->particles[3 * s1];
      p1.y = blocks[gid2lid[g1]]->particles[3 * s1 + 1];
      p1.z = blocks[gid2lid[g1]]->particles[3 * s1 + 2];

      // wraparound transform
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 1] & DIY_X0) == DIY_X0)
	p1.x += (data_max.x - data_min.x);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 1] & DIY_X1) == DIY_X1)
	p1.x -= (data_max.x - data_min.x);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 1] & DIY_Y0) == DIY_Y0)
	p1.y += (data_max.y - data_min.y);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 1] & DIY_Y1) == DIY_Y1)
	p1.y -= (data_max.y - data_min.y);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 1] & DIY_Z0) == DIY_Z0)
	p1.z += (data_max.z - data_min.z);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 1] & DIY_Z1) == DIY_Z1)
	p1.z -= (data_max.z - data_min.z);

      // p2
      p2.x = blocks[gid2lid[g2]]->particles[3 * s2];
      p2.y = blocks[gid2lid[g2]]->particles[3 * s2 + 1];
      p2.z = blocks[gid2lid[g2]]->particles[3 * s2 + 2];

      // wraparound transform
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 2] & DIY_X0) == DIY_X0)
	p2.x += (data_max.x - data_min.x);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 2] & DIY_X1) == DIY_X1)
	p2.x -= (data_max.x - data_min.x);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 2] & DIY_Y0) == DIY_Y0)
	p2.y += (data_max.y - data_min.y);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 2] & DIY_Y1) == DIY_Y1)
	p2.y -= (data_max.y - data_min.y);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 2] & DIY_Z0) == DIY_Z0)
	p2.z += (data_max.z - data_min.z);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 2] & DIY_Z1) == DIY_Z1)
	p2.z -= (data_max.z - data_min.z);

      // p3
      p3.x = blocks[gid2lid[g3]]->particles[3 * s3];
      p3.y = blocks[gid2lid[g3]]->particles[3 * s3 + 1];
      p3.z = blocks[gid2lid[g3]]->particles[3 * s3 + 2];

      // wraparaound transform
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 3] & DIY_X0) == DIY_X0)
	p3.x += (data_max.x - data_min.x);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 3] & DIY_X1) == DIY_X1)
	p3.x -= (data_max.x - data_min.x);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 3] & DIY_Y0) == DIY_Y0)
	p3.y += (data_max.y - data_min.y);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 3] & DIY_Y1) == DIY_Y1)
	p3.y -= (data_max.y - data_min.y);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 3] & DIY_Z0) == DIY_Z0)
	p3.z += (data_max.z - data_min.z);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 3] & DIY_Z1) == DIY_Z1)
	p3.z -= (data_max.z - data_min.z);

      // add the vertices
      tet_verts.push_back(p0);
      tet_verts.push_back(p1);
      tet_verts.push_back(p2);
      tet_verts.push_back(p3);

      num_rem_tets++;

    } // remote tets

#endif

    // tet face normals
    for (int t = 0; t < (int)tet_verts.size() / 4; t++) {

      vec3d normal;
      vec3d tri[3]; // temporary vertices in one triangle
      vec3d centroid; // controid of one tett
      vec3d v; // vector from centroid to first face vertex
      int n = t * 4;

      Centroid(&tet_verts[n], 4, centroid);

      // flat shading, one normal per face
      // package verts into a contigous array to compute normal
      tri[0].x = tet_verts[n + 2].x;
      tri[0].y = tet_verts[n + 2].y;
      tri[0].z = tet_verts[n + 2].z;
      tri[1].x = tet_verts[n + 1].x;
      tri[1].y = tet_verts[n + 1].y;
      tri[1].z = tet_verts[n + 1].z;
      tri[2].x = tet_verts[n    ].x;
      tri[2].y = tet_verts[n    ].y;
      tri[2].z = tet_verts[n    ].z;
      Normal(tri, normal);
      v.x = tet_verts[n].x - centroid.x;
      v.y = tet_verts[n].y - centroid.y;
      v.z = tet_verts[n].z - centroid.z;
      if (v.x * normal.x + v.y * normal.y + v.z * normal.z < 0.0) {
	normal.x *= -1.0;
	normal.y *= -1.0;
	normal.z *= -1.0;
      }
      tet_normals.push_back(normal);

      tri[0].x = tet_verts[n + 3].x;
      tri[0].y = tet_verts[n + 3].y;
      tri[0].z = tet_verts[n + 3].z;
      tri[1].x = tet_verts[n    ].x;
      tri[1].y = tet_verts[n    ].y;
      tri[1].z = tet_verts[n    ].z;
      tri[2].x = tet_verts[n + 1].x;
      tri[2].y = tet_verts[n + 1].y;
      tri[2].z = tet_verts[n + 1].z;
      Normal(tri, normal);
      v.x = tet_verts[n + 1].x - centroid.x;
      v.y = tet_verts[n + 1].y - centroid.y;
      v.z = tet_verts[n + 1].z - centroid.z;
      if (v.x * normal.x + v.y * normal.y + v.z * normal.z < 0.0) {
	normal.x *= -1.0;
	normal.y *= -1.0;
	normal.z *= -1.0;
      }
      tet_normals.push_back(normal);

      tri[0].x = tet_verts[n    ].x;
      tri[0].y = tet_verts[n    ].y;
      tri[0].z = tet_verts[n    ].z;
      tri[1].x = tet_verts[n + 3].x;
      tri[1].y = tet_verts[n + 2].y;
      tri[1].z = tet_verts[n + 3].z;
      tri[2].x = tet_verts[n + 2].x;
      tri[2].y = tet_verts[n + 2].y;
      tri[2].z = tet_verts[n + 2].z;
      Normal(tri, normal);
      v.x = tet_verts[n + 2].x - centroid.x;
      v.y = tet_verts[n + 2].y - centroid.y;
      v.z = tet_verts[n + 2].z - centroid.z;
      if (v.x * normal.x + v.y * normal.y + v.z * normal.z < 0.0) {
	normal.x *= -1.0;
	normal.y *= -1.0;
	normal.z *= -1.0;
      }
      tet_normals.push_back(normal);

      tri[0].x = tet_verts[n + 1].x;
      tri[0].y = tet_verts[n + 1].y;
      tri[0].z = tet_verts[n + 1].z;
      tri[1].x = tet_verts[n + 2].x;
      tri[1].y = tet_verts[n + 2].y;
      tri[1].z = tet_verts[n + 2].z;
      tri[2].x = tet_verts[n + 3].x;
      tri[2].y = tet_verts[n + 3].y;
      tri[2].z = tet_verts[n + 3].z;
      Normal(tri, normal);
      v.x = tet_verts[n + 3].x - centroid.x;
      v.y = tet_verts[n + 3].y - centroid.y;
      v.z = tet_verts[n + 3].z - centroid.z;
      if (v.x * normal.x + v.y * normal.y + v.z * normal.z < 0.0) {
	normal.x *= -1.0;
	normal.y *= -1.0;
	normal.z *= -1.0;
      }
      tet_normals.push_back(normal);

    }

  } // blocks

}
//--------------------------------------------------------------------------
//
// package sites for rendering
//
// num_sites: (output) number of sites
//
void PrepSiteRendering(int &num_sites) {

  int n;

  for (int i = 0; i < nblocks; i++) { // blocks

    // sites
    n = 0;
    for (int j = 0; j < blocks[i]->num_orig_particles; j++) {

      vec3d s;
      s.x = blocks[i]->particles[n];
      s.y = blocks[i]->particles[n + 1];
      s.z = blocks[i]->particles[n + 2];
      n += 3;
      sites.push_back(s);

    }

  } // blocks

  num_sites = (int)sites.size();

}
//--------------------------------------------------------------------------

#else

// prep rendering for older vblock model

//--------------------------------------------------------------------------
//
// package cell faces for rendering
//
// num_vis_cells: (output) number of visible cells
//
void PrepCellRendering(int &num_vis_cells) {

  num_vis_cells = 0; // numbe of visible cells

  for (int i = 0; i < nblocks; i++) { // blocks

    for (int j = 0; j < blocks[i]->num_complete_cells; j++) { // cells

      int cell = blocks[i]->complete_cells[j]; // current cell
      int num_faces; // number of faces in the current cell
      int num_verts; // number of vertices in the current face

      if (cell < blocks[i]->num_orig_particles - 1)
	num_faces = blocks[i]->cell_faces_start[cell + 1] -
	  blocks[i]->cell_faces_start[cell];
      else
	num_faces = blocks[i]->tot_num_cell_faces -
	  blocks[i]->cell_faces_start[cell];

      for (int k = 0; k < num_faces; k++) { // faces

	int start = blocks[i]->cell_faces_start[cell];
	int face = blocks[i]->cell_faces[start + k];
	num_verts = blocks[i]->faces[face].num_verts;

	if (blocks[i]->vols[j] >= min_vol &&
	    (max_vol <= 0.0 || blocks[i]->vols[j] <= max_vol)) {
	  num_face_verts.push_back(num_verts);
	  face_vols.push_back(blocks[i]->vols[j]);
	  if (i == 0 && j == 0)
	    min_vol_act = blocks[i]->vols[j];
	  if (blocks[i]->vols[j] < min_vol_act)
	    min_vol_act = blocks[i]->vols[j];
	  if (blocks[i]->vols[j] > max_vol_act)
	    max_vol_act = blocks[i]->vols[j];
	}

	int v0; // starting vertex of this face in verts list

	for (int l = 0; l < num_verts; l++) { // vertices

	  int v = blocks[i]->faces[face].verts[l];
	  vec3d s;
	  s.x = blocks[i]->save_verts[3 * v];
	  s.y = blocks[i]->save_verts[3 * v + 1];
	  s.z = blocks[i]->save_verts[3 * v + 2];
	  if (blocks[i]->vols[j] >= min_vol &&
	      (max_vol <= 0.0 || blocks[i]->vols[j] <= max_vol)) {
	    verts.push_back(s);
	    if (l == 0)
	      v0 = (int)verts.size() - 1; // note starting vertex of this face
	  }

	} // vertices

	// face normal (flat shading, one normal per face)
	vec3d normal;
	Normal(&verts[v0], normal);

	// check sign of dot product of normal with vector from site 
	// to first face vertex to see if normal has correct direction
	// want outward normal
	vec3d v;
	v.x = verts[v0].x - sites[cell].x;
	v.y = verts[v0].y - sites[cell].y;
	v.z = verts[v0].z - sites[cell].z;
	if (v.x * normal.x + v.y * normal.y + v.z * normal.z < 0.0) {
	  normal.x *= -1.0;
	  normal.y *= -1.0;
	  normal.z *= -1.0;
	}
	vor_normals.push_back(normal);

      } // faces

      if (blocks[i]->vols[j] >= min_vol &&
	  (max_vol <= 0.0 || blocks[i]->vols[j] <= max_vol))
	num_vis_cells++;

    } // cells

  } // blocks

}
//--------------------------------------------------------------------------
//
// package tets for rendering
//
// num_loc_tets: (output) number of local tets
// num_rem_tets: (output) number of remote tets
// gid2lid: mapping of gids to lids
//
void PrepTetRendering(int &num_loc_tets, int &num_rem_tets, int *gid2lid) {

  num_loc_tets = 0;
  num_rem_tets = 0;

  for (int i = 0; i < nblocks; i++) { // blocks

    // local tets
    for (int j = 0; j < blocks[i]->num_loc_tets; j++) {

      // site indices for tet vertices
      int s0 = blocks[i]->loc_tets[4 * j];
      int s1 = blocks[i]->loc_tets[4 * j + 1];
      int s2 = blocks[i]->loc_tets[4 * j + 2];
      int s3 = blocks[i]->loc_tets[4 * j + 3];

      // coordinates for tet vertices
      vec3d p0, p1, p2, p3;
      p0.x = blocks[i]->sites[3 * s0];
      p0.y = blocks[i]->sites[3 * s0 + 1];
      p0.z = blocks[i]->sites[3 * s0 + 2];
      p1.x = blocks[i]->sites[3 * s1];
      p1.y = blocks[i]->sites[3 * s1 + 1];
      p1.z = blocks[i]->sites[3 * s1 + 2];
      p2.x = blocks[i]->sites[3 * s2];
      p2.y = blocks[i]->sites[3 * s2 + 1];
      p2.z = blocks[i]->sites[3 * s2 + 2];
      p3.x = blocks[i]->sites[3 * s3];
      p3.y = blocks[i]->sites[3 * s3 + 1];
      p3.z = blocks[i]->sites[3 * s3 + 2];

      // add the vertices
      tet_verts.push_back(p0);
      tet_verts.push_back(p1);
      tet_verts.push_back(p2);
      tet_verts.push_back(p3);

      num_loc_tets++;

    } // local tets

    // remote tets
    for (int j = 0; j < blocks[i]->num_rem_tets; j++) {

      // gids for tet vertices
      int g0 = blocks[i]->rem_tet_gids[4 * j];
      int g1 = blocks[i]->rem_tet_gids[4 * j + 1];
      int g2 = blocks[i]->rem_tet_gids[4 * j + 2];
      int g3 = blocks[i]->rem_tet_gids[4 * j + 3];

      // site indices for tet vertices
      int s0 = blocks[i]->rem_tet_nids[4 * j];
      int s1 = blocks[i]->rem_tet_nids[4 * j + 1];
      int s2 = blocks[i]->rem_tet_nids[4 * j + 2];
      int s3 = blocks[i]->rem_tet_nids[4 * j + 3];

      // coordinates for tet vertices
      // assuming that gid = lid, ie, blocks were written and read in gid order
      vec3d p0, p1, p2, p3;

      // p0
      p0.x = blocks[gid2lid[g0]]->sites[3 * s0];
      p0.y = blocks[gid2lid[g0]]->sites[3 * s0 + 1];
      p0.z = blocks[gid2lid[g0]]->sites[3 * s0 + 2];

      // wraparound transform
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j] & DIY_X0) == DIY_X0)
	p0.x += (data_max.x - data_min.x);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j] & DIY_X1) == DIY_X1)
	p0.x -= (data_max.x - data_min.x);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j] & DIY_Y0) == DIY_Y0)
	p0.y += (data_max.y - data_min.y);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j] & DIY_Y1) == DIY_Y1)
	p0.y -= (data_max.y - data_min.y);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j] & DIY_Z0) == DIY_Z0)
	p0.z += (data_max.z - data_min.z);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j] & DIY_Z1) == DIY_Z1)
	p0.z -= (data_max.z - data_min.z);

      // p1
      p1.x = blocks[gid2lid[g1]]->sites[3 * s1];
      p1.y = blocks[gid2lid[g1]]->sites[3 * s1 + 1];
      p1.z = blocks[gid2lid[g1]]->sites[3 * s1 + 2];

      // wraparound transform
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 1] & DIY_X0) == DIY_X0)
	p1.x += (data_max.x - data_min.x);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 1] & DIY_X1) == DIY_X1)
	p1.x -= (data_max.x - data_min.x);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 1] & DIY_Y0) == DIY_Y0)
	p1.y += (data_max.y - data_min.y);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 1] & DIY_Y1) == DIY_Y1)
	p1.y -= (data_max.y - data_min.y);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 1] & DIY_Z0) == DIY_Z0)
	p1.z += (data_max.z - data_min.z);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 1] & DIY_Z1) == DIY_Z1)
	p1.z -= (data_max.z - data_min.z);

      // p2
      p2.x = blocks[gid2lid[g2]]->sites[3 * s2];
      p2.y = blocks[gid2lid[g2]]->sites[3 * s2 + 1];
      p2.z = blocks[gid2lid[g2]]->sites[3 * s2 + 2];

      // wraparound transform
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 2] & DIY_X0) == DIY_X0)
	p2.x += (data_max.x - data_min.x);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 2] & DIY_X1) == DIY_X1)
	p2.x -= (data_max.x - data_min.x);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 2] & DIY_Y0) == DIY_Y0)
	p2.y += (data_max.y - data_min.y);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 2] & DIY_Y1) == DIY_Y1)
	p2.y -= (data_max.y - data_min.y);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 2] & DIY_Z0) == DIY_Z0)
	p2.z += (data_max.z - data_min.z);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 2] & DIY_Z1) == DIY_Z1)
	p2.z -= (data_max.z - data_min.z);

      // p3
      p3.x = blocks[gid2lid[g3]]->sites[3 * s3];
      p3.y = blocks[gid2lid[g3]]->sites[3 * s3 + 1];
      p3.z = blocks[gid2lid[g3]]->sites[3 * s3 + 2];

      // wraparaound transform
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 3] & DIY_X0) == DIY_X0)
	p3.x += (data_max.x - data_min.x);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 3] & DIY_X1) == DIY_X1)
	p3.x -= (data_max.x - data_min.x);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 3] & DIY_Y0) == DIY_Y0)
	p3.y += (data_max.y - data_min.y);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 3] & DIY_Y1) == DIY_Y1)
	p3.y -= (data_max.y - data_min.y);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 3] & DIY_Z0) == DIY_Z0)
	p3.z += (data_max.z - data_min.z);
      if ((blocks[i]->rem_tet_wrap_dirs[4 * j + 3] & DIY_Z1) == DIY_Z1)
	p3.z -= (data_max.z - data_min.z);

      // add the vertices
      tet_verts.push_back(p0);
      tet_verts.push_back(p1);
      tet_verts.push_back(p2);
      tet_verts.push_back(p3);

      num_rem_tets++;

    } // remote tets

    // tet face normals
    for (int t = 0; t < (int)tet_verts.size() / 4; t++) {

      vec3d normal;
      vec3d tri[3]; // temporary vertices in one triangle
      vec3d centroid; // controid of one tett
      vec3d v; // vector from centroid to first face vertex
      int n = t * 4;

      Centroid(&tet_verts[n], 4, centroid);

      // flat shading, one normal per face
      // package verts into a contigous array to compute normal
      tri[0].x = tet_verts[n + 2].x;
      tri[0].y = tet_verts[n + 2].y;
      tri[0].z = tet_verts[n + 2].z;
      tri[1].x = tet_verts[n + 1].x;
      tri[1].y = tet_verts[n + 1].y;
      tri[1].z = tet_verts[n + 1].z;
      tri[2].x = tet_verts[n    ].x;
      tri[2].y = tet_verts[n    ].y;
      tri[2].z = tet_verts[n    ].z;
      Normal(tri, normal);
      v.x = tet_verts[n].x - centroid.x;
      v.y = tet_verts[n].y - centroid.y;
      v.z = tet_verts[n].z - centroid.z;
      if (v.x * normal.x + v.y * normal.y + v.z * normal.z < 0.0) {
	normal.x *= -1.0;
	normal.y *= -1.0;
	normal.z *= -1.0;
      }
      tet_normals.push_back(normal);

      tri[0].x = tet_verts[n + 3].x;
      tri[0].y = tet_verts[n + 3].y;
      tri[0].z = tet_verts[n + 3].z;
      tri[1].x = tet_verts[n    ].x;
      tri[1].y = tet_verts[n    ].y;
      tri[1].z = tet_verts[n    ].z;
      tri[2].x = tet_verts[n + 1].x;
      tri[2].y = tet_verts[n + 1].y;
      tri[2].z = tet_verts[n + 1].z;
      Normal(tri, normal);
      v.x = tet_verts[n + 1].x - centroid.x;
      v.y = tet_verts[n + 1].y - centroid.y;
      v.z = tet_verts[n + 1].z - centroid.z;
      if (v.x * normal.x + v.y * normal.y + v.z * normal.z < 0.0) {
	normal.x *= -1.0;
	normal.y *= -1.0;
	normal.z *= -1.0;
      }
      tet_normals.push_back(normal);

      tri[0].x = tet_verts[n    ].x;
      tri[0].y = tet_verts[n    ].y;
      tri[0].z = tet_verts[n    ].z;
      tri[1].x = tet_verts[n + 3].x;
      tri[1].y = tet_verts[n + 2].y;
      tri[1].z = tet_verts[n + 3].z;
      tri[2].x = tet_verts[n + 2].x;
      tri[2].y = tet_verts[n + 2].y;
      tri[2].z = tet_verts[n + 2].z;
      Normal(tri, normal);
      v.x = tet_verts[n + 2].x - centroid.x;
      v.y = tet_verts[n + 2].y - centroid.y;
      v.z = tet_verts[n + 2].z - centroid.z;
      if (v.x * normal.x + v.y * normal.y + v.z * normal.z < 0.0) {
	normal.x *= -1.0;
	normal.y *= -1.0;
	normal.z *= -1.0;
      }
      tet_normals.push_back(normal);

      tri[0].x = tet_verts[n + 1].x;
      tri[0].y = tet_verts[n + 1].y;
      tri[0].z = tet_verts[n + 1].z;
      tri[1].x = tet_verts[n + 2].x;
      tri[1].y = tet_verts[n + 2].y;
      tri[1].z = tet_verts[n + 2].z;
      tri[2].x = tet_verts[n + 3].x;
      tri[2].y = tet_verts[n + 3].y;
      tri[2].z = tet_verts[n + 3].z;
      Normal(tri, normal);
      v.x = tet_verts[n + 3].x - centroid.x;
      v.y = tet_verts[n + 3].y - centroid.y;
      v.z = tet_verts[n + 3].z - centroid.z;
      if (v.x * normal.x + v.y * normal.y + v.z * normal.z < 0.0) {
	normal.x *= -1.0;
	normal.y *= -1.0;
	normal.z *= -1.0;
      }
      tet_normals.push_back(normal);

    }

  } // blocks

}
//--------------------------------------------------------------------------
//
// package sites for rendering
//
// num_sites: (output) number of sites
//
void PrepSiteRendering(int &num_sites) {

  int n;

  for (int i = 0; i < nblocks; i++) { // blocks

    // sites
    n = 0;
    for (int j = 0; j < blocks[i]->num_orig_particles; j++) {

      vec3d s;
      s.x = blocks[i]->sites[n];
      s.y = blocks[i]->sites[n + 1];
      s.z = blocks[i]->sites[n + 2];
      n += 3;
      sites.push_back(s);

    }

  } // blocks

  num_sites = (int)sites.size();

}
//--------------------------------------------------------------------------

#endif
