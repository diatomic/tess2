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
//--------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include "tess/delaunay.h"
#include "tess/tet-neighbors.h"
#include "tess/tet.h"
#include <math.h>
#include "mpi.h"
#include "tess/volume.h"

#include "tess/tess.hpp"

#include <diy/master.hpp>
#include <diy/io/block.hpp>

#if defined(MAC_OSX)
#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/glut.h>
#include <GL/gl.h>
#endif

#define SPHERE_RAD_FACTOR .001 // used to compute sphere radius
// #define PAPER // color scheme for paper (white backgound)

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

// data overall extent scaling factor for clipping cells
float ds = 1.01;

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
bool draw_verts = false;

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

// voronoi vertices, faces, cells, normals, volumes
vector<vec3d> verts;
vector<vec3d> cell_verts;
vector<int> num_face_verts;
vector<vec3d> vor_normals;
vector<float> vols;

// delaunay tet vertics, face normals
vector<vec3d> tet_verts;
vector<vec3d> tet_normals;

// volumes associated with faces
vector <float> face_vols;

// global data extents
vec3d data_min, data_max;

// local blocks
//dblock_t *blocks; // newer delaunay blocks
int nblocks;
diy::Master* master;

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

// statistical summary
#define NUM_HIST_BINS 50
struct stats_t {
  int tot_particles; // total number of particles
  int tot_tets; // total number of delaunay tetrahedra found
  int tot_cells; // total number of finit voronoi cells found
  int tot_cell_faces; // total number of faces in all finite voronoi cells (not unique)
  int tot_cell_verts; // total number of vertices in all finite voronoi cells (not unique)
  float avg_cell_faces; // average number of faces per cell
  float avg_cell_verts; // average number of vertices per cell
  float min_cell_vol; // minimum cell volume for finite voronoi cells
  float max_cell_vol; // maximum cell volume for finite voronoi cells
  float tot_cell_vol; // total cell volume for finite voronoi cells
  float avg_cell_vol; // average cell volume for finite voronoi cells
  int vol_hist[NUM_HIST_BINS]; // cell volume histogram
};

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
void draw_cell_verts();
void reshape(int w, int h);
void init_model();
void init_viewport(bool reset);
void headlight();
void Centroid(vec3d *verts, int num_verts, vec3d &centroid);
void NewellNormal(vec3d *verts, int num_verts, vec3d &normal);
void PrepRenderingData();
void PrepSiteRendering(stats_t& stats);
void PrepCellRendering(stats_t& stats);
void PrepCellVertRendering(stats_t& stats);
void PrepTetRendering(stats_t& stats);
bool my_tet(dblock_t& dblock, int t);
bool check_if_delaunay(dblock_t& dblock, int t);
void collect_stats(stats_t& stats);
void histogram(struct stats_t& stats);

// debug
float flatness(dblock_t& b, int t);

//--------------------------------------------------------------------------

int main(int argc, char** argv) {

  if (argc < 2) {
    fprintf(stderr, "Usage: draw <filename>\n");
    exit(0);
  }

  // read the file
  int tot_blocks; // total number of blocks
  int *num_neighbors; // number of neighbors for each local block (unused)
  int **neighbors; // neighbors of each local block (unused)
  int **neigh_procs; // procs of neighbors of each local block (unused)

  //MPI_Init(&argc, &argv);
  diy::mpi::environment	    env(argc, argv);
  diy::mpi::communicator    world;

  diy::Master               master(world, -1, -1,
                                   &create_block,
                                   &destroy_block);

  diy::ContiguousAssigner   assigner(world.size(), -1);	    // number of blocks will be set by read_blocks()
  diy::io::read_blocks(argv[1], world, assigner, master, &load_block_light);
  //pnetcdf_read(&nblocks, &tot_blocks, &blocks, argv[1], MPI_COMM_WORLD,
  //             &num_neighbors, &neighbors, &neigh_procs);

  //MPI_Finalize();
  
  tot_blocks = nblocks = master.size();
  ::master = &master;

  fprintf(stderr, "Blocks read: %d\n", tot_blocks);

  // get overall data extent
  for (int i = 0; i < nblocks; i++) {
    if (i == 0) {
      data_min.x = master.block<dblock_t>(i)->mins[0];
      data_min.y = master.block<dblock_t>(i)->mins[1];
      data_min.z = master.block<dblock_t>(i)->mins[2];
      data_max.x = master.block<dblock_t>(i)->maxs[0];
      data_max.y = master.block<dblock_t>(i)->maxs[1];
      data_max.z = master.block<dblock_t>(i)->maxs[2];
    }
    if (master.block<dblock_t>(i)->mins[0] < data_min.x)
      data_min.x = master.block<dblock_t>(i)->mins[0];
    if (master.block<dblock_t>(i)->mins[1] < data_min.y)
      data_min.y = master.block<dblock_t>(i)->mins[1];
    if (master.block<dblock_t>(i)->mins[2] < data_min.z)
      data_min.z = master.block<dblock_t>(i)->mins[2];
    if (master.block<dblock_t>(i)->maxs[0] > data_max.x)
      data_max.x = master.block<dblock_t>(i)->maxs[0];
    if (master.block<dblock_t>(i)->maxs[1] > data_max.y)
      data_max.y = master.block<dblock_t>(i)->maxs[1];
    if (master.block<dblock_t>(i)->maxs[2] > data_max.z)
      data_max.z = master.block<dblock_t>(i)->maxs[2];
  }

  // debug
  fprintf(stderr, "data sizes mins[%.3f %.3f %.3f] maxs[%.3f %.3f %.3f]\n",
	  data_min.x, data_min.y, data_min.z,
	  data_max.x, data_max.y, data_max.z);

  // package rendering data
  PrepRenderingData();

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

  // block bounds
  if (block_mode) {
    glDisable(GL_LIGHTING);
    for (int i = 0; i < nblocks; i++)
      draw_cube(master->block<dblock_t>(i)->mins, master->block<dblock_t>(i)->maxs, 1.0, 0.0, 1.0);
  }

  // cell verts
  if (draw_verts)
    draw_cell_verts();

  // delaunay tets
  if (draw_del)
    draw_tets();

  // cell edges
  if (draw_tess) {

    glDisable(GL_LIGHTING);
    glColor4f(0.9, 0.9, 0.9, 1.0);
    if (draw_fancy)
      glLineWidth(1.3);
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
      GLfloat spec[] = {0.2, 0.2, 0.2};
      GLfloat shine[] = {128}; // 0 - 128, 0 = shiny, 128 = dull
      glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, spec);
      glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shine);


      GLfloat mat[] = {0.3, 0.3, 0.5, 1.0};
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
// draw cell vertices
//
void draw_cell_verts() {

    glDisable(GL_LIGHTING);
    glColor3f(0.9, 0.9, 0.0);
    glEnable(GL_POINT_SMOOTH);
    glPointSize(1.0);

    glBegin(GL_POINTS);
    for (int i = 0; i < (int)cell_verts.size(); i++)
      glVertex3f(cell_verts[i].x, cell_verts[i].y, cell_verts[i].z);
    glEnd();

    glDisable(GL_COLOR_MATERIAL);

}
//--------------------------------------------------------------------------
//
// draw delaunay tets
//
void draw_tets() {

  glDisable(GL_LIGHTING);
  glColor4f(0.7, 0.7, 0.7, 1.0);
  if (draw_fancy)
    glLineWidth(1.5);
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
  case 'w':  // show voronoi cell vertices
    draw_verts = !draw_verts;
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
  default:
    break;

  }
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
// package rendering data
//
void PrepRenderingData() {

  stats_t stats;

  // sites
  PrepSiteRendering(stats);

  // voronoi cells
  PrepCellRendering(stats);

  // delauany tets
  PrepTetRendering(stats);

  collect_stats(stats);

}
//--------------------------------------------------------------------------
//
// package cell vertices for rendering
// used for debugging
//
void PrepCellVertRendering() {

  for (int b = 0; b < nblocks; b++) { // blocks

    // tets
    for (int t = 0; t < master->block<dblock_t>(b)->num_tets; t++) {

      // push voronoi vertex for rendering
      // voronoi vertex is the circumcenter of the tet
      vec3d center;
      circumcenter(&center.x,
		   &(master->block<dblock_t>(b)->tets[t]), master->block<dblock_t>(b)->particles);

#if 0	// debug purpuses only
      const tet_t& tt = master->block<dblock_t>(b)->tets[t];
      float dist = distance(&center.x, &master->block<dblock_t>(b)->particles[3*tt.verts[0]]);
      for (int i = 1; i < 4; ++i) {
	float dist2 = distance(&center.x, &master->block<dblock_t>(b)->particles[3*tt.verts[i]]);
	if (fabs(dist - dist2) > .00001) {
	  fprintf(stderr, "Warning: %f %f\n", dist, dist2);
	}
      }
#endif

      cell_verts.push_back(center);

    } // tets

  } // blocks

}
//--------------------------------------------------------------------------
//
// package cell faces for rendering
//
void PrepCellRendering(stats_t& stats) {

  int v0 = 0; // starting vertex of the current face
  stats.tot_cells = 0; // number of visible cells
  stats.tot_cell_vol = 0.0; // total cell volume

  for (int b = 0; b < nblocks; b++) { // blocks

    vector<float> circumcenters;
    fill_circumcenters(circumcenters, master->block<dblock_t>(b)->tets, master->block<dblock_t>(b)->num_tets, master->block<dblock_t>(b)->particles);

    // for all voronoi cells
    for (int p = 0; p < master->block<dblock_t>(b)->num_orig_particles; p++) {

      // tet
      int t = master->block<dblock_t>(b)->vert_to_tet[p];

      // skip tets with missing neighbors
      if (master->block<dblock_t>(b)->tets[t].tets[0] == -1 || master->block<dblock_t>(b)->tets[t].tets[1] == -1 ||
	  master->block<dblock_t>(b)->tets[t].tets[2] == -1 || master->block<dblock_t>(b)->tets[t].tets[3] == -1)
	continue;

      // neighbor edges a vector of (vertex u, tet of vertex u) pairs
      // that neighbor vertex v
      vector< pair<int, int> > nbrs;
      bool finite = neighbor_edges(nbrs, p, master->block<dblock_t>(b)->tets, t);

      // skip tet vertices corresponding to incomplete voronoi cells
      if (!finite)
        continue;

      bool keep = true; // this cell passes all tests, volume, data extents
      vector <vec3d> temp_verts; // verts in this cell
      vector <int> temp_num_face_verts;  // number of face verts in this call
      vector <vec3d> temp_vor_normals; // face normals in this cell

      // the following loop is the equivalent of
      // for all faces in a voronoi cell
      for (int i = 0; i < (int)nbrs.size(); ++i) {

	v0 = (int)temp_verts.size(); // note starting vertex of this face

	// get edge link
	int u  = nbrs[i].first;
	int ut = nbrs[i].second;
	std::vector<int> edge_link;
	fill_edge_link(edge_link, p, u, ut, master->block<dblock_t>(b)->tets);

	// following is equivalent of all vertices in a face
	for (int j = 0; j < (int)edge_link.size(); ++j) {

	  vec3d center;
	  circumcenter((float *)&(center.x),
		       &(master->block<dblock_t>(b)->tets[edge_link[j]]), master->block<dblock_t>(b)->particles);

	  // filter out cells far outside the overal extents
	  if (center.x > data_max.x + (data_max.x - data_min.x) * (ds - 1) ||
	      center.x < data_min.x - (data_max.x - data_min.x) * (ds - 1) ||
	      center.y > data_max.y + (data_max.y - data_min.y) * (ds - 1) ||
	      center.y < data_min.y - (data_max.y - data_min.y) * (ds - 1) ||
	      center.z > data_max.z + (data_max.z - data_min.z) * (ds - 1) ||
	      center.z < data_min.z - (data_max.z - data_min.z) * (ds - 1))
	    keep = false;

	  temp_verts.push_back(center);

	}

	temp_num_face_verts.push_back(edge_link.size());

	// face normal (flat shading, one normal per face)
	vec3d normal;
	NewellNormal(&temp_verts[v0], edge_link.size(), normal);

	// check sign of dot product of normal with vector from site
	// to first face vertex to see if normal has correct direction
	// want outward normal
	vec3d vec;
	vec.x = temp_verts[v0].x - sites[p].x;
	vec.y = temp_verts[v0].y - sites[p].y;
	vec.z = temp_verts[v0].z - sites[p].z;
	if (vec.x * normal.x + vec.y * normal.y + vec.z * normal.z < 0.0) {
	  normal.x *= -1.0;
	  normal.y *= -1.0;
	  normal.z *= -1.0;
	}
	temp_vor_normals.push_back(normal);

      } // for all faces in a voronoi cell

      // keep the cell
      // todo: check the cell volume here against min,max thresholds
      if (keep) {
	for (int k = 0; k < (int)temp_verts.size(); k++)
	  verts.push_back(temp_verts[k]);
	for (int k = 0; k < (int)temp_num_face_verts.size(); k++)
	  num_face_verts.push_back(temp_num_face_verts[k]);
	for (int k = 0; k < (int)temp_vor_normals.size(); k++)
	  vor_normals.push_back(temp_vor_normals[k]);
        stats.tot_cells++;
        vols.push_back(volume(p, master->block<dblock_t>(b)->vert_to_tet, master->block<dblock_t>(b)->tets, master->block<dblock_t>(b)->num_tets,
                              master->block<dblock_t>(b)->particles, circumcenters));
        if (vols.size() == 1 || vols.back() < stats.min_cell_vol)
          stats.min_cell_vol = vols.back();
        if (vols.size() == 1 || vols.back() > stats.max_cell_vol)
          stats.max_cell_vol = vols.back();
        stats.tot_cell_vol += vols.back();
      }

    } // voronoi cells

  } // blocks

  stats.tot_cell_faces = num_face_verts.size();
  stats.tot_cell_verts = verts.size();

}
//--------------------------------------------------------------------------

bool check_if_delaunay(dblock_t& dblock, int t)
{
  vec3d center;
  circumcenter(&center.x,
	       &(dblock.tets[t]), dblock.particles);

  const tet_t& tt = dblock.tets[t];
  float dist = distance(&center.x, &dblock.particles[3*tt.verts[0]]);

  for (unsigned i = 0; i < dblock.num_particles; ++i)
  {
    float dist_i = distance(&center.x, &dblock.particles[3*i]);
    if (dist_i < dist - .00001)
    {
      std::cout << dist_i << " < " << dist << std::endl;
      return false;
    }
  }

  return true;
}

//--------------------------------------------------------------------------
//
// package tets for rendering
//
void PrepTetRendering(stats_t& stats) {

  stats.tot_tets = 0;

  for (int b = 0; b < nblocks; b++) { // blocks

    // tets
    for (int t = 0; t < master->block<dblock_t>(b)->num_tets; t++) {

      // determine unique ownership of the tet
      if (!my_tet(*(master->block<dblock_t>(b)), t))
        continue;

      for (int v = 0; v < 4; v++) {

	int s = master->block<dblock_t>(b)->tets[t].verts[v]; // index pf particle
	vec3d p; // coordinates for tet vertices
        p.x = master->block<dblock_t>(b)->particles[3 * s];
        p.y = master->block<dblock_t>(b)->particles[3 * s + 1];
        p.z = master->block<dblock_t>(b)->particles[3 * s + 2];
	tet_verts.push_back(p);

      }

      stats.tot_tets++;

    } // tets

    // tet face normals

    // for all tets
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
      NewellNormal(tri,3,normal);
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
      NewellNormal(tri,3,normal);
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
      tri[1].y = tet_verts[n + 3].y;
      tri[1].z = tet_verts[n + 3].z;
      tri[2].x = tet_verts[n + 2].x;
      tri[2].y = tet_verts[n + 2].y;
      tri[2].z = tet_verts[n + 2].z;
      NewellNormal(tri,3,normal);
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
      NewellNormal(tri,3,normal);
      v.x = tet_verts[n + 3].x - centroid.x;
      v.y = tet_verts[n + 3].y - centroid.y;
      v.z = tet_verts[n + 3].z - centroid.z;
      if (v.x * normal.x + v.y * normal.y + v.z * normal.z < 0.0) {
	normal.x *= -1.0;
	normal.y *= -1.0;
	normal.z *= -1.0;
      }
      tet_normals.push_back(normal);

    } // tets

  } // blocks

}
//--------------------------------------------------------------------------
//
// package sites for rendering
//
void PrepSiteRendering(stats_t& stats) {

  int n;

  for (int i = 0; i < nblocks; i++) { // blocks

    // sites
    n = 0;
    for (int j = 0; j < master->block<dblock_t>(i)->num_orig_particles; j++) {

      vec3d s;
      s.x = master->block<dblock_t>(i)->particles[n];
      s.y = master->block<dblock_t>(i)->particles[n + 1];
      s.z = master->block<dblock_t>(i)->particles[n + 2];
      n += 3;
      sites.push_back(s);

    }

  } // blocks

  stats.tot_particles = (int)sites.size();

}
//--------------------------------------------------------------------------
//
// determines whether this block owns this tet
// the minimum gid block of the four vertices is the tet owner
//
// dblock: local delaunay block
// t: index of tet
//
bool my_tet(dblock_t& dblock, int t) {

  int v; // tet vertex (0-3)

  // Check whether this tet is entirely remote.
  // This test is redundant in most cases because it is covered by the check for minimum gid, but
  // when wrapping and only one block in a dimension (a degenerate case that we don't handle),
  // ruling out the tets that can't belong to me based on particle id produces a "nicer" result;
  // in those cases the minimum gid test is used only for tets that have possible multiple owners
  for (v = 0; v < 4; v++) {
    int p = dblock.tets[t].verts[v];
    if (p < dblock.num_orig_particles)
      break;
  }
  if (v == 4)
    return false;

  vector <int> gids; // owners (block gids) of tet vertices
  for (int v = 0; v < 4; v++) {
    int p = dblock.tets[t].verts[v];
    if (p < dblock.num_orig_particles)
      gids.push_back(dblock.gid);
    else
      gids.push_back(dblock.rem_gids[p - dblock.num_orig_particles]);
  }

  return(*min_element(gids.begin(), gids.end()) == dblock.gid);

}
//--------------------------------------------------------------------------
//
// debug: check flatness of tet
// (minimum unsigned distance from 4th point to the plane of the other 3,
// for all 4 vertex candidates for 4th point)
//
// dblock: local delaunay block
// t: index of tet
float flatness(dblock_t& b, int t) {

  vec3d x[3]; // triangle vertices from v0, v1, v2
  vec3d n; // unit normal to v0, v1, v2
  vec3d d; // v3 - v0
  float min_flat; // minimum flatness so far

  // fourth vertex
  for (int vv = 0; vv < 4; vv++) {

    // unit normal from other 3 verts
    int m = 0; // triangle vertex counter
    for (int v = 0; v < 4; v++) {
      if (v == vv)
        continue;
      int p = b.tets[t].verts[v];
      x[m].x = b.particles[3 * p];
      x[m].y = b.particles[3 * p + 1];
      x[m].z = b.particles[3 * p + 2];
      m++;
    }
    NewellNormal(x, 3, n);

    // distance from fourth vertex to any of the other vertices
    int p3 = b.tets[t].verts[vv];
    int p0 = b.tets[t].verts[(vv + 1) % 4];
    d.x = b.particles[3 * p3    ] - b.particles[3 * p0    ];
    d.y = b.particles[3 * p3 + 1] - b.particles[3 * p0 + 1];
    d.z = b.particles[3 * p3 + 2] - b.particles[3 * p0 + 2];

    float flat = fabs(n.x * d.x + n.y * d.y + n.z * d.z);
    if (vv == 0 || flat < min_flat)
      min_flat = flat;

  }

  // unsigned distance is absolute value of n dot dv
  return min_flat;

}
//--------------------------------------------------------------------------
//
//  collects statistics
//
void collect_stats(stats_t& stats) {

  float vol_bin_width; // width of a volume histogram bin

  stats.avg_cell_faces = stats.tot_cell_faces / stats.tot_cells;
  stats.avg_cell_verts = stats.tot_cell_verts / stats.tot_cells;
  stats.avg_cell_vol   = stats.tot_cell_vol   / stats.tot_cells;
  histogram(stats);

  // --- print output ---

  vol_bin_width = (stats.max_cell_vol - stats.min_cell_vol) / NUM_HIST_BINS;
  fprintf(stderr, "----------------- global stats ------------------\n");
  fprintf(stderr, "total delaunay tetrahedra                = %d\n", stats.tot_tets);
  fprintf(stderr, "total voronoi cells                      = %d\n", stats.tot_cells);
  fprintf(stderr, "average number of faces    per vor. cell = %.0lf\n", stats.avg_cell_faces);
  fprintf(stderr, "average number of vertices per vor. cell = %.0lf\n", stats.avg_cell_verts);
  fprintf(stderr, "average number of vertices per vor. face = %.0lf\n",
          stats.avg_cell_verts / stats.avg_cell_faces);
  fprintf(stderr, "-----\n");
  fprintf(stderr, "min vor. cell volume = %.3lf max vor. cell volume = %.3lf\n"
          "avg vor. cell volume = %.3lf units^3\n",
          stats.min_cell_vol, stats.max_cell_vol, stats.avg_cell_vol);
  fprintf(stderr, "number of vor. cell volume histogram bins = %d\n", NUM_HIST_BINS);
  fprintf(stderr, "-----\n");
  fprintf(stderr, "voronoi cell volume histogram:\n");
  fprintf(stderr, "min value\tcount\t\tmax value\n");
  for (int k = 0; k < NUM_HIST_BINS; k++)
    fprintf(stderr, "%.3lf\t\t%d\t\t%.3lf\n",
            stats.min_cell_vol + k * vol_bin_width, stats.vol_hist[k],
            stats.min_cell_vol + (k + 1) * vol_bin_width);
  fprintf(stderr, "-------------------------------------------------\n");

}
//--------------------------------------------------------------------------
//
//  computes histogram
//
void histogram(struct stats_t& stats) {

  float vol_bin_width; // width of a volume histogram bin

  // find local cell volume and density histograms
  vol_bin_width = (stats.max_cell_vol - stats.min_cell_vol) / NUM_HIST_BINS;
  for (int j = 0; j < NUM_HIST_BINS; j++) // volume
    stats.vol_hist[j] = 0;

  for (int j = 0; j < stats.tot_cells; j++) { // for all cells

    int k;
    for (k = 0; k < NUM_HIST_BINS; k++) { // for all bins
      if (vols[j] >= stats.min_cell_vol + k * vol_bin_width &&
          vols[j] < stats.min_cell_vol + (k + 1) * vol_bin_width) {
        stats.vol_hist[k]++;
        break;
      }
    } // for all bins
    if (k == NUM_HIST_BINS)
      stats.vol_hist[k - 1]++; // catch roundoff error and open interval on right side of bin

  } // for all cells

}
//--------------------------------------------------------------------------
