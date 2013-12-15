//---------------------------------------------------------------------------
//
// dump of voronoi output file
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
#include "ser_io.hpp"

using namespace std;

// 3d point or vector
struct vec3d {
  float x, y, z;
};

//--------------------------------------------------------------------------

int main(int argc, char** argv) {

  // voronoi sites, vertices, number of vertices per face
  vector<vec3d> sites;
  vector<vec3d> verts;
  vector<int> num_face_verts;

  // voronoi blocks
  vblock_t **vblocks;
  int nblocks;

  int n, m;

  if (argc < 2) {
    fprintf(stderr, "Usage: draw <filename>\n");
    exit(0);

  }

  // read the file
  SER_IO *io = new SER_IO; // io object
  nblocks = io->ReadAllBlocks(argv[1], vblocks, false);

  // package rendering data
  for (int i = 0; i < nblocks; i++) { // blocks

    n = 0;
    for (int j = 0; j < vblocks[i]->num_orig_particles; j++) {

	vec3d s;
	s.x = vblocks[i]->sites[n];
	s.y = vblocks[i]->sites[n + 1];
	s.z = vblocks[i]->sites[n + 2];
	n += 3;
	sites.push_back(s);
    }

    n = 0;
    m = 0;
    for (int j = 0; j < vblocks[i]->num_complete_cells; j++) { // cells

      int cell = vblocks[i]->complete_cells[j]; // current cell
      int num_faces; // number of face in the current cell
      int num_verts; // number of vertices in the current face

      if (cell < vblocks[i]->num_orig_particles - 1)
	num_faces = vblocks[i]->cell_faces_start[cell + 1] -
	  vblocks[i]->cell_faces_start[cell];
      else
	num_faces = vblocks[i]->tot_num_cell_faces -
	  vblocks[i]->cell_faces_start[cell];

      for (int k = 0; k < vblocks[i]->num_faces; k++) { // faces

	int start = vblocks[i]->cell_faces_start[cell];
	int face = vblocks[i]->cell_faces[start + k];
	num_verts = vblocks[i]->faces[face].num_verts;

	num_face_verts.push_back(num_verts);

	for (int l = 0; l < num_verts; l++) { // vertices

	  int v = vblocks[i]->faces[face].verts[l];
	  vec3d s;
	  s.x = vblocks[i]->verts[3 * v];
	  s.y = vblocks[i]->verts[3 * v + 1];
	  s.z = vblocks[i]->verts[3 * v + 2];
	  m++;
	  verts.push_back(s);

	} // vertices

	n++;

      } // faces

    } // cells

  } // blocks

  // print the sites
  fprintf(stderr, "There are %lu sites\n\n", sites.size());
  for (int i = 0; i < (int)sites.size(); i++)
    fprintf(stderr, "[%.3f %.3f %.3f]\n", sites[i].x, sites[i].y, sites[i].z);

  // print the face vertices
  n = 0;
  fprintf(stderr, "\nThere are %lu faces\n\n", num_face_verts.size());
  for (int i = 0; i < (int)num_face_verts.size(); i++) {
    fprintf(stderr, "There are %d vertices in face %d\n", num_face_verts[i], i);
    for (int j = 0; j < num_face_verts[i]; j++) {
      fprintf(stderr, "[%.3f %.3f %.3f] \n", 
	      verts[n].x, verts[n].y, verts[n].z);
      n++;
    }
    fprintf(stderr, "\n");
  }

}
//--------------------------------------------------------------------------
