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

      for (int k = 0; k < vblocks[i]->num_cell_faces[j]; k++) { // faces

	num_face_verts.push_back(vblocks[i]->num_face_verts[n]);

	for (int l = 0; l < vblocks[i]->num_face_verts[n]; l++) { // vertices

	  int v = vblocks[i]->face_verts[m];
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
