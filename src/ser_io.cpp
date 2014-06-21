//------------------------------------------------------------------------------
//
// serial io class
//
// Tom Peterka
// Argonne National Laboratory
// 9700 S. Cass Ave.
// Argonne, IL 60439
// tpeterka@mcs.anl.gov
//
// All rights reserved. May not be used, modified, or copied
// without permission
//
//--------------------------------------------------------------------------

#include "tess/ser_io.hpp"
#include "tess/swap.hpp"

//----------------------------------------------------------------------------
//
// reads all delauanay blocks in a file
//
// filename: input filename
// blocks: pointers to blocks (output)
// compress: whether file is compressed (optional, default = not compressed)
//
// side effects: allocates space for the new blocks
//
// returns: total number of blocks
//
int SER_IO::ReadAllBlocks(const char *filename, dblock_t** &blocks,
			  bool compress) {

  compress = compress; // quiet compiler warning

  FILE *fd;
  int64_t *ftr; // footer
  int tb; // total number of blocks

  fd = fopen(filename, "r");
  assert(fd != NULL);

  ReadFooter(fd, ftr, tb);

  blocks = new dblock_t*[tb];

  for (int i = 0; i < tb; i++)
    ReadBlock(fd, blocks[i], ftr[i]);

  fclose(fd);

  delete[] ftr;
  return tb;

}
//----------------------------------------------------------------------------
//
// reads the file footer
// footer in file is always ordered by global block id
// output footer is in the same order
//
// fd: open file
// ftr: footer data (output)
// tb: total number of blocks in the file (output)
//
// side effects: allocates ftr
//
void SER_IO::ReadFooter(FILE*& fd, int64_t*& ftr, int& tb) {

  int ofst;
  int count;
  int64_t temp;

  ofst = sizeof(int64_t);
  fseek(fd, -ofst, SEEK_END);
  count = fread(&temp, sizeof(int64_t), 1, fd); // total number of blocks
  assert(count == 1); // total number of blocks

  if (swap_bytes)
    Swap((char *)&temp, 1, sizeof(int64_t));
  tb = temp;

  if (tb > 0) {

    ftr = new int64_t[tb];
    ofst = (tb + 1) * sizeof(int64_t);
    fseek(fd, -ofst, SEEK_END);
    count = fread(ftr, sizeof(int64_t), tb, fd);
    assert(count == tb);

    if (swap_bytes)
      Swap((char *)ftr, tb, sizeof(int64_t));

  }

}
//----------------------------------------------------------------------------
//
// reads the header for one block from a file
//
// fd: open file
// hdr: allocated header data
// ofst: location in file of the header (bytes)
//
void SER_IO::ReadHeader(FILE *fd, int *hdr, int64_t ofst) {

  int count;

  fseek(fd, ofst, SEEK_SET);
  count = fread(hdr, sizeof(int), DIY_MAX_HDR_ELEMENTS, fd);
  assert(count == DIY_MAX_HDR_ELEMENTS);

  if (swap_bytes)
    Swap((char *)hdr, DIY_MAX_HDR_ELEMENTS, sizeof(int));

}
//----------------------------------------------------------------------------
//
// Copies the header for one block from a buffer in memory
//
// in_buf: input buffer location
// hdr: allocated header data
//
// returns: number of bytes copies
//
int SER_IO::CopyHeader(unsigned char *in_buf, int *hdr) {

  memcpy(hdr, in_buf, DIY_MAX_HDR_ELEMENTS * sizeof(int));

  if (swap_bytes)
    Swap((char *)hdr, DIY_MAX_HDR_ELEMENTS, sizeof(int));

  return(DIY_MAX_HDR_ELEMENTS * sizeof(int));

}
//----------------------------------------------------------------------------
//
// reads one delaunay block from a file
//
// fd: open file
// d: pointer to output block
// ofst: file file pointer to start of header for this block
//
// side-effects: allocates block
//
void SER_IO::ReadBlock(FILE *fd, dblock_t* &d, int64_t ofst) {

  // get header info
  int hdr[DIY_MAX_HDR_ELEMENTS];
  ReadHeader(fd, hdr, ofst);

  // create block
  d = new dblock_t;
  d->num_orig_particles = hdr[NUM_ORIG_PARTICLES];
  d->num_tets = hdr[NUM_TETS];
  d->num_rem_tet_verts = hdr[NUM_REM_TET_VERTS];

  if (d->num_orig_particles > 0) {
    d->particles = new float[3 * d->num_orig_particles];
    d->vert_to_tet = new int[d->num_orig_particles];
  }
  if (d->num_tets > 0)
    d->tets = new tet_t[d->num_tets];
  if (d->num_rem_tet_verts > 0)
    d->rem_tet_verts = new remote_vert_t[d->num_rem_tet_verts];

  fread(d->mins, sizeof(float), 3, fd);
  fread(d->particles, sizeof(float), 3 * d->num_orig_particles, fd);
  fread(d->vert_to_tet, sizeof(int), d->num_orig_particles, fd);
  fread(d->tets, sizeof(struct tet_t), d->num_tets, fd);
  fread(d->rem_tet_verts, sizeof(struct remote_vert_t), 
	d->num_rem_tet_verts, fd);
  fread(d->maxs, sizeof(float), 3, fd);

  if (swap_bytes) {

    Swap((char *)d->mins, 3, sizeof(float));
    Swap((char *)d->particles, 3 * d->num_orig_particles, sizeof(float));
    Swap((char *)d->vert_to_tet, d->num_orig_particles, sizeof(int));

    // tets and rem_tet_verts are structs, need to swap each element
    // need to swap items individually (annoying)
    for (int i = 0; i < d->num_tets; i++) {
      Swap((char *)d->tets[i].verts, 4, sizeof(int));
      Swap((char *)d->tets[i].tets, 4, sizeof(int));
    }
    for (int i = 0; i < d->num_rem_tet_verts; i++) {
      Swap((char *)&(d->rem_tet_verts[i].gid), 1, sizeof(int));
      Swap((char *)&(d->rem_tet_verts[i].nid), 1, sizeof(int));
      // no need to swap dir, unsigned char
    }

    Swap((char *)d->maxs, 3, sizeof(float));

  }

}
//----------------------------------------------------------------------------
