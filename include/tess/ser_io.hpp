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

#ifndef _SER_IO
#define _SER_IO

#include "delaunay.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <vector>
#include <string.h>

#if 0 // not using swapping or compression for now

#include "swap.hpp"
#include "zlib.h"

#endif

using namespace std;

#define DIY_MAX_HDR_ELEMENTS 256
#define CHUNK 262144 // chunk size for zlib

//------------------------------------------------------------------------------

class SER_IO {

 public:

  SER_IO(bool swap_bytes = false) { 
    this->swap_bytes = swap_bytes;
  }
  ~SER_IO(){};
  int ReadAllBlocks(const char *filename, dblock_t** &blocks, 
		    bool compress = false); 

 private:

  void ReadFooter(FILE*& fd, int64_t*& ftr, int& tb);
  void ReadHeader(FILE *fd, int *hdr, int64_t ofst);
  int CopyHeader(unsigned char *in_buf, int *hdr);
  void ReadBlock(FILE *fd, dblock_t* &v, int64_t ofst);

  int dim; // number of dimensions in the dataset
  bool swap_bytes; // whether to swap bytes for endian conversion

};

#endif
