// ---------------------------------------------------------------------------
//
// delaunay block C++ version
//
// Tom Peterka
// Argonne National Laboratory
// 9700 S. Cass Ave.
// Argonne, IL 60439
// tpeterka@mcs.anl.gov
//
// ---------------------------------------------------------------------------

#ifndef _DELAUNAY_HPP
#define _DELAUNAY_HPP

#include <diy/types.hpp>

struct DBlock : dblock_t
{
    diy::Bounds<float> bounds;                    // local block extents
    diy::Bounds<float> data_bounds;               // global data extents
    diy::Bounds<float> box;                       // box in current round of point redistribution
};

#endif
