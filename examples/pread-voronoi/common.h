#ifndef PREAD_COMMON_H
#define PREAD_COMMON_H

#include <tess/tet-neighbors.h>

/**
 * Deduplicate functionality: get rid of duplicate points, since downstream
 * code can't handle them.
 */

typedef     std::map<size_t, int>       DuplicateCountMap;

struct DedupPoint
{
    float data[3];
    bool	    operator<(const DedupPoint& other) const	    { return std::lexicographical_compare(data, data + 3, other.data, other.data + 3); }
    bool	    operator==(const DedupPoint& other) const	    { return std::equal(data, data + 3, other.data); }
};
void deduplicate(void* b_, const diy::Master::ProxyWithLink& cp, void* aux)
{
    dblock_t* b = static_cast<dblock_t*>(b_);

    if (!b->num_particles)
        return;

    // simple static_assert to ensure sizeof(Point) == sizeof(float[3]);
    // necessary to make this hack work
    typedef int static_assert_Point_size[sizeof(DedupPoint) == sizeof(float[3]) ? 1 : -1];
    DedupPoint* bg  = (DedupPoint*) &b->particles[0];
    DedupPoint* end = (DedupPoint*) &b->particles[3*b->num_particles];
    std::sort(bg,end);

    DuplicateCountMap* count = (DuplicateCountMap*) aux;
    DedupPoint* out = bg + 1;
    for (DedupPoint* it = bg + 1; it != end; ++it)
    {
        if (*it == *(it - 1))
            (*count)[out - bg - 1]++;
        else
        {
            *out = *it;
            ++out;
        }
    }
    b->num_orig_particles = b->num_particles = out - bg;

    if (!count->empty())
    {
        size_t total = 0;
        for (DuplicateCountMap::const_iterator it = count->begin(); it != count->end(); ++it)
            total += it->second;
        std::cout << b->gid << ": Found " << count->size() << " particles that appear more than once, with " << total << " total extra copies\n";
    }
}

// check if the particles fall inside the block bounds
void verify_particles(void* b_, const diy::Master::ProxyWithLink& cp, void*)
{
    dblock_t* b = static_cast<dblock_t*>(b_);

    // TP: testing 0-particle blocks by removing all the particles in block 0
    /* if (!b->gid) */
    /*     b->num_particles = b->num_orig_particles = 0; */

    /* fprintf(stderr, "gid %d has %d particles\n", b->gid, b->num_particles); */

    for (size_t i = 0; i < b->num_particles; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            if (b->particles[3*i + j] < b->mins[j] || b->particles[3*i + j] > b->maxs[j])
            {
                fprintf(stderr, "Particle outside the block: %f %f %f\n",
                        b->particles[3*i],
                        b->particles[3*i + 1],
                        b->particles[3*i + 2]);
                fprintf(stderr, "    block mins: %f %f %f\n",
                        b->mins[0],
                        b->mins[1],
                        b->mins[2]);
                fprintf(stderr, "    block maxs: %f %f %f\n",
                        b->maxs[0],
                        b->maxs[1],
                        b->maxs[2]);
                /* std::exit(1); */
            }
        }
    }
}

void enumerate_cells(void* b_, const diy::Master::ProxyWithLink& cp, void*)
{
    dblock_t* b = static_cast<dblock_t*>(b_);
    cp.collectives()->clear();

    size_t infinite = 0;
    for (size_t p = 0; p < b->num_orig_particles; ++p)
    {
      int t = b->vert_to_tet[p];
      if (t < 0)
	fprintf(stderr, "[%d] Warning: no matching tet for point %ld\n", cp.gid(), p);
      vector< pair<int, int> > nbrs;
      bool finite = neighbor_edges(nbrs, p, b->tets, t);
      if (!finite)
	++infinite;
    }
    //fprintf(stderr, "[%d] %lu infinite Voronoi cells\n", cp.gid(), infinite);

    cp.all_reduce(infinite, std::plus<size_t>());
}

// Swaps 8  bytes from 1-2-3-4-5-6-7-8 to 8-7-6-5-4-3-2-1 order.
// cast the input as a char and use on any 8 byte variable
void swap8(char *n) {

    char *n1;
    char c;

    n1 = n + 7;
    c = *n;
    *n = *n1;
    *n1 = c;

    n++;
    n1--;
    c = *n;
    *n = *n1;
    *n1 = c;

    n++;
    n1--;
    c = *n;
    *n = *n1;
    *n1 = c;

    n++;
    n1--;
    c = *n;
    *n = *n1;
    *n1 = c;

}

// Swaps 4 bytes from 1-2-3-4 to 4-3-2-1 order.
// cast the input as a char and use on any 4 byte variable
void swap4(char *n) {

    char *n1;
    char c;

    n1 = n + 3;
    c = *n;
    *n = *n1;
    *n1 = c;

    n++;
    n1--;
    c = *n;
    *n = *n1;
    *n1 = c;

}

// Swaps 2 bytes from 1-2 to 2-1 order.
// cast the input as a char and use on any 2 byte variable
void swap2(char *n){

    char c;

    c = *n;
    *n = n[1];
    n[1] = c;

}

void swap_bytes(void *p,                           // address of items
                int nitems,                        // number of items
                int item_size)                     // 2, 4, or 8 bytes (returns quietly if 1 byte)
{
    int i;
    char*n = (char*)p;
    switch(item_size) {
    case 1:
        return;
        break;
    case 2:
        for (i = 0; i < nitems; i++) {
            swap2(n);
            n += 2;
        }
        break;
    case 4:
        for (i = 0; i < nitems; i++) {
            swap4(n);
            n += 4;
        }
        break;
    case 8:
        for (i = 0; i < nitems; i++) {
            swap8(n);
            n += 8;
        }
        break;
    default:
        fprintf(stderr, "Error: size of data must be either 1, 2, 4, or 8 bytes per item\n");

    }
}

#endif
