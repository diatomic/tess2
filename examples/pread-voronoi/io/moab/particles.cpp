#include <iostream>
#include <stdexcept>
#include "particles.h"

#include <diy/mpi.hpp>
#include <diy/serialization.hpp>



// read global domain bounds
void
io::moab::
read_domain(MPI_Comm            comm_,       // MPI comm
            const char*         infile,      // input file name
            Bounds&             domain)      // output global domain bounds
{
    // TODO: read domain bounds

    double origin[3], scale[3];              // global min and max as doubles

    // some hard-coded values for now
    domain.min[0] = -1.0;
    domain.min[1] = -1.0;
    domain.min[2] = -1.0;
    domain.max[0] = 4.0;
    domain.max[1] = 4.0;
    domain.max[2] = 4.0;
}

void
io::moab::
read_particles(MPI_Comm            comm_,       // MPI comm
               const char*         infile,      // input file name
               std::vector<float>& particles)   // output particles
{
    ErrorCode rval; // moab return value

    // load mesh
    Interface *mb = new Core();
    const char* options = ";;PARALLEL=READ_PART;PARALLEL_RESOLVE_SHARED_ENTS;"
        "PARTITION=MATERIAL_SET;PARTITION_DISTRIBUTE";
    EntityHandle file_set;
    rval = mb->create_meshset(MESHSET_SET, file_set); ERR;
    rval = mb->load_file(infile, &file_set, options); ERR;

    // get vertices (0-dimensional entities)
    Range pts;
    rval = mb->get_entities_by_dimension(file_set, 0, pts); ERR;

    // the point coordinates could be be extracted in one line, except that we
    // need to convert double to float for each coordinate; hence need to loop over the vertices
    // and copy each coordinate separately
    particles.reserve(pts.size() * 3);
    double pt[3];
    for (Range::iterator it = pts.begin(); it != pts.end(); it++)
    {
        rval = mb->get_coords(&(*it), 1, pt); ERR;
        particles.push_back(pt[0]);
        particles.push_back(pt[1]);
        particles.push_back(pt[2]);
    }

    // cleanup
    delete mb;
}
