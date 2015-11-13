#ifndef __IO_HACC_PARTICLES_H__
#define __IO_HACC_PARTICLES_H__

#include "mpi.h"
#include <vector>
#include <set>
#include <stdio.h>
#include "GenericIODefinitions.hpp"
#include "GenericIOReader.h"
#include "GenericIOMPIReader.h"
#include "GenericIOPosixReader.h"
#include <math.h>

#include <diy/decomposition.hpp>

typedef     diy::ContinuousBounds         Bounds;

using namespace std;

namespace io
{

    namespace hacc
    {
        void read_domain(MPI_Comm            comm_,
                         const char*         infile,
                         Bounds&             domain);

        void read_particles(MPI_Comm            comm_,
                            const char*         infile,
                            int                 rank,
                            int                 size,
                            std::vector<float>& particles,
                            int                 sample_rate);

        namespace detail
        {
            size_t read_gio(MPI_Comm              comm_,
                            gio::GenericIOReader* reader,
                            vector<float>&        x,
                            vector<float>&        y,
                            vector<float>&        z,
                            vector<int64_t>&      id);
        }
    }
}

#endif // __IO_HACC_PARTICLES_H__
