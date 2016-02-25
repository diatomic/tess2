#ifndef __IO_MOAB_PARTICLES_H__
#define __IO_MOAB_PARTICLES_H__

#include "mpi.h"

#include <vector>
#include <set>
#include <stdio.h>
#include <math.h>

// TODO: which of the following do I actually need?
#include "iMesh.h"
#include "MBiMesh.hpp"
#include "moab/Core.hpp"
#include "moab/Range.hpp"
#include "MBTagConventions.hpp"
#include "moab/ParallelComm.hpp"
#include "moab/HomXform.hpp"
#include "moab/ReadUtilIface.hpp"
#include "Coupler.hpp"

#include <diy/decomposition.hpp>

#define ERR {if(rval!=MB_SUCCESS)printf("MOAB error at line %d in %s\n", __LINE__, __FILE__);}

typedef     diy::ContinuousBounds         Bounds;

using namespace std;

namespace io
{

    namespace moab
    {
        void read_domain(MPI_Comm            comm_,
                         const char*         infile,
                         Bounds&             domain);

        void read_particles(MPI_Comm            comm_,
                            const char*         infile,
                            std::vector<float>& particles);
    }
}

#endif // __IO_MOAB_PARTICLES_H__
