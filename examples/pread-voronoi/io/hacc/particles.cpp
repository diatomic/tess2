#include <iostream>
#include <stdexcept>
#include "particles.h"

#include <diy/mpi.hpp>
#include <diy/serialization.hpp>

void
io::hacc::
read_domain(MPI_Comm            comm_,
            const char*         infile,
            Bounds&             domain)
{
    gio::GenericIOReader *reader = new gio::GenericIOMPIReader();
    reader->SetFileName(infile);
    reader->SetCommunicator(comm_);
    reader->OpenAndReadHeader();

    double origin[3], scale[3];              // global min and max as doubles
    reader->GetPhysOrigin(origin);
    reader->GetPhysScale(scale);
    domain.min[0] = origin[0];
    domain.min[1] = origin[1];
    domain.min[2] = origin[2];
    domain.max[0] = scale[0];
    domain.max[1] = scale[1];
    domain.max[2] = scale[2];

    reader->Close();
    delete reader;
}

void
io::hacc::
read_particles(MPI_Comm            comm_,
               const char*         infile,
               int                 rank,
               int                 size,
               std::vector<float>& particles,
               int                 sample_rate)
{
    // intialize reader
    gio::GenericIOReader *reader = new gio::GenericIOMPIReader();
    reader->SetFileName(infile);
    reader->SetCommunicator(comm_);
    reader->OpenAndReadHeader();

    // read generic I/O data
    vector<float> x, y, z;
    vector<int64_t> id;
    size_t num_particles = detail::read_gio(comm_, reader, x, y, z, id);

    // unique_ids is used to weed out duplicate particles, which sometimes happens in hacc
    set <int64_t> unique_ids;

    // package particles, sampling as specified and filtering out duplicates
    num_particles /= sample_rate;
    particles.resize(num_particles * 3);
    size_t nu = 0; // number of unique points
    for (size_t i = 0; i < num_particles; i++) {
        if (unique_ids.find(id[i * sample_rate]) == unique_ids.end()) {
            particles[3 * nu]     = x[i * sample_rate];
            particles[3 * nu + 1] = y[i * sample_rate];
            particles[3 * nu + 2] = z[i * sample_rate];
            unique_ids.insert(id[i * sample_rate]);
            nu++;
        }
    }

    // debug
    diy::mpi::communicator comm(comm_);
    if (nu < num_particles)
        fprintf(stderr, "%lu duplicate particles found and removed in rank %d\n",
                num_particles - nu, comm.rank());

    // cleanup
    reader->Close();
    delete reader;
}

size_t
io::hacc::detail::
read_gio(MPI_Comm              comm_,
         gio::GenericIOReader* reader,
         vector<float>&        x,
         vector<float>&        y,
         vector<float>&        z,
         vector<int64_t>&      id)
{
    diy::mpi::communicator comm(comm_);
    double min[3], max[3]; // local block bounds

    // total number of blocks and local blocks
    int tot_blocks = reader->GetTotalNumberOfBlocks();
    int max_blocks = ceilf((float)tot_blocks / comm.size()); // max in any process
    vector<int>gids(max_blocks);

    // contiguous block distribution is fine for now; will be redistributed anyway
    int nblocks = (comm.rank() < comm.size() - 1) ?
        nblocks = tot_blocks / comm.size() :
        nblocks = tot_blocks - (comm.size() - 1) * tot_blocks / comm.size();
    for (int b = 0; b < nblocks; ++b)
        gids[b] = comm.rank() * tot_blocks / comm.size() + b;

    // read local blocks
    size_t tot_p = 0;                        // total number particles so far
    for (int b = 0; b < nblocks; b++) {
        reader->ClearVariables();            // clear reader variables

        double min[3], max[3];               // block bounds, NB, the reader wants lid, not gid
        reader->GetBlockBounds(b, min, max);

        // number of particles in this block, NB, the reader wants gid now
        int num_particles = reader->GetNumberOfElements(gids[b]);

        // padsize CRC for floats
        int floatpadsize = gio::CRCSize / sizeof(float);
        int idpadsize    = gio::CRCSize / sizeof(int64_t);

        // particles
        if (nblocks > 1)
        {
            vector<float>   x0(num_particles  + floatpadsize);
            vector<float>   y0(num_particles  + floatpadsize);
            vector<float>   z0(num_particles  + floatpadsize);
            vector<int64_t> id0(num_particles + idpadsize);
            reader->AddVariable("x",  &x0[0],  gio::GenericIOBase::ValueHasExtraSpace);
            reader->AddVariable("y",  &y0[0],  gio::GenericIOBase::ValueHasExtraSpace);
            reader->AddVariable("z",  &z0[0],  gio::GenericIOBase::ValueHasExtraSpace);
            reader->AddVariable("id", &id0[0], gio::GenericIOBase::ValueHasExtraSpace);

            reader->ReadBlock(gids[b]);      // read the particles

            // append particles from current block together to all particles for this process
            x.resize(x.size()   + num_particles);
            y.resize(y.size()   + num_particles);
            z.resize(z.size()   + num_particles);
            id.resize(id.size() + num_particles);
            for (size_t i = 0; i < num_particles; i++)
            {
                x[tot_p  + i] = x0[i];
                y[tot_p  + i] = y0[i];
                z[tot_p  + i] = z0[i];
                id[tot_p + i] = id0[i];
            }
            tot_p += num_particles;
        }
        else
        {
            x.resize(num_particles  + floatpadsize);
            y.resize(num_particles  + floatpadsize);
            z.resize(num_particles  + floatpadsize);
            id.resize(num_particles + idpadsize);
            reader->AddVariable("x",  &x[0],  gio::GenericIOBase::ValueHasExtraSpace);
            reader->AddVariable("y",  &y[0],  gio::GenericIOBase::ValueHasExtraSpace);
            reader->AddVariable("z",  &z[0],  gio::GenericIOBase::ValueHasExtraSpace);
            reader->AddVariable("id", &id[0], gio::GenericIOBase::ValueHasExtraSpace);

            reader->ReadBlock(gids[b]);      // read the particles
            tot_p = num_particles;
        }
    }
    return tot_p;
}
