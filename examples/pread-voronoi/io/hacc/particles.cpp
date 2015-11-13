#include <iostream>
#include <stdexcept>
#include "particles.h"

#include <diy/mpi.hpp>
#include <diy/serialization.hpp>

void
io::hacc::
read_particles(MPI_Comm            comm_,
               const char*         infile,
               int                 rank,
               int                 size,
               std::vector<float>& particles,
               int                 sample_rate,
               Bounds*             data_bounds)
{
    // debug
    // fprintf(stderr, "infile %s rank %d size %d sr %d\n",
    //         infile, rank, size, sample_rate);

    // intialize reader
    gio::GenericIOReader *reader = new gio::GenericIOMPIReader();
    reader->SetFileName(infile);
    reader->SetCommunicator(comm_);
    reader->OpenAndReadHeader();

    // read generic I/O data
    vector<float> x, y, z;
    vector<int64_t> id;
    size_t num_particles = detail::read_gio(comm_, reader, x, y, z, id, data_bounds);

    // debug
    // fprintf(stderr,"%lu particles:\n", num_particles);
    // for (size_t i = 0; i < 10; i++)
    //     fprintf(stderr, "[%.3f %.3f %.3f] ", x[i], y[i], z[i]);
    // fprintf(stderr, "\n");

    // unique_ids is used to weed out duplicate particles, which sometimes
    // can happen in hacc
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
    // fprintf(stderr, "num_particles %lu sample_rate %d nu %lu\n", num_particles, sample_rate, nu);
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
         vector<int64_t>&      id,
         Bounds*               data_bounds)
{
    diy::mpi::communicator comm(comm_);
    double min[3], max[3]; // local block bounds

    // total number of blocks and local blocks
    int tot_blocks = reader->GetTotalNumberOfBlocks();
    int max_blocks = ceilf((float)tot_blocks / comm.size()); // max in any process
    vector<int>gids(max_blocks);

    // debug
    // fprintf(stderr, "tot_blocks %d max_blocks %d\n", tot_blocks, max_blocks);

    // contiguous block distribution is fine for now; will be redistributed anyway
    int nblocks = (comm.rank() < comm.size() - 1) ?
        nblocks = tot_blocks / comm.size() :
        nblocks = tot_blocks - (comm.size() - 1) * tot_blocks / comm.size();
    for (int b = 0; b < nblocks; ++b)
    {
        gids[b] = comm.rank() * tot_blocks / comm.size() + b;
        // debug
        // fprintf(stderr, "gids[%d] = %d\n", b, gids[b]);
    }

    // global data
    double origin[3], scale[3]; // global min and max as doubles
    uint64_t dims[3]; // global block dims as unint64
    reader->GetPhysOrigin(origin);
    reader->GetPhysScale(scale);
    reader->GetGlobalDimensions(dims);
    // debug
    // fprintf(stderr, "origin %.3lf %.3lf %.3lf scale %.3lf %.3lf %.3lf dims %llu %llu %llu\n",
    //         origin[0], origin[1], origin[2], scale[0], scale[1], scale[2],
    //         dims[0], dims[1], dims[2]);

    data_bounds->min[0] = origin[0];
    data_bounds->min[1] = origin[1];
    data_bounds->min[2] = origin[2];
    data_bounds->max[0] = scale[0];
    data_bounds->max[1] = scale[1];
    data_bounds->max[2] = scale[2];
    // block_dims[0] = dims[0];
    // block_dims[1] = dims[1];
    // block_dims[2] = dims[2];

    // read local blocks
    // TODO: skip the copy if there is only one block
    size_t ofst = 0;
    for (int b = 0; b < nblocks; b++) {
        // clear reader variables
        reader->ClearVariables();

        //  block bounds, NB, the reader wants lid, not gid
        double min[3], max[3];
        reader->GetBlockBounds(b, min, max);

        // number of particles in this block, NB, the reader wants gid now
        int num_particles = reader->GetNumberOfElements(gids[b]);

        // debug
        // fprintf(stderr, "gid %d num_particles = %d min [%.1f %.1f %.1f] max [%.1f %.1f %.1f]\n",
        //         gids[b], num_particles, min[0], min[1], min[2], max[0], max[1], max[2]);

        // padsize CRC for floats
        int floatpadsize = gio::CRCSize / sizeof(float);
        int idpadsize    = gio::CRCSize / sizeof(int64_t);

        // particles
        vector<float>   x0(num_particles  + floatpadsize);
        vector<float>   y0(num_particles  + floatpadsize);
        vector<float>   z0(num_particles  + floatpadsize);
        vector<int64_t> id0(num_particles + idpadsize);

        // clear variables and then register application arrays with the reader
        reader->AddVariable("x",  &x0[0],  gio::GenericIOBase::ValueHasExtraSpace);
        reader->AddVariable("y",  &y0[0],  gio::GenericIOBase::ValueHasExtraSpace);
        reader->AddVariable("z",  &z0[0],  gio::GenericIOBase::ValueHasExtraSpace);
        reader->AddVariable("id", &id0[0], gio::GenericIOBase::ValueHasExtraSpace);

        // read the particles
        reader->ReadBlock(gids[b]);

        // append particles from current block together to all particles for this process
        x.resize(x.size()   + num_particles);
        y.resize(y.size()   + num_particles);
        z.resize(z.size()   + num_particles);
        id.resize(id.size() + num_particles);
        for (size_t i = 0; i < num_particles; i++)
        {
            x[ofst  + i] = x0[i];
            y[ofst  + i] = y0[i];
            z[ofst  + i] = z0[i];
            id[ofst + i] = id0[i];
        }
        ofst += num_particles;
    }
    return x.size();
}

