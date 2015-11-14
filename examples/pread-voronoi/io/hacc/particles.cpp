#include <iostream>
#include <stdexcept>
#include "particles.h"

#include <diy/mpi.hpp>
#include <diy/serialization.hpp>

void
io::hacc::
read_domain(MPI_Comm            comm_,       // MPI comm
            const char*         infile,      // input file name
            Bounds&             domain)      // output global domain bounds
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
read_particles(MPI_Comm            comm_,       // MPI comm
               const char*         infile,      // input file name
               std::vector<float>& particles,   // output particles
               int                 sample_rate) // output sample rate
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

    // cleanup
    reader->Close();
    delete reader;
}

size_t
io::hacc::detail::
read_gio(MPI_Comm              comm_,        // MPI comm
         gio::GenericIOReader* reader,       // generic io reader
         vector<float>&        x,            // output x coords
         vector<float>&        y,            // output y coords
         vector<float>&        z,            // output z coords
         vector<int64_t>&      id)           // output particle ids
{
    diy::mpi::communicator comm(comm_);

    // genericio can only handle one block per process
    // even though in theory I should be able to concatenate multiple genericio block reads
    // it returns 0 particles when there are not enough mpi ranks for genericio blocks
    assert(comm.size() == reader->GetTotalNumberOfBlocks());

    // read local genericio blocks
    reader->ClearVariables();            // clear reader variables

    // number of particles in this block
    size_t num_particles = reader->GetNumberOfElements(comm.rank());

    // padsize CRC for floats
    int floatpadsize = gio::CRCSize / sizeof(float);
    int idpadsize    = gio::CRCSize / sizeof(int64_t);

    // particles
    x.resize(num_particles  + floatpadsize);
    y.resize(num_particles  + floatpadsize);
    z.resize(num_particles  + floatpadsize);
    id.resize(num_particles + idpadsize);
    reader->AddVariable("x",  &x[0],  gio::GenericIOBase::ValueHasExtraSpace);
    reader->AddVariable("y",  &y[0],  gio::GenericIOBase::ValueHasExtraSpace);
    reader->AddVariable("z",  &z[0],  gio::GenericIOBase::ValueHasExtraSpace);
    reader->AddVariable("id", &id[0], gio::GenericIOBase::ValueHasExtraSpace);

    reader->ReadBlock(comm.rank());      // read the particles

    return num_particles;
}
