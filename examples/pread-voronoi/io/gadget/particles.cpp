#include <iostream>
#include <stdexcept>
#include "particles.h"

#include <diy/mpi.hpp>
#include <diy/serialization.hpp>


ssize_t
io::gadget::
read_particles(MPI_Comm comm_,
               const char *infile, int rank, int size,
               std::vector<float>& particles,
               const std::vector<std::string>& coordinates)
{
    typedef         std::vector<float>                              PointContainer;
    typedef         std::vector<size_t>                             IntVector;

    diy::mpi::communicator comm(comm_);

    IntVector                   individual_count_, cumulative_count_;
    std::vector<std::string>    data_files_;

    int                         format;
    bool                        swap;
    int                         blockSize, blockSize2;

    std::string                 filename_(infile);
    std::vector<unsigned>       coordinates_;

    using namespace detail;

    for(size_t i = 0; i < coordinates.size(); ++i)
        coordinates_.push_back(boost::lexical_cast<unsigned>(coordinates[i]));

    if (comm.rank() == 0)
    {
      bf::path p(filename_);
      if (!bf::exists(p))
	throw std::runtime_error("Could not open the directory: " + boost::lexical_cast<std::string>(p));

      for (bf::directory_iterator cur = bf::directory_iterator(p); cur != bf::directory_iterator(); ++cur)
	  data_files_.push_back(cur->path().string());
      boost::sort(data_files_);

      for(size_t i = 0; i < data_files_.size(); ++i)
      {
	  const std::string& fn = data_files_[i];
	  std::string gadget2;
	  struct GadgetHeader header;
	  format = GADGET_1;
	  swap = false;

	  std::ifstream igStr(fn.c_str(), std::ios::in | std::ios::binary);
	  std::ifstream* gStr = &igStr;
	  if (gStr->fail())
	  {
	      std::cout << "File: " << fn << " cannot be opened" << std::endl;
	      exit(-1);
	  }

	  // Set the gadget format type by reading the first 4 byte integer
	  // If it is not "256" or "65536" then gadget-2 format with 16 bytes in front
	  readData(swap, (void*) &blockSize, GADGET_SKIP, 1, gStr);
	  if (blockSize != GADGET_HEADER_SIZE && blockSize != GADGET_HEADER_SIZE_SWP)
	  {
	    format = GADGET_2;
	    gadget2 = readString(gStr, GADGET_2_SKIP - GADGET_SKIP);
	    readData(swap, (void*) &blockSize, GADGET_SKIP, 1, gStr);
	  }

	  // Set the swap type
	  if (blockSize != GADGET_HEADER_SIZE)
	  {
	    swap = true;
	    blockSize = GADGET_HEADER_SIZE;
	  }

	  // Read the Gadget header
	  readData(swap, (void*) &header.npart[0],
				 sizeof(int), NUM_GADGET_TYPES, gStr);
	  readData(swap, (void*) &header.mass[0],
				 sizeof(double), NUM_GADGET_TYPES, gStr);
	  readData(swap, (void*) &header.time, sizeof(double), 1, gStr);
	  readData(swap, (void*) &header.redshift, sizeof(double), 1, gStr);
	  readData(swap, (void*) &header.flag_sfr, sizeof(int), 1, gStr);
	  readData(swap, (void*) &header.flag_feedback, sizeof(int), 1, gStr);
	  readData(swap, (void*) &header.npartTotal[0],
				 sizeof(int), NUM_GADGET_TYPES, gStr);
	  readData(swap, (void*) &header.flag_cooling, sizeof(int), 1, gStr);
	  readData(swap, (void*) &header.num_files, sizeof(int), 1, gStr);
	  readData(swap, (void*) &header.BoxSize, sizeof(double), 1, gStr);
	  readData(swap, (void*) &header.Omega0, sizeof(double), 1, gStr);
	  readData(swap, (void*) &header.OmegaLambda, sizeof(double), 1, gStr);
	  readData(swap, (void*) &header.HubbleParam, sizeof(double), 1, gStr);
	  readData(swap, (void*) &header.flag_stellarage, sizeof(int), 1, gStr);
	  readData(swap, (void*) &header.flag_metals, sizeof(int), 1, gStr);
	  readData(swap, (void*) &header.HighWord[0],
				 sizeof(int), NUM_GADGET_TYPES, gStr);
	  readData(swap, (void*) &header.flag_entropy, sizeof(int), 1, gStr);
	  std::string fill = readString(gStr, 60);
	  strcpy(&header.fill[0], fill.c_str());

	  // Read the Gadget header size to verify block
	  readData(swap, (void*) &blockSize2, GADGET_SKIP, 1, gStr);
	  if (blockSize != blockSize2)
	      throw std::runtime_error("Error reading header: end position is wrong");

	  // Every type particle will have location, velocity and tag so sum up
	  long int particleCount = 0;
	  for (int i = 0; i < NUM_GADGET_TYPES; i++)
	    particleCount += header.npart[i];

	  individual_count_.push_back(particleCount);
      }

      diy::MemoryBuffer	bb;
      diy::save(bb, data_files_);
      diy::save(bb, individual_count_);
      diy::save(bb, format);
      diy::save(bb, swap);
      diy::save(bb, blockSize);
      diy::save(bb, blockSize2);
      diy::mpi::broadcast(comm, bb.buffer, 0);
    } else
    {
      diy::MemoryBuffer bb;
      diy::mpi::broadcast(comm, bb.buffer, 0);
      diy::load(bb, data_files_);
      diy::load(bb, individual_count_);
      diy::load(bb, format);
      diy::load(bb, swap);
      diy::load(bb, blockSize);
      diy::load(bb, blockSize2);
    }

    size_t total = 0;
    for(size_t i = 0; i < individual_count_.size(); ++i)
    {
        cumulative_count_.push_back(total);
        total += individual_count_[i];
    }


    size_t  offset = total/size*rank;
    size_t  count  = (rank != size - 1 ? total/size : total - total/size*rank);

    size_t o_offset = offset;

    particles.resize(count * DIMENSION);

    size_t grd = boost::lower_bound(cumulative_count_, offset) - cumulative_count_.begin();
    if (grd == cumulative_count_.size() || offset < cumulative_count_[grd])
        --grd;
    offset -= cumulative_count_[grd];

    size_t i = 0;
    while (count > 0)
    {
        std::ifstream ParticleFile(data_files_[grd].c_str(), std::ios::in | std::ios::binary);
        std::ifstream* gStr = &ParticleFile;

        size_t particleCount = individual_count_[grd];

        size_t to_read;
        if (individual_count_[grd] - offset <= count)
            to_read = individual_count_[grd] - offset;
        else
            to_read = count;

        if (offset + to_read > individual_count_[grd])
	  throw std::runtime_error("Cannot read more particles than are in the file");

        readString(gStr, GADGET_HEADER_SIZE + sizeof(blockSize) + sizeof(blockSize2));
        if (format == GADGET_2)
          readString(gStr, GADGET_2_SKIP);

        std::vector<POSVEL_T> location(particleCount * DIMENSION);

        // XXX: reading everything and then extracting a subset is not the most efficient strategy
        readData(swap, (void*) &blockSize, GADGET_SKIP, 1, gStr);
        readData(swap, (void*) &location[0],
                               sizeof(POSVEL_T), particleCount*DIMENSION, gStr);
        readData(swap, (void*) &blockSize2, GADGET_SKIP, 1, gStr);
        if (blockSize != blockSize2)
          throw std::runtime_error("Error reading locations: end position is wrong " +
				    boost::lexical_cast<std::string>(blockSize) +
				    " vs " +
				    boost::lexical_cast<std::string>(blockSize2));

        for (size_t j = 0; j < to_read; ++j, ++i)
            for (size_t k = 0; k < DIMENSION; ++k)
                particles[i*DIMENSION + k] = location[(offset + j)*DIMENSION + coordinates_[k]];

        offset = 0;
        count -= to_read;
        ++grd;
    }

    return total;
}

std::string
io::gadget::detail::
readString(std::ifstream* inStr, int size)
{
   char* buffer = new char[size + 1];
   inStr->read(buffer, size);
   buffer[size] = '\0';

   // Make sure string has legal values
   if (isalnum(buffer[0]) == 0)
      buffer[0] = '\0';
   for (int i = 1; i < size; i++)
      if (isprint(buffer[i]) == 0)
         buffer[i] = '\0';

   std::string retString = buffer;
   delete [] buffer;
   return retString;
}

void
io::gadget::detail::
readData(
        bool swap,
        void* data,
        unsigned long dataSize,
        unsigned long dataCount,
        std::ifstream* inStr)
{
   // Read all the data from the file
   inStr->read(reinterpret_cast<char*>(data), dataSize*dataCount);

   if (swap == true) {

      // Byte swap each integer
      char* dataPtr = (char*) data;
      char temp;
      for (unsigned long item = 0; item < dataCount; item++) {

         // Do a byte-by-byte swap, reversing the order.
         for (unsigned int i = 0; i < dataSize / 2; i++) {
            temp = dataPtr[i];
            dataPtr[i] = dataPtr[dataSize - 1 - i];
            dataPtr[dataSize - 1 - i] = temp;
         }
         dataPtr += dataSize;
      }
   }
}
