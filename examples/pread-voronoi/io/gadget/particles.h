#ifndef __IO_GADGET_PARTICLES_H__
#define __IO_GADGET_PARTICLES_H__

#include <fstream>
#include <string>

#include "mpi.h"

#include <boost/range/algorithm/lower_bound.hpp>
#include <boost/range/algorithm/sort.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>

namespace io
{

namespace gadget
{

ssize_t read_particles(MPI_Comm comm,
                       const char *infile, int rank, int size,
                       std::vector<float>& particles,
                       const std::vector<std::string>& coordinates);

namespace bf = boost::filesystem;

namespace detail
{
#ifdef ID_64
   typedef      int64_t ID_T;           // Particle and halo ids
#else
   typedef      int32_t ID_T;           // Particle and halo ids
#endif

#ifdef POSVEL_64
   typedef      double  POSVEL_T;       // Position,velocity
#else
   typedef      float   POSVEL_T;       // Position,velocity
#endif

const int   DIMENSION   = 3;

const int NUM_GADGET_TYPES      = 6;    // Types of gadget particles

const int GADGET_HEADER_SIZE    = 256;  // Size when the endian matches
const int GADGET_HEADER_SIZE_SWP= 65536;// Size when the endian doesn't match
const int GADGET_FILL           = 60;   // Current fill to HEADER SIZE
const int GADGET_SKIP           = 4;    // Bytes the indicate block size
const int GADGET_2_SKIP         = 16;   // Extra bytes in gadget-2

const int GADGET_1              = 1;
const int GADGET_2              = 2;


struct GadgetHeader {
  int      npart[NUM_GADGET_TYPES];
  double   mass[NUM_GADGET_TYPES];
  double   time;
  double   redshift;
  int      flag_sfr;
  int      flag_feedback;
  int      npartTotal[NUM_GADGET_TYPES];
  int      flag_cooling;
  int      num_files;
  double   BoxSize;
  double   Omega0;
  double   OmegaLambda;
  double   HubbleParam;
  int      flag_stellarage;
  int      flag_metals;
  int      HighWord[NUM_GADGET_TYPES];
  int      flag_entropy;
  char     fill[GADGET_FILL];
};

void readData(
        bool swap,
        void* data,
        unsigned long dataSize,
        unsigned long dataCount,
        std::ifstream* inStr);

std::string readString(std::ifstream* inStr, int size);

}

}
}
#endif // __IO_GADGET_PARTICLES_H__
