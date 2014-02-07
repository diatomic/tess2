#include <iostream>

#include <H5Cpp.h>

#include "pread.h"
#include <swap.hpp>

void read_particles(char *infile, int rank, int size,
		    std::vector <float> &particles,
		    const std::vector <std::string>& coordinates,
		    int   swap)
{
  H5::H5File file(infile, H5F_ACC_RDONLY);
  
  H5::DataSet	dataset	    = file.openDataSet(coordinates[0]);
  H5::DataSpace dataspace   = dataset.getSpace();
  int           r	    = dataspace.getSimpleExtentNdims();     // should be 1
  std::vector<hsize_t>        dims(r);
  int           ndims       = dataspace.getSimpleExtentDims(&dims[0], NULL);
  hsize_t	count	    = dims[0];
  
  hsize_t  offset = count/size*rank;
  hsize_t  local_count = (rank != size - 1 ? count/size : count - count/size*rank);
  
  std::vector<float>  tmp(local_count);         // optimizes reading
  particles.resize(3*local_count);

  for (size_t i = 0; i < 3; ++i)
  {
      std::string   c		= coordinates[i];
      H5::DataSet   dataset     = file.openDataSet(c);
      H5::DataSpace dataspace   = dataset.getSpace();
      dataspace.selectHyperslab(H5S_SELECT_SET, &local_count, &offset);

      H5::DataSpace mspace(1,&local_count);
      dataset.read(&tmp[0], H5::PredType::NATIVE_FLOAT, mspace, dataspace);

      // stripe into particles
      for (size_t j = 0; j < local_count; ++j)
          particles[3*j + i] = tmp[j];			// probably wildly cache inefficient
  }

  if (swap)
    Swap((char*) &particles[0], local_count, sizeof(float));
}
