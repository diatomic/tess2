#include <iostream>
#include <cassert>

#include <hdf5.h>

#include "pread.h"
#include <tess/swap.hpp>

#ifndef H5_HAVE_PARALLEL
#warning Parallel HDF5 not available, using serial version
#endif

void read_particles(MPI_Comm comm,
                    const char *infile, int rank, int size,
                    std::vector <float> &particles,
                    const std::vector <std::string>& coordinates)
{
  herr_t status;

#ifdef H5_HAVE_PARALLEL
  MPI_Info  info        = MPI_INFO_NULL;
  hid_t     acc_tpl1    = H5Pcreate (H5P_FILE_ACCESS);
  assert(acc_tpl1 != -1);
  herr_t    ret         = H5Pset_fapl_mpio(acc_tpl1, comm, info);       // set up parallel access with communicator
  assert(ret != -1);

  hid_t     file_id       = H5Fopen(infile, H5F_ACC_RDONLY, acc_tpl1);
#else
  hid_t     file_id       = H5Fopen(infile, H5F_ACC_RDONLY, H5P_DEFAULT);
#endif
  hid_t     dataset_id    = H5Dopen2(file_id, coordinates[0].c_str(), H5P_DEFAULT);
  hid_t     dataspace_id  = H5Dget_space(dataset_id);

  int       r     = H5Sget_simple_extent_ndims(dataspace_id);       // should be 1
  std::vector<hsize_t>  dims(r);
  int       ndims = H5Sget_simple_extent_dims(dataspace_id, &dims[0], NULL);
  hsize_t   count = dims[0];

  status = H5Sclose(dataspace_id);
  status = H5Dclose(dataset_id);

  hsize_t  offset = count/size*rank;
  hsize_t  local_count = (rank != size - 1 ? count/size : count - count/size*rank);

  std::vector<float>  tmp(local_count);         // optimizes reading
  particles.resize(3*local_count);

  for (size_t i = 0; i < 3; ++i)
  {
    std::string   c             = coordinates[i];
    hid_t         dataset_id    = H5Dopen2(file_id, c.c_str(), H5P_DEFAULT);
    hid_t         dataspace_id  = H5Dget_space(dataset_id);

    status = H5Sselect_hyperslab (dataspace_id, H5S_SELECT_SET, &offset, NULL, &local_count, NULL);

    hid_t         memspace_id   = H5Screate_simple(1, &local_count, NULL);
    status = H5Dread (dataset_id, H5T_NATIVE_FLOAT, memspace_id, dataspace_id, H5P_DEFAULT, &tmp[0]);

    // stripe into particles
    for (size_t j = 0; j < local_count; ++j)
        particles[3*j + i] = tmp[j];                  // probably wildly cache inefficient

    status = H5Sclose(memspace_id);
    status = H5Sclose(dataspace_id);
    status = H5Dclose(dataset_id);
  }

  status = H5Fclose(file_id);
#ifdef H5_HAVE_PARALLEL
  status = H5Pclose(acc_tpl1);
#endif
}
