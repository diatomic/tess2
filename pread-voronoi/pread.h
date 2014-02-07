#ifndef _PREAD_H
#define _PREAD_H

#include <vector>
#include <string>

void read_particles(char *infile, int rank, int size,
		    std::vector <float> &particles,
		    const std::vector<std::string>& coordinates,
		    int   swap);

#endif
