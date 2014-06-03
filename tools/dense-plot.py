#! /usr/bin/python

#---------------------------------------------------------------------------
#
# plots 2D density 
#
# Tom Peterka
# Argonne National Laboratory
# 9700 S. Cass Ave.
# Argonne, IL 60439
# tpeterka@mcs.anl.gov
#
# (C) 2013 by Argonne National Laboratory.
# See COPYRIGHT in top-level directory.
#
#--------------------------------------------------------------------------

import numpy as np
import optparse
import struct
from matplotlib import mpl, pyplot

#-----

# parse arguments
parser = optparse.OptionParser()
parser.add_option("--raw", dest="infile1", help="raw (binary) density input file 1")
parser.add_option("--numpts", dest="numpts", help="number of grid point position in one axis (same for x and y)")
(options, args) = parser.parse_args()

#-----

# construct raw density x, y, d arrays

# using np.fromfile() does not work correctly on my mac, so I need to read
# the file one value at a time and append to the array

f1 = open(options.infile1, "rb")
s = int(options.numpts) # size in one dimension
ss = s * s # total size in two dimensions
d = np.zeros((s, s), dtype = np.float32) # empty array for the density
for i in range(s):
	for j in range(s):
		d1 = f1.read(4)
		dense = struct.unpack('f', d1)
		d[i, j] = dense[0]
f1.close()

#-----

# plot array

img = pyplot.imshow(d, interpolation = 'nearest', cmap = 'pink')
pyplot.colorbar(img)
pyplot.show()
