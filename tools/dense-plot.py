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
import argparse
import struct
import matplotlib

#-----

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--in", dest="infile", help="raw (binary) density input file")
parser.add_argument("-n", "--numpts", dest="numpts",
                    help="number of grid point position in one axis (same for x and y)")
parser.add_argument("-o", "--out", dest="out", type=str,
                    choices=["pdf", "svg", "png"], default="png",
                    help="Output format: pdf, svg, none or png also runs interactive display.")
parser.add_argument("-c", "--color", dest="color", type=str,
                    choices=["lin", "log"], default="lin",
                    help="Color map: lin (linear, default) or log (logarithmic)")
parser.add_argument("-min", "--min", dest="min", type=float,
                    default=-1.0, help="Color map minimum value (optional)")
parser.add_argument("-max", "--max", dest="max", type=float,
                    default=-1.0, help="Color map maximum value (optional)")
args = parser.parse_args()

# output format (default = png + interactive display)
if args.out == 'svg':
    print "Output will be in myfig.svg"
    matplotlib.use('svg')
elif args.out == 'pdf':
    print "Output will be in myfig.pdf"
    matplotlib.use('pdf')
else:
    print "Output will be in myfig.png as well as in interactive disply"
import matplotlib.pyplot as plt

#-----

# construct raw density x, y, d arrays

# using np.fromfile() does not work correctly on my mac, so I need to read
# the file one value at a time and append to the array

f1 = open(args.infile, "rb")
s = int(args.numpts) # size in one dimension
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

# get extents
dmin = np.amin(d)
dmax = np.amax(d)
print 'data min =', dmin, 'max =', dmax
if args.min >= 0.0:
        dmin = args.min
        print 'setting min for colormap to', dmin
if args.max >= 0.0:
        dmax = args.max
        print 'setting max for colormap to', dmax

# plot it
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 24}
matplotlib.rc('font', **font)
if args.color == 'log': # log scale
    if dmin <= 0.0:
        img = plt.imshow(d, interpolation = 'nearest', cmap = 'Greys',
                         norm=matplotlib.colors.LogNorm(), vmax=dmax)
    else:
        img = plt.imshow(d, interpolation = 'nearest', cmap = 'Greys',
                         norm=matplotlib.colors.LogNorm(), vmin=dmin, vmax=dmax)
else: # or linear scale
        img = plt.imshow(d, interpolation = 'nearest', cmap = 'pink', vmin=dmin, vmax=dmax)

ax = plt.gca()                               # current graph axes
ax.set_xticks([])                            # no x ticks or labels
ax.set_yticks([])                            # no y ticks or labels
plt.colorbar(img)
plt.savefig('myfig', bbox_inches='tight')
plt.show() # only runs interactive display when ouput mode is default=png
