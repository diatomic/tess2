#!/bin/bash

##PBS -A CSC033
#PBS -A CI-CCR000086
#PBS -N t
#PBS -j oe
#PBS -l walltime=0:10:00,size=12

#----------------------------------------------------------------------------
#
# mpi run script
#
# Tom Peterka
# Argonne National Laboratory
# 9700 S. Cass Ave.
# Argonne, IL 60439
# tpeterka@mcs.anl.gov
#
# All rights reserved. May not be used, modified, or copied
# without permission
#
#----------------------------------------------------------------------------
ARCH=MAC_OSX
#ARCH=LINUX
#ARCH=BGP
#ARCH=BGQ
#ARCH=FUSION
#ARCH=XT
#ARCH=XE

# number of procs
num_procs=1

# procs per node
#ppn=4 # for BG/P
ppn=1 # adjustable for BG/Q, allowed 1, 2, 4, 8, 16, 32, 64

# number of BG/P nodes for vn mode
num_nodes=$[$num_procs / $ppn]
if [ $num_nodes -lt 1 ]; then
    num_nodes=1
fi

# executable
exe=./post-delaunay

# total number of blocks in the domain
tb=$[$num_procs * 1]
#tb=128

# inout file
#infile="${HOME}/hacc/voronoi/density-estimator/tests/halos/input-data/7445077095.dat"
#infile="${HOME}/hacc/voronoi/density-estimator/tests/halos/input-data/6606356352.dat"
infile="${HOME}/hacc/voronoi/density-estimator/tests/nfw/input-data/nfw_particles_1e4.dat"
#infile="${HOME}/hacc/voronoi/density-estimator/tests/cnfw/input-data/cnfw_particles_2e5.dat"
#infile="./pts.out"

# output file, "!" to disable output
#outfile="7445077095.out"
#outfile="6606356352.out"
outfile="nfw.out"
#outfile="mchalo00.out"
#outfile="nor_ex.out"
#outfile="cnfw_2e5.out"
#outfile="pts-vor.out"

#volume range (-1.0: unused)
minv=-1.0
maxv=-1.0

# input file type 
# 0 = text
# 1 = float x's followed by y's followed by z's
# 2 = float interleaved x y z
# 3 = double x's followed by y's followed by z's
# 4 = double interleaved x y z
it=0
#it=3
#it=2

#byte swapping
swap=0

# wrapped neihbors 0 or 1
wrap=0

#------
#
# program arguments
#

args="$tb $infile $outfile $minv $maxv $it $swap $wrap"

#------
#
# run commands
#

if [ "$ARCH" = "MAC_OSX" ]; then

mpiexec -l -n $num_procs $exe $args

#dsymutil $exe ; mpiexec -l -n $num_procs xterm -e gdb -x gdb.run --args $exe $args

#dsymutil $exe ; mpiexec -l -n $num_procs valgrind -q $exe $args

#dsymutil $exe ; mpiexec -n $num_procs valgrind -q --leak-check=yes $exe $args

fi

if [ "$ARCH" = "LINUX" ]; then

mpiexec -n $num_procs $exe $args

#mpiexec -n $num_procs xterm -e gdb -x gdb.run --args $exe $args

#mpiexec -n $num_procs valgrind -q $exe $args

#mpiexec -n $num_procs valgrind -q --leak-check=yes $exe $args

fi

if [ "$ARCH" = "BGP" ]; then

#cqsub -n $num_nodes -c $num_procs -p UltraVis -t 30 -m vn $exe $args
cqsub -n $num_procs -c $num_procs -p UltraVis -t 30 -m smp $exe $args

# for use with valgrind_memcheck.o
#cqsub -n $num_procs -c $num_procs -p UltraVis -t 30 -m smp $exe -- $args

fi

if [ "$ARCH" = "BGQ" ]; then

qsub -n $num_nodes --mode c$ppn -A SDAV -t 60 $exe $args

fi

if [ "$ARCH" = "FUSION" ]; then

mpiexec $exe $args

fi

if [ "$ARCH" = "XT" ]; then

cd /tmp/work/$USER
aprun -n $num_procs $exe $args

fi

if [ "$ARCH" = "XE" ]; then

aprun -n $num_procs $exe $args

fi
