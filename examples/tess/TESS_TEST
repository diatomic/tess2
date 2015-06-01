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
#----------------------------------------------------------------------------
ARCH=MAC_OSX
#ARCH=LINUX
#ARCH=BGQ
#ARCH=FUSION
#ARCH=XT
#ARCH=XE

# number of procs
num_procs=4

# procs per node
ppn=8 # adjustable for BG/Q, allowed 1, 2, 4, 8, 16, 32, 64

# number of BG/P nodes for vn mode
num_nodes=$[$num_procs / $ppn]
if [ $num_nodes -lt 1 ]; then
    num_nodes=1
fi

# executable
exe=./delaunay

# total number of blocks in the domain
tb=$[$num_procs * 1]
#tb=8

# max number of blocks in memory (-1 for no limit)
mb=-1

# data size x y z (always 3D)
#dsize="3 2 2"
#dsize="3 3 3"
#dsize="4 4 4"
#dsize="5 5 5"
#dsize="6 6 6"
#dsize="7 7 7"
#dsize="8 8 8"
#dsize="9 9 9"
#dsize="10 10 10"
dsize="11 11 11"
#dsize="13 13 13"
#dsize="16 16 16"
#dsize="32 32 32"
#dsize="33 33 33"
#dsize="64 64 64"
#dsize="128 128 128"
#dsize="129 129 129"
#dsize="256 256 256"

jitter=2.0

# volume range (-1.0: unused)
minv=-1.0
maxv=-1.0

# wrapped neighbors 0 or 1
wrap=0

# walls 0 or 1 (wrap should be 0)
walls=0

# output file name, "!" to disable output
#outfile="vor.out"
outfile="del.out"
#outfile="!"

#------
#
# program arguments
#
args="$tb $mb $dsize $jitter $minv $maxv $wrap $walls $outfile"

#------
#
# run commands
#

if [ "$ARCH" = "MAC_OSX" ]; then

mpiexec -l -n $num_procs $exe $args

#dsymutil $exe ; mpiexec -l -n $num_procs xterm -e gdb -x gdb.run --args $exe $args

#dsymutil $exe ; mpiexec -l -n $num_procs valgrind -q $exe $args

#dsymutil $exe ; mpiexec -l -n $num_procs valgrind -q --tool=massif $exe $args

#dsymutil $exe ; mpiexec -n $num_procs valgrind -q --leak-check=yes $exe $args

fi

if [ "$ARCH" = "LINUX" ]; then

#mpiexec -n $num_procs $exe $args

#mpiexec -n $num_procs xterm -e gdb -x gdb.run --args $exe $args

#mpiexec -n $num_procs valgrind -q $exe $args

mpiexec -n $num_procs valgrind -q --leak-check=yes $exe $args

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
