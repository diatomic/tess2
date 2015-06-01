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
num_procs=8

# procs per node
ppn=8 # adjustable for BG/Q, allowed 1, 2, 4, 8, 16, 32, 64

# number of BG/P nodes for vn mode
num_nodes=$[$num_procs / $ppn]
if [ $num_nodes -lt 1 ]; then
    num_nodes=1
fi

# executable
exe=./tess-dense

# total number of blocks in the domain
tb=$[$num_procs * 1]
#tb=4

# algorithm (0=tess, 1 = cic)
alg=0

# data size x y z (always 3D)
#dsize="3 2 2"
#dsize="3 3 3"
#dsize="4 4 4"
#dsize="5 5 5"
#dsize="6 6 6"
#dsize="8 8 8"
#dsize="10 10 10"
#dsize="16 16 16"
dsize="32 32 32"
#dsize="64 64 64"
#dsize="128 128 128"
#dsize="256 256 256"

jitter=2.0

# volume range (-1.0: unused)
minv=-1.0
maxv=-1.0

# wrapped neighbors 0 or 1
wrap=0

# walls 0 or 1 (wrap should be 0)
walls=0

# output file name
outfile="dense.raw"

# sample grid size (number of points) x y z
#gsize="16 16 16"
#gsize="32 32 32"
#gsize="64 64 64"
#gsize="128 128 128"
#gsize="256 256 256"
gsize="512 512 512"
#gsize="1024 1024 1024"
#gsize="2048 2048 2048"
#gsize="4096 4096 4096"
#gsize="8192 8192 8192"

#projection plane
#project=!
project="0.0 0.0 1.0" #normal to plane, xy plane is the only one supported so far"

# particle mass
mass=1

# given bounds
ng=0
gmin="-1.5 -1.5"
gmax="1.5 1.5"

#------
#
# program arguments
#
args="$alg $tb $dsize $jitter $minv $maxv $wrap $walls $outfile $gsize $project $mass $ng $gmin $gmax"

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
