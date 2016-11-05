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
# must be <= tb
num_procs=4

# procs per node
ppn=2 # adjustable for BG/Q, allowed 1, 2, 4, 8, 16, 32, 64

# number of BG/P nodes for vn mode
num_nodes=$[$num_procs / $ppn]
if [ $num_nodes -lt 1 ]; then
    num_nodes=1
fi

# executable
exe=./dense

# input file
#infile="/projects/SSSPPg/tpeterka/128p4096b-vor.out.nc"
#infile="../driver/vor.out.nc"
#infile="../tess-driver/del.out.nc"
#infile="vor.out-100K-2b.nc"
#infile="vor-99.out.nc"
#infile="7445077095.out.nc"
#infile="${HOME}/tess/post-tess-driver/6606356352.out.nc"
#infile="${HOME}/tess/post-tess-driver/7445077095.out.nc"
#infile="${HOME}/tess/post-tess-driver/nfw-1e5.out.nc"
#infile="cnfw_2e5.out.nc"
#infile="${HOME}/hacc/voronoi/density-estimator/tests/nfw/voronoi-results/nfw.out.nc"
#infile="${HOME}/hacc/voronoi/density-estimator/tests/cnfw/cic-results/cnfw_2e5.out.nc"
infile="../tess/del.out"

# output file
outfile="dense.raw"

# algorithm (0=tess, 1 = cic)
alg=0

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

args="$infile $outfile $alg $gsize $project $mass $ng $gmin $gmax"

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

qsub -n $num_nodes --mode c$ppn -A SSSPPg -t 60 $exe $args

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
