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
#ARCH=BGQ
#ARCH=FUSION
#ARCH=XT
#ARCH=XE

# number of procs
num_procs=8

# procs per node
ppn=4 # adjustable for BG/Q, allowed 1, 2, 4, 8, 16, 32, 64

# number of BG/Q nodes
num_nodes=$[$num_procs / $ppn]
if [ $num_nodes -lt 1 ]; then
    num_nodes=1
fi

# executable
exe=./hacc-test

# inout file
infile="/Users/tpeterka/datasets/hacc/m000.mpicosmo.499"
#infile="/projects/DarkUniverse_esp/heitmann/OuterRim/M000/L355/HACC000/output/STEP499/m000.full.mpicosmo.499"
#infile="/projects/SSSPPg/tpeterka/STEP68/m003.full.mpicosmo.68"

# output file
outfile="del.out"
#outfile="/projects/SSSPPg/tpeterka/hacc-vor.out"
#outfile="!"

# sample rate (1 = keep every particle, 10 = keep every 10th particle)
sr=1

# options: kd-tree
# possibilities are --wrap --kdtree --blocks <totblocks> --debug --chunk <chunk size>
# opts="--blocks 64"
opts="--kdtree --blocks 64"

#------
#
# program arguments
#

args="$opts $infile $outfile $sr"

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
