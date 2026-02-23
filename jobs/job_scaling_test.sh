#!/bin/bash
#PBS -N scaling_test_64
#PBS -l nodes=1:ppn=32
#PBS -l walltime=00:30:00
#PBS -j oe

cd $PBS_O_WORKDIR

unset LD_LIBRARY_PATH
export MPI_ROOT=/storage/local/exp_soft/local_sl7/mpi/openmpi-4.0.5
export PATH=$MPI_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib
export OMP_PROC_BIND=close
export OMP_PLACES=cores

mpicxx -O3 -std=c++17 -fopenmp -DROWING -DPREFETCH_CACHE \
    -Iinclude -Irandom123/include src/main.cpp -o ising_rowing.exe

mkdir -p output/64x64x64

NDIM=3; L=64; NCONFS=5000; BETA=0.2217; SEED=124634

echo "=== 1 rank x 32 thread ==="
mpirun -n 1 ./ising_rowing.exe $NDIM $L $L $L $NCONFS 32 $BETA $SEED
echo ""

echo "=== 2 rank x 16 thread ==="
mpirun -n 2 ./ising_rowing.exe $NDIM $L $L $L $NCONFS 16 $BETA $SEED
echo ""

echo "=== 4 rank x 8 thread ==="
mpirun -n 4 ./ising_rowing.exe $NDIM $L $L $L $NCONFS  8 $BETA $SEED
echo ""

echo "Fine: $(date)"
