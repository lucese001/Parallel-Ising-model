#!/bin/bash
#PBS -N ising_1node
#PBS -l nodes=4:ppn=24
#PBS -l walltime=02:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

mkdir -p logs

unset LD_LIBRARY_PATH
export MPI_ROOT=/storage/local/exp_soft/local_sl7/mpi/openmpi-4.0.5
export PATH=$MPI_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib

# Configurazione: 1 nodo, 4 MPI ranks, 16 thread ognuno
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Starting job on $(hostname) at $(date)"
echo "Working directory: $(pwd)"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"

mpirun -n 4 ./ising_philox.exe 2>&1 | tee logs/ising_1node.log
echo "Job finished at $(date)"