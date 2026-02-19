#!/bin/bash
#PBS -N finite_scaling
#PBS -l nodes=1:ppn=32
#PBS -l walltime=02:00:00
#PBS -j oe

cd $PBS_O_WORKDIR
mkdir -p logs output

# Setup MPI
unset LD_LIBRARY_PATH
export MPI_ROOT=/storage/local/exp_soft/local_sl7/mpi/openmpi-4.0.5
export PATH=$MPI_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Finite-size scaling"
echo "Start: $(date)"
echo ""

# Parametri fissi
NDIM=2
NCONFS=100000
SEED=124634
NRANKS=4
NTHREADS=8
L0=64
L1=64
T=(0.40 0.405 0.41 0.415 0.42 0.425 0.43 0.435 0.44 0.445 0.45 0.455 0.46)

# Compila
mpicxx -O3 -std=c++17 -fopenmp -DROWING \
    -Iinclude -Irandom123/include \
    src/main.cpp -o ising_rowing.exe

for i in "${!T[@]}"; do

    BETA=$(awk "BEGIN {printf \"%.10f\", 1.0/${T[$i]}}")
    echo "=== T=${T[$i]}  BETA=$BETA ==="
    mpirun -n $NRANKS ./ising_rowing.exe \
        $NDIM $L0 $L1 $NCONFS $NTHREADS $BETA $SEED \
        2>&1 | tee "logs/finite_scaling_${NRANKS}rank_${L0}x${L1}_T${T[$i]}.log"
    echo ""
done

echo "Finite size scaling completato"
echo "Fine: $(date)"
