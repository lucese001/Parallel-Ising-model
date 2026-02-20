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
L0=16
L1=16
L3=16
T=(4.25 4.3 4.35 4.4 4.45 4.5 4.55 4.6 4.65 4.7 4.75)

# Compila
mpicxx -O3 -std=c++17 -fopenmp -DROWING \
    -Iinclude -Irandom123/include \
    src/main.cpp -o ising_rowing.exe

for i in "${!T[@]}"; do

    BETA=$(awk "BEGIN {printf \"%.10f\", 1.0/${T[$i]}}")
    echo "=== T=${T[$i]}  BETA=$BETA ==="
    mpirun -n $NRANKS ./ising_rowing.exe \
        $NDIM $L0 $L1 $L2 $NCONFS $NTHREADS $BETA $SEED \
        2>&1 | tee "logs/finite_scaling_${NRANKS}rank_${L0}x${L1}_T${T[$i]}.log"
    echo ""
done

echo "Finite size scaling completato"
echo "Fine: $(date)"
