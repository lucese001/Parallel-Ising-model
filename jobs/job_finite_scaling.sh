#!/bin/bash
#PBS -N finite_scaling
#PBS -l nodes=1:ppn=32
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o /dev/null

cd $PBS_O_WORKDIR

LOGFILE="output/finite_scaling_${PBS_JOBID}.log"
exec > "$LOGFILE" 2>&1

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
# Temperature e configurazioni corrispondenti
T=(     1.90    1.95   2      2.05    2.10   2.15    2.20    2.25    2.30    2.35    2.40    2.45   2.50)
NCONFS=( 100000 100000 100000 100000  100000 1000000 2000000 2000000 1000000 1000000 1000000 1000000 100000)

# Compila
mpicxx -O3 -std=c++17 -fopenmp -DROWING \
    -Iinclude -Irandom123/include \
    src/main.cpp -o ising_rowing.exe

for i in "${!T[@]}"; do

    BETA=$(awk "BEGIN {printf \"%.10f\", 1.0/${T[$i]}}")

    echo "=== T=${T[$i]}  BETA=$BETA  (hot) ==="
    mpirun -n $NRANKS ./ising_rowing.exe \
        $NDIM $L0 $L1 ${NCONFS[$i]} $NTHREADS $BETA $SEED
    echo ""

    echo "=== T=${T[$i]}  BETA=$BETA  (cold) ==="
    mpirun -n $NRANKS ./ising_rowing.exe \
        $NDIM $L0 $L1 ${NCONFS[$i]} $NTHREADS $BETA $SEED -cold
    echo ""
done

echo "Finite size scaling completato"
echo "Fine: $(date)"
