#!/bin/bash
#PBS -N ising_weak
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
NCONFS=100
SEED=124634
RANKS=4
L0S= 15000
L1S= 15000
T= (0.40, 0.405, 0.41, 0.415, 0.42, 0.425, 0.43, 0.435, 0.44, 0.445,
    0.45, 0.455, 0.46)
STEP=T_F/T_IN

# Compila
mpicxx -O3 -std=c++17 -fopenmp -DROWING \
    -Iinclude -Irandom123/include \
    src/main.cpp -o ising_rowing.exe

for i in "${!T[@]}"; do

    BETA=1/T
    echo "===  ==="
    mpirun -n $NRANKS ./ising_rowing.exe \
        $NDIM $L0 $L1 $NCONFS $NTHREADS $BETA $SEED \
        2>&1 | tee FINITE SIZE-SCALING ${NRANKS}rank_${L0}x${L1}.log
        $T={T}
    echo ""
done

echo "Weak Scaling Completato"
echo "Fine: $(date)"