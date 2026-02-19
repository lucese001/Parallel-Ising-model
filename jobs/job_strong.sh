#!/bin/bash
#PBS -N ising_strong
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

echo "Strong Scaling Test"
echo "Start: $(date)"
echo ""

# Parametri fissi
NDIM=2
L0=16000
L1=16000
NCONFS=100
BETA=0.45
SEED=124634

# Divisori di 32 (ranks x threads = 32)
RANKS=(1 2 4 6 8 10 12 14 16)

# Compila
mpicxx -O3 -std=c++17 -fopenmp -DROWING \
    -Iinclude -Irandom123/include \
    src/main.cpp -o ising_rowing.exe

echo "Reticolo fisso ${L0}x${L1}, $NCONFS configurazioni"
echo ""

for NRANKS in "${RANKS[@]}"; do
    NTHREADS=$((32 / NRANKS))

    echo "=== $NRANKS rank x $NTHREADS threads ==="
    # CLI: <N_dim> <L0> <L1> <nConfs> <nThreads> <Beta> <seed>
    mpirun -n $NRANKS ./ising_rowing.exe \
        $NDIM $L0 $L1 $NCONFS $NTHREADS $BETA $SEED \
        2>&1 | tee logs/strong_${NRANKS}rank.log
    echo ""
done

echo "Strong Scaling Completato"
echo "Fine: $(date)"
