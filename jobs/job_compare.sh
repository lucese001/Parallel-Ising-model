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

echo "Weak Scaling Test"
echo "Volume per rank costante: 1500x1500 = 2250000 siti/rank"
echo "Start: $(date)"
echo ""

# Parametri fissi
NDIM=2
LOCAL_L=1500       # lato locale per rank
NCONFS=100
BETA=0.45
SEED=124634

# Array paralleli: ranks, L0, L1
RANKS=(1    2    3    4    5    6    7     8)
L0S=(  1500 3000 3000 6000 6000 12000)
L1S=(  1500 1500 3000 3000 6000 6000)

# Compila
mpicxx -O3 -std=c++17 -fopenmp -DROWING \
    -Iinclude -Irandom123/include \
    src/main.cpp -o ising_rowing.exe

for i in "${!RANKS[@]}"; do
    NRANKS=${RANKS[$i]}
    L0=${L0S[$i]}
    L1=${L1S[$i]}
    NTHREADS=$((32 / NRANKS))

    echo "=== $NRANKS rank x $NTHREADS threads, reticolo ${L0}x${L1} ==="
    echo "    Siti totali: $((L0 * L1)), per rank: $((L0 * L1 / NRANKS))"
    mpirun -n $NRANKS ./ising_rowing.exe \
        $NDIM $L0 $L1 $NCONFS $NTHREADS $BETA $SEED \
        2>&1 | tee logs/weak_${NRANKS}rank_${L0}x${L1}.log
    echo ""
done

echo "Weak Scaling Completato"
echo "Fine: $(date)"
