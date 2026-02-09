#!/bin/bash
#PBS -N ising_weak
#PBS -l nodes=1:ppn=64
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
echo "Volume per rank costante: 1875x1875 = 3515625 siti/rank"
echo "Start: $(date)"
echo ""

# 1875 * sqrt(NRANKS) = lato totale
NCONFS=100
BETA=0.45
SEED=124634

# Array paralleli: (ranks, lato totale)
RANKS=(1    4    16   64)
SIDES=(1875 3750 7500 15000)

# Compila
mpicxx -O3 -std=c++17 -fopenmp -DUSE_PHILOX \
    -Iinclude -Irandom123/include \
    src/main.cpp -o ising_philox.exe

for i in "${!RANKS[@]}"; do
    NRANKS=${RANKS[$i]}
    SIDE=${SIDES[$i]}
    NTHREADS=$((64 / NRANKS))

    cat > input/dimensioni.txt << EOF
2
$SIDE $SIDE
$NCONFS
$NTHREADS
$BETA
$SEED
EOF

    echo "=== $NRANKS rank x $NTHREADS threads, reticolo ${SIDE}x${SIDE} ==="
    echo "    Siti totali: $((SIDE * SIDE)), per rank: $((SIDE * SIDE / NRANKS))"
    mpirun -n $NRANKS ./ising_philox.exe 2>&1 | tee logs/weak_${NRANKS}rank_${SIDE}x${SIDE}.log
    echo ""
done

echo " Weak Scaling Completato "
echo "Fine: $(date)"
