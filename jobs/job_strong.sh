#!/bin/bash
#PBS -N ising_strong
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

echo "Strong Scaling Test"
echo "Reticolo fisso 10000x10000, 100 configurazioni"
echo "Start: $(date)"
echo ""

# Parametri fissi
LATTICE="10000 10000"
NCONFS=100
BETA=0.45
SEED=124634
RANKS=(1 2 4 8 16 32 64)

for NRANKS in "${RANKS[@]}"; do
    NTHREADS=$((64 / NRANKS))

    cat > input/dimensioni.txt << EOF
2
$LATTICE
$NCONFS
$NTHREADS
$BETA
$SEED
EOF

    echo "$NRANKS rank x $NTHREADS threads"
    mpirun -n $NRANKS ./ising_philox.exe 2>&1 | tee logs/strong_${NRANKS}rank.log
    echo ""
done

echo "Strong Scaling Completato"
echo "Fine: $(date)"
echo ""
