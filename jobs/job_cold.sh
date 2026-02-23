#!/bin/bash
#PBS -N finite_scaling_2D_cold
#PBS -l nodes=1:ppn=32
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o /dev/null

cd $PBS_O_WORKDIR

LOGFILE="output/finite_scaling_2D_cold_${PBS_JOBID}.log"
exec > "$LOGFILE" 2>&1

# Setup MPI
unset LD_LIBRARY_PATH
export MPI_ROOT=/storage/local/exp_soft/local_sl7/mpi/openmpi-4.0.5
export PATH=$MPI_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Finite-size scaling 2D — COLD START"
echo "Start: $(date)"
echo ""

# Il cold va lanciato UNA SOLA VOLTA (non dipende dal seed).
# Conf fisse: abbastanza per termalizzare a tutte le T + statistiche di base.
SEED=124634

NDIM=2
NRANKS=4
NTHREADS=8
L0=64
L1=64
OUTDIR="output/${L0}x${L1}"

T=( 1.90 1.95 2.00 2.05 2.10 2.15 2.20 2.25 2.30 2.35 2.40 2.45 2.50 )

# Confs fisse: piu' vicino a Tc piu' ne servono per termalizzare da cold
NCONFS=( 200000 200000 200000 200000 300000 500000 500000 500000 300000 200000 200000 200000 200000 )

# Compila (solo se non gia' compilato dal job hot)
if [ ! -f ising_rowing.exe ]; then
    mpicxx -O3 -std=c++17 -fopenmp -DROWING \
        -Iinclude -Irandom123/include \
        src/main.cpp -o ising_rowing.exe
fi

for i in "${!T[@]}"; do

    BETA=$(awk "BEGIN {printf \"%.10f\", 1.0/${T[$i]}}")

    echo "=== T=${T[$i]}  BETA=$BETA  (cold) ==="
    mpirun -n $NRANKS ./ising_rowing.exe \
        $NDIM $L0 $L1 ${NCONFS[$i]} $NTHREADS $BETA $SEED -cold
    echo "  -> salvato come meas_T${T[$i]}_cold.txt"
    echo ""

done

echo "Finite size scaling 2D cold completato"
echo "Fine: $(date)"
