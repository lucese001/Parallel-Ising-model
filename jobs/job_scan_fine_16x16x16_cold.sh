#!/bin/bash
#PBS -N scan_fine_16_cold
#PBS -l nodes=1:ppn=32
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o /dev/null

cd $PBS_O_WORKDIR

# Seed: passa con  qsub -v SEED=<valore>  oppure usa il default
SEED=${SEED:-124634}

LOGFILE="output/scan_fine_16x16x16_cold_s${SEED}_${PBS_JOBID}.log"
exec > "$LOGFILE" 2>&1

# Setup MPI
unset LD_LIBRARY_PATH
export MPI_ROOT=/storage/local/exp_soft/local_sl7/mpi/openmpi-4.0.5
export PATH=$MPI_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "3D 16x16x16 — COLD START  (scan fine vicino a Tc)"
echo "SEED=$SEED"
echo "Start: $(date)"
echo ""

NDIM=3
NRANKS=4
NTHREADS=8
L0=16
L1=16
L2=16
OUTDIR="output/${L0}x${L1}x${L2}"
mkdir -p "$OUTDIR"

if [ ! -f ising_rowing.exe ]; then
    mpicxx -O3 -std=c++17 -fopenmp -DROWING -DPREFETCH_CACHE \
        -Iinclude -Irandom123/include \
        src/main.cpp -o ising_rowing.exe
fi

echo "========================================"
echo "Scan fine  passo 0.005  (Tc_3D ~ 4.511)"
echo "========================================"

T=(      4.491   4.496   4.501   4.506   4.511   4.516   4.521   4.526   4.531   )
NCONFS=( 1000000 1000000 1000000 1000000 1000000 1000000 1000000 1000000 1000000  )

for i in "${!T[@]}"; do
    BETA=$(awk "BEGIN {printf \"%.10f\", 1.0/${T[$i]}}")
    echo "=== T=${T[$i]}  BETA=$BETA  [$(($i+1))/${#T[@]}]  $(date +%H:%M:%S) ==="
    mpirun -n $NRANKS --mca btl vader,self ./ising_rowing.exe \
        $NDIM $L0 $L1 $L2 ${NCONFS[$i]} $NTHREADS $BETA $SEED -cold
    mv "${OUTDIR}/meas_T${T[$i]}_cold.txt" "${OUTDIR}/meas_T${T[$i]}_cold_s${SEED}.txt"
    echo "  -> salvato come meas_T${T[$i]}_cold_s${SEED}.txt"
    echo ""
done

echo "scan fine 16x16x16 cold completato"
echo "Fine: $(date)"
