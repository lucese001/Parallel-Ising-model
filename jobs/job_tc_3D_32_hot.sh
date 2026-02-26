#!/bin/bash
#PBS -N tc_3D_32_hot
#PBS -l nodes=1:ppn=32
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /dev/null

cd $PBS_O_WORKDIR

# === SEED: passa con  qsub -v SEED=<valore>  oppure modifica qui ===
SEED=${SEED:-124634}

LOGFILE="output/tc_3D_32_hot_s${SEED}_${PBS_JOBID}.log"
exec > "$LOGFILE" 2>&1

# Setup MPI
unset LD_LIBRARY_PATH
export MPI_ROOT=/storage/local/exp_soft/local_sl7/mpi/openmpi-4.0.5
export PATH=$MPI_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Scan completo Tc 3D — 32x32x32 — HOT START"
echo "SEED=$SEED"
echo "Start: $(date)"
echo ""

NDIM=3
NRANKS=4
NTHREADS=8
L0=32
L1=32
L2=32
OUTDIR="output/${L0}x${L1}x${L2}"
mkdir -p "$OUTDIR"

T=(       4.25    4.30    4.35    4.40    4.45    4.491   4.496   4.501   4.506   4.511   4.516   4.521   4.526   4.531   4.55    4.60    4.65    4.70    4.75    )
NCONFS=(  200000  200000  300000  300000  500000  1200000 1200000 1200000 1200000 1200000 1200000 1200000 1200000 1200000 500000  300000  300000  200000  200000  )

if [ ! -f ising_rowing.exe ]; then
    mpicxx -O3 -std=c++17 -fopenmp -DROWING -DPREFETCH_CACHE \
        -Iinclude -Irandom123/include \
        src/main.cpp -o ising_rowing.exe
fi

for i in "${!T[@]}"; do
    BETA=$(awk "BEGIN {printf \"%.10f\", 1.0/${T[$i]}}")
    echo "=== T=${T[$i]}  BETA=$BETA  seed=$SEED ==="
    mpirun -n $NRANKS ./ising_rowing.exe \
        $NDIM $L0 $L1 $L2 ${NCONFS[$i]} $NTHREADS $BETA $SEED
    mv "${OUTDIR}/meas_T${T[$i]}_hot.txt" "${OUTDIR}/meas_T${T[$i]}_hot_s${SEED}.txt"
    echo "  -> salvato come meas_T${T[$i]}_hot_s${SEED}.txt"
    echo ""
done

echo "Scan Tc 32x32x32 hot completato"
echo "Fine: $(date)"
