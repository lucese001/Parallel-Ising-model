#!/bin/bash
#PBS -N tc_2D_32_hot
#PBS -l nodes=1:ppn=32
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o /dev/null

cd $PBS_O_WORKDIR

# === SEED: passa con  qsub -v SEED=<valore>  oppure modifica qui ===
SEED=${SEED:-124634}

LOGFILE="output/tc_2D_32_hot_s${SEED}_${PBS_JOBID}.log"
exec > "$LOGFILE" 2>&1

# Setup MPI
unset LD_LIBRARY_PATH
export MPI_ROOT=/storage/local/exp_soft/local_sl7/mpi/openmpi-4.0.5
export PATH=$MPI_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "SEED=$SEED"
echo "Start: $(date)"
echo ""

NDIM=2
NRANKS=4
NTHREADS=8
L0=32
L1=32
OUTDIR="output/${L0}x${L1}"
mkdir -p "$OUTDIR"

T=(      2.249  2.254  2.259  2.264  2.269  2.274  2.279  2.284  2.289  )
NCONFS=( 500000 500000 500000 500000 500000 500000 500000 500000 500000  )

if [ ! -f ising_rowing.exe ]; then
    mpicxx -O3 -std=c++17 -fopenmp -DROWING \
        -Iinclude -Irandom123/include \
        src/main.cpp -o ising_rowing.exe
fi

for i in "${!T[@]}"; do
    BETA=$(awk "BEGIN {printf \"%.10f\", 1.0/${T[$i]}}")
    echo "=== T=${T[$i]}  BETA=$BETA  seed=$SEED ==="
    mpirun -n $NRANKS ./ising_rowing.exe \
        $NDIM $L0 $L1 ${NCONFS[$i]} $NTHREADS $BETA $SEED
    mv "${OUTDIR}/meas_T${T[$i]}_hot.txt" "${OUTDIR}/meas_T${T[$i]}_hot_s${SEED}.txt"
    echo "  -> salvato come meas_T${T[$i]}_hot_s${SEED}.txt"
    echo ""
done

echo "Scan Tc 32x32 hot completato"
echo "Fine: $(date)"
