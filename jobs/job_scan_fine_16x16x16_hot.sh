#!/bin/bash
#PBS -N scan_fine_16_hot
#PBS -l nodes=1:ppn=32
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o /dev/null

cd $PBS_O_WORKDIR

# Seed obbligatorio: qsub -v SEED=<valore> job_scan_fine_16x16x16_hot.sh
if [ -z "$SEED" ]; then
    echo "ERRORE: SEED non impostato."
    echo "Uso: qsub -v SEED=<valore> $(basename $0)"
    exit 1
fi

LOGFILE="output/scan_fine_16x16x16_hot_s${SEED}_${PBS_JOBID}.log"
exec > "$LOGFILE" 2>&1

# Setup MPI
unset LD_LIBRARY_PATH
export MPI_ROOT=/storage/local/exp_soft/local_sl7/mpi/openmpi-4.0.5
export PATH=$MPI_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "3D 16x16x16 — HOT START  (scan fine vicino a Tc)"
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
        $NDIM $L0 $L1 $L2 ${NCONFS[$i]} $NTHREADS $BETA $SEED
    mv "${OUTDIR}/meas_T${T[$i]}_hot.txt" "${OUTDIR}/meas_T${T[$i]}_hot_s${SEED}.txt"
    echo "  -> salvato come meas_T${T[$i]}_hot_s${SEED}.txt"
    echo ""
done

echo "scan fine 16x16x16 hot completato"
echo "Fine: $(date)"
