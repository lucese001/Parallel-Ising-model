#!/bin/bash
#PBS -N scan_2D_128_hot
#PBS -l nodes=1:ppn=32
#PBS -l walltime=80:00:00
#PBS -j oe
#PBS -o /dev/null

cd $PBS_O_WORKDIR

# === SEED: passa con  qsub -v SEED=<valore>  oppure modifica qui ===
SEED=${SEED:-124634}

LOGFILE="output/scan_2D_48_hot_s${SEED}_${PBS_JOBID}.log"
exec > "$LOGFILE" 2>&1

# Setup MPI
unset LD_LIBRARY_PATH
export MPI_ROOT=/storage/local/exp_soft/local_sl7/mpi/openmpi-4.0.5
export PATH=$MPI_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Full scan 2D — 128x128 — HOT START  (full scan + scan fine vicino a Tc)"
echo "SEED=$SEED"
echo "Start: $(date)"
echo ""

NDIM=2
NRANKS=4
NTHREADS=8
L0=128
L1=128
OUTDIR="output/${L0}x${L1}"
mkdir -p "$OUTDIR"

# Ricompila sempre
mpicxx -O3 -std=c++17 -fopenmp -DROWING \
    -Iinclude -Irandom123/include \
    src/main.cpp -o ising_rowing.exe

# ---------------------------------------------------------------
# PARTE 1 — full scan  passo 0.05
# ---------------------------------------------------------------
echo "========================================"
echo "PARTE 1 — full scan  passo 0.05"
echo "========================================"

T_FULL=(      1.90    1.95    2.00    2.05    2.10    2.15    2.20       2.30    2.35    2.40    2.45    2.50    )
NCONFS_FULL=( 300000  300000  300000  800000  2000000 3000000 4000000   4000000 3000000 2000000 800000  300000  )

for i in "${!T_FULL[@]}"; do
    BETA=$(awk "BEGIN {printf \"%.10f\", 1.0/${T_FULL[$i]}}")
    T_STR=$(awk "BEGIN {printf \"%.6g\",  ${T_FULL[$i]}}")
    echo "=== T=${T_FULL[$i]}  BETA=$BETA  [$(($i+1))/${#T_FULL[@]}]  seed=$SEED  $(date +%H:%M:%S) ==="
    mpirun -n $NRANKS ./ising_rowing.exe \
        $NDIM $L0 $L1 ${NCONFS_FULL[$i]} $NTHREADS $BETA $SEED
    mv "${OUTDIR}/meas_T${T_STR}_hot.txt" "${OUTDIR}/meas_T${T_STR}_hot_s${SEED}.txt"
    echo "  -> salvato come meas_T${T_STR}_hot_s${SEED}.txt"
    echo ""
done

# ---------------------------------------------------------------
# PARTE 2 — scan fine  passo 0.005  intorno a Tc_2D ~ 2.269
# ---------------------------------------------------------------
echo "========================================"
echo "PARTE 2 — scan fine  passo 0.005  (Tc_2D ~ 2.269)"
echo "========================================"

T_FINE=(      2.249   2.254   2.259   2.264   2.269   2.274   2.279   2.284   2.289   )
NCONFS_FINE=( 5000000 5000000 5000000 5000000 5000000 5000000 5000000 5000000 5000000  )

for i in "${!T_FINE[@]}"; do
    BETA=$(awk "BEGIN {printf \"%.10f\", 1.0/${T_FINE[$i]}}")
    echo "=== T=${T_FINE[$i]}  BETA=$BETA  [$(($i+1))/${#T_FINE[@]}]  seed=$SEED  $(date +%H:%M:%S) ==="
    mpirun -n $NRANKS ./ising_rowing.exe \
        $NDIM $L0 $L1 ${NCONFS_FINE[$i]} $NTHREADS $BETA $SEED
    mv "${OUTDIR}/meas_T${T_FINE[$i]}_hot.txt" "${OUTDIR}/meas_T${T_FINE[$i]}_hot_s${SEED}.txt"
    echo "  -> salvato come meas_T${T_FINE[$i]}_hot_s${SEED}.txt"
    echo ""
done

echo "Full scan 128x128 hot completato"
echo "Fine: $(date)"
