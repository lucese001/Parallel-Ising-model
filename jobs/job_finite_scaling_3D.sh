#!/bin/bash
#PBS -N finite_scaling_3D_hot
#PBS -l nodes=1:ppn=32
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /dev/null

cd $PBS_O_WORKDIR

LOGFILE="output/finite_scaling_3D_hot_${PBS_JOBID}.log"
exec > "$LOGFILE" 2>&1

# Setup MPI
unset LD_LIBRARY_PATH
export MPI_ROOT=/storage/local/exp_soft/local_sl7/mpi/openmpi-4.0.5
export PATH=$MPI_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Finite-size scaling 3D — HOT START"
echo "Start: $(date)"
echo ""

SEED=124634

NDIM=3
NRANKS=4
NTHREADS=8
L0=64
L1=64
L2=64
OUTDIR="output/${L0}x${L1}x${L2}"

T=(      4.3    4.35    4.4     4.45    4.5     4.55    4.6     4.65    4.7    4.75   4.8    )
NCONFS=( 200000 400000 800000 1600000 3200000 1600000 800000 800000 400000 200000 200000 )

# Compila
mpicxx -O3 -std=c++17 -fopenmp -DROWING -DPREFETCH_CACHE \
    -Iinclude -Irandom123/include \
    src/main.cpp -o ising_rowing.exe

for i in "${!T[@]}"; do

    BETA=$(awk "BEGIN {printf \"%.10f\", 1.0/${T[$i]}}")

    echo "=== T=${T[$i]}  BETA=$BETA  seed=$SEED ==="
    mpirun -n $NRANKS ./ising_rowing.exe \
        $NDIM $L0 $L1 $L2 ${NCONFS[$i]} $NTHREADS $BETA $SEED
    mv "${OUTDIR}/meas_T${T[$i]}_hot.txt" "${OUTDIR}/meas_T${T[$i]}_hot_s${SEED}.txt"
    echo ""

done

echo "Finite size scaling 3D hot completato"
echo "Fine: $(date)"
