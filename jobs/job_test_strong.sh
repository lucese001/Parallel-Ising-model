#!/bin/bash
#PBS -N ising_compare
#PBS -l nodes=1:ppn=32
#PBS -l walltime=01:00:00
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
# Disabilita la ricerca di InfiniBand
export OMPI_MCA_btl=tcp,self

echo "Confronto IDX_ALLOC vs ROWING"
echo "Start: $(date)"
echo ""

# Parametri
NDIM=2
NCONFS=100
BETA=0.45
SEED=124634

# Compila entrambe le versioni
echo "Compilazione IDX_ALLOC..."
mpicxx -O3 -std=c++17 -fopenmp -DIDX_ALLOC \
    -Iinclude -Irandom123/include \
    src/main.cpp -o ising_idx.exe

echo "Compilazione ROWING..."
mpicxx -O3 -std=c++17 -fopenmp -DROWING \
    -Iinclude -Irandom123/include \
    src/main.cpp -o ising_rowing.exe
echo ""

# STRONG SCALING: reticolo fisso 8400x8400
echo "========== STRONG SCALING =========="
echo "Reticolo fisso: 16000x16000"
echo ""

L0=16000
L1=16000

for NRANKS in 1 2 4 ; do
    NTHREADS=$((32 / NRANKS))

    echo "--- Strong: $NRANKS rank x $NTHREADS threads ---"

    for MODE in idx rowing; do
        echo "  [$MODE]"
        mpirun -n $NRANKS ./ising_${MODE}.exe \
            $NDIM $L0 $L1 $NCONFS $NTHREADS $BETA $SEED \
            2>&1 | tee logs/strong_${MODE}_${NRANKS}rank.log
    done
    echo ""
done

echo "Confronto completato"
echo "Fine: $(date)"
