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

echo "Confronto IDX_ALLOC vs ROWING vs PREFETCH"
echo "Start: $(date)"
echo ""

# Parametri
NDIM=2
NCONFS=100
BETA=0.45
SEED=124634

# Compila le tre versioni
echo "Compilazione IDX_ALLOC..."
mpicxx -O3 -std=c++17 -fopenmp -DIDX_ALLOC \
    -Iinclude -Irandom123/include \
    src/main.cpp -o ising_idx.exe

echo "Compilazione ROWING..."
mpicxx -O3 -std=c++17 -fopenmp -DROWING \
    -Iinclude -Irandom123/include \
    src/main.cpp -o ising_rowing.exe

echo "Compilazione PREFETCH..."
mpicxx -O3 -std=c++17 -fopenmp -DROWING -DPREFETCH_CACHE \
    -Iinclude -Irandom123/include \
    src/main.cpp -o ising_prefetch.exe
echo ""

# WEAK SCALING: volume per rank costante 2000x2000 = 4M siti/rank
echo "========== WEAK SCALING =========="
echo "Volume per rank costante: 2000x2000 = 4000000 siti/rank"
echo ""

RANKS_W=(1     2     4     8     16   )
L0_W=(   2000  4000  4000  8000  8000 )
L1_W=(   2000  2000  4000  4000  8000 )

for i in "${!RANKS_W[@]}"; do
    NRANKS=${RANKS_W[$i]}
    L0=${L0_W[$i]}
    L1=${L1_W[$i]}
    NTHREADS=$((32 / NRANKS))

    echo "--- Weak: $NRANKS rank x $NTHREADS threads, ${L0}x${L1} ---"

    for MODE in idx rowing prefetch; do
        echo "  [$MODE]"
        mpirun -n $NRANKS ./ising_${MODE}.exe \
            $NDIM $L0 $L1 $NCONFS $NTHREADS $BETA $SEED \
            2>&1 | tee logs/weak_${MODE}_${NRANKS}rank.log
    done
    echo ""
done

# STRONG SCALING: reticolo fisso 16000x16000
echo "========== STRONG SCALING =========="
echo "Reticolo fisso: 16000x16000"
echo ""

L0=16000
L1=16000

for NRANKS in 1 2 4 8 16; do
    NTHREADS=$((32 / NRANKS))

    echo "--- Strong: $NRANKS rank x $NTHREADS threads ---"

    for MODE in idx rowing prefetch; do
        echo "  [$MODE]"
        mpirun -n $NRANKS ./ising_${MODE}.exe \
            $NDIM $L0 $L1 $NCONFS $NTHREADS $BETA $SEED \
            2>&1 | tee logs/strong_${MODE}_${NRANKS}rank.log
    done
    echo ""
done

echo "Confronto completato"
echo "Fine: $(date)"
