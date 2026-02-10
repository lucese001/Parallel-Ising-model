#!/bin/bash
#PBS -N ising_repro
#PBS -l nodes=1:ppn=32
#PBS -l walltime=00:30:00
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

echo "Test riproducibilita "
echo "Start: $(date)"

# Parametri del test
NDIM=2
SIDE=120
NCONFS=1000
BETA=0.45
SEED=124634

# Compilazione
mpicxx -O3 -std=c++17 -fopenmp -DUSE_PHILOX \
    -Iinclude -Irandom123/include \
    src/main.cpp -o ising_philox.exe

grep Cpus_allowed_list /proc/self/status

# Test con 1, 2, 4 rank (threads = 32/ranks)
for NRANKS in 1 2 4; do
    NTHREADS=$((32 / NRANKS))

    echo "--- $NRANKS rank x $NTHREADS threads ---"
    # CLI: <N_dim> <L0> <L1> <nConfs> <nThreads> <Beta> <seed>
    mpirun -n $NRANKS ./ising_philox.exe \
        $NDIM $SIDE $SIDE $NCONFS $NTHREADS $BETA $SEED \
        2>&1 | tee logs/repro_${NRANKS}rank.log
    echo ""
done

# Confronta gli output
echo "Confronto output"
REF="output/meas_1rank_120x120.txt"
PASS=true

for NRANKS in 2 4; do
    FILE="output/meas_${NRANKS}rank_120x120.txt"
    if diff -q "$REF" "$FILE" > /dev/null 2>&1; then
        echo "OK: 1 rank vs $NRANKS rank -> IDENTICI"
    else
        echo "ERRORE: 1 rank vs $NRANKS rank -> DIVERSI!"
        diff "$REF" "$FILE" | head -5
        PASS=false
    fi
done

if $PASS; then
    echo "RIPRODUCIBILITA PASSATA"
else
    echo "RIPRODUCIBILITA FALLITA"
fi

echo "Fine: $(date)"