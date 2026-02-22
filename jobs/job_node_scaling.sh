#!/bin/bash
#PBS -N ising_node_scaling
#PBS -l nodes=4:ppn=32
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o /dev/null

cd $PBS_O_WORKDIR
mkdir -p logs output

LOGFILE="output/node_scaling_${PBS_JOBID}.log"
exec > "$LOGFILE" 2>&1

# Setup MPI
unset LD_LIBRARY_PATH
export MPI_ROOT=/storage/local/exp_soft/local_sl7/mpi/openmpi-4.0.5
export PATH=$MPI_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib
export OMP_PROC_BIND=close
export OMP_PLACES=cores
# Disabilita InfiniBand per evitare warning
export OMPI_MCA_btl=tcp,self

echo "Node Scaling Test (1, 2 e 4 nodi)"
echo "Start: $(date)"
echo ""

# Parametri fissi
NCONFS=100
BETA=0.45
SEED=124634
NTHREADS=32
RANKS_PER_NODE=4

# Compila
mpicxx -O3 -std=c++17 -fopenmp -DROWING \
    -Iinclude -Irandom123/include \
    src/main.cpp -o ising_rowing.exe

echo "========================================"
echo "  RETICOLI 2D"
echo "========================================"
echo ""

for L in 64 128 256; do
    echo "--- Reticolo 2D: ${L}x${L} ---"
    echo ""

    for NNODES in 1 2 4; do
        NRANKS=$((NNODES * RANKS_PER_NODE))
        echo "  === $NNODES nodi, $NRANKS ranks x $NTHREADS threads ==="
        mpirun -n $NRANKS --map-by ppr:${RANKS_PER_NODE}:node ./ising_rowing.exe \
            2 $L $L $NCONFS $NTHREADS $BETA $SEED
        echo ""
    done
done

echo "========================================"
echo "  RETICOLI 3D"
echo "========================================"
echo ""

for L in 8 16 32; do
    echo "--- Reticolo 3D: ${L}x${L}x${L} ---"
    echo ""

    for NNODES in 1 2 4; do
        NRANKS=$((NNODES * RANKS_PER_NODE))
        echo "  === $NNODES nodi, $NRANKS ranks x $NTHREADS threads ==="
        mpirun -n $NRANKS --map-by ppr:${RANKS_PER_NODE}:node ./ising_rowing.exe \
            3 $L $L $L $NCONFS $NTHREADS $BETA $SEED
        echo ""
    done
done

echo "Node scaling completato"
echo "Fine: $(date)"
