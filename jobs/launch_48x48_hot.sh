#!/bin/bash
# Lancia 8 job hot 48x48 in parallelo con seed diversi

SEEDS=(124634 12080 721683 7281973 919273 741852)

for SEED in "${SEEDS[@]}"; do
    JOBID=$(qsub -v SEED=$SEED jobs/job_48x48_hot.sh)
    echo "Lanciato job hot s${SEED} → $JOBID"
done
