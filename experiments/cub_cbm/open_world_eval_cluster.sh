#!/bin/bash
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N cbmow_eval
#$ -o logs/cbm_open_world_eval
#$ -q all.q

# CPU evaluation after BOTH detector array jobs have completed successfully.
# Submit with scheduler dependencies, for example:
#   qsub -hold_jid <cub_job_id>,<awa2_job_id> \
#       experiments/cub_cbm/open_world_eval_cluster.sh

micromamba activate datasci

CUB_ART=results/cub_cbm_artifacts/cub_koh112_full
AWA2_ART=results/awa2_cbm_artifacts/awa2
OUT=results/cbm_open_world
SEEDS="0 1 2"

export PYTHONPATH=.
mkdir -p "$OUT"

for artifact in "$CUB_ART" "$AWA2_ART"; do
    for fold in 0 1 2 3 4; do
        for seed in $SEEDS; do
            prefix="openworld_fold${fold}_seed${seed}"
            if [ ! -f "$artifact/${prefix}_provenance.json" ] || \
               [ ! -f "$artifact/${prefix}_test.npz" ]; then
                echo "ERROR: incomplete open-world artifact $artifact/$prefix"
                exit 1
            fi
        done
    done
done

python3 experiments/cub_cbm/probe_open_world_cbm.py \
    --artifact-dir "$CUB_ART" --dataset cub --subset full \
    --n-folds 5 --detector-seeds $SEEDS --learned-depth 60 \
    --output-dir "$OUT"

python3 experiments/cub_cbm/probe_open_world_cbm.py \
    --artifact-dir "$AWA2_ART" --dataset awa2 --subset awa2 \
    --n-folds 5 --detector-seeds $SEEDS --learned-depth 60 \
    --output-dir "$OUT"

echo CBM_OPEN_WORLD_EVAL_DONE
