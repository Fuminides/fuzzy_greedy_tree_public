#!/bin/bash
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N cbmow_cub
#$ -o logs/cbm_open_world_cub
#$ -q gpu.q
#$ -l gpu=1
#$ -t 1-15

# Detector-class-disjoint CUB artifacts for E16.  The 15 array tasks are
# 5 held-out-species folds x 3 detector seeds.  Submit from the FERL root:
#   mkdir -p logs/cbm_open_world_cub
#   qsub experiments/cub_cbm/open_world_cub_gpu.sh

source /usr/local/gpuallocation.sh
micromamba activate gpuenv

CUB=~/Datasets/CUB_200_2011
SS=~/Github/symbol_sanity
ART=results/cub_cbm_artifacts/cub_koh112_full
MANIFEST=results/cub_cbm_artifacts/full_manifest
SEEDS=(0 1 2)

export PYTHONPATH=.

if [ ! -d "$CUB/images" ]; then
    echo "ERROR: CUB image tree not found at $CUB"
    exit 1
fi
if [ ! -d "$SS/src/symbol_sanity" ]; then
    echo "ERROR: symbol_sanity not found at $SS"
    exit 1
fi
if [ ! -f "$MANIFEST/train/metadata.jsonl" ]; then
    echo "ERROR: full CUB manifest missing at $MANIFEST"
    exit 1
fi
if [ ! -f "$ART/oracle_train.npz" ]; then
    echo "ERROR: CUB oracle artifact missing at $ART"
    exit 1
fi

TASK=${SGE_TASK_ID:-1}
INDEX=$((TASK - 1))
FOLD=$((INDEX / 3))
SEED_INDEX=$((INDEX % 3))
SEED=${SEEDS[$SEED_INDEX]}

echo "CUB open-world detector fold=$FOLD seed=$SEED task=$TASK"
python3 experiments/cub_cbm/train_cub_open_world_detector.py \
    --cub-root "$CUB" \
    --manifest-dir "$MANIFEST" \
    --artifact-dir "$ART" \
    --symbol-sanity-root "$SS" \
    --n-folds 5 --folds "$FOLD" --seeds "$SEED" \
    --epochs 15 --batch-size 16 --image-size 299 --device cuda

echo "CUB_OPEN_WORLD_DETECTOR_DONE fold=$FOLD seed=$SEED"
