#!/bin/bash
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N cbmow_awa2
#$ -o logs/cbm_open_world_awa2
#$ -q gpu.q
#$ -l gpu=1
#$ -t 1-15

# Detector-class-disjoint AwA2 artifacts for E16.  The 15 array tasks are
# 5 held-out-class folds x 3 detector seeds.  Submit from the FERL root:
#   mkdir -p logs/cbm_open_world_awa2
#   qsub experiments/cub_cbm/open_world_awa2_gpu.sh

source /usr/local/gpuallocation.sh
micromamba activate gpuenv

AWA2=~/Datasets/Animals_with_Attributes2
ART=results/awa2_cbm_artifacts/awa2
SEEDS=(0 1 2)

export PYTHONPATH=.

if [ ! -f "$AWA2/predicate-matrix-binary.txt" ] || [ ! -d "$AWA2/JPEGImages" ]; then
    echo "ERROR: AwA2 dataset not found at $AWA2"
    exit 1
fi
if [ ! -f "$ART/oracle_train.npz" ]; then
    echo "ERROR: AwA2 oracle artifact missing at $ART"
    exit 1
fi

TASK=${SGE_TASK_ID:-1}
INDEX=$((TASK - 1))
FOLD=$((INDEX / 3))
SEED_INDEX=$((INDEX % 3))
SEED=${SEEDS[$SEED_INDEX]}

echo "AwA2 open-world detector fold=$FOLD seed=$SEED task=$TASK"
python3 experiments/cub_cbm/train_awa2_detector.py \
    --awa2-root "$AWA2" --artifact-dir "$ART" \
    --seeds "$SEED" --open-world-folds 5 --folds "$FOLD" \
    --epochs 15 --batch-size 64 --num-workers 4

echo "AWA2_OPEN_WORLD_DETECTOR_DONE fold=$FOLD seed=$SEED"
