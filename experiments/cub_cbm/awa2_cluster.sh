#!/bin/bash
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N awa2_cbm
#$ -o logs/awa2_cbm
#$ -q gpu.q
#$ -l gpu=1

# AwA2 concept-bottleneck pipeline on an SGE cluster:
#   1. build the oracle artifact bundle from the AwA2 metadata (CPU, fast)
#   2. train N attribute detectors and export predicted concepts (GPU)
#   3. run the full FERL CBM suite (E1-E4 + extras E5-E11) on both regimes
#
# Submit from the repo root:  qsub experiments/cub_cbm/awa2_cluster.sh
# The detector step is the reason this is a gpu.q job; steps 1 and 3 are CPU
# but kept here so the whole pipeline is one submission.

source /usr/local/gpuallocation.sh
micromamba activate gpuenv

mkdir -p logs/awa2_cbm

AWA2=~/Datasets/Animals_with_Attributes2
ART=results/awa2_cbm_artifacts/awa2
OUT=results/awa2_cbm_perf
SEEDS="0 1 2"

export PYTHONPATH=.

# fail fast if the AwA2 image tree is not where we expect it (the usual cause
# of a run that "finishes" with only FileNotFoundError cascades).
if [ ! -f "$AWA2/predicate-matrix-binary.txt" ] || [ ! -d "$AWA2/JPEGImages" ]; then
    echo "ERROR: AwA2 dataset not found at $AWA2"
    echo "  need classes.txt, predicates.txt, predicate-matrix-binary.txt, JPEGImages/"
    echo "  edit AWA2= in this script or place the dataset there."
    exit 1
fi

# 1. oracle bundle (skip if already built) -- must succeed before anything else
if [ ! -f "$ART/oracle_train.npz" ]; then
    python3 experiments/cub_cbm/make_awa2_artifact.py --awa2-root "$AWA2" --out-dir "$ART" \
        || { echo "ERROR: AwA2 oracle generation failed"; exit 1; }
fi
if [ ! -f "$ART/oracle_train.npz" ]; then
    echo "ERROR: oracle bundle missing after generation at $ART; aborting"; exit 1
fi

# 2. detectors -> predicted_seed<k>_*.npz (GPU). Needs cached ImageNet weights;
#    if the cluster has no internet, pre-populate $TORCH_HOME/hub/checkpoints.
#    batch 64 + AMP fits ResNet50 on the ~11GB GTX 1080Ti; drop to 32 if it OOMs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 experiments/cub_cbm/train_awa2_detector.py \
    --awa2-root "$AWA2" --artifact-dir "$ART" \
    --seeds $SEEDS --epochs 15 --batch-size 64 --num-workers 4

# 3. FERL CBM suite on oracle + predicted (runs under gpuenv too, which must
#    have ferl + ex_fuzzy on top of torch). Clear the append-mode result tables
#    first so a rerun is fresh and non-duplicated (PNG/txt overwrite by name).
rm -f "$OUT"/e1_*.csv "$OUT"/e2_*.csv "$OUT"/e3_*.csv "$OUT"/e4_*.csv \
      "$OUT"/e5_*.csv "$OUT"/e6_*.csv "$OUT"/e7_*.csv "$OUT"/e8*_*.csv \
      "$OUT"/e9_*.csv "$OUT"/e10_*.csv "$OUT"/e11_*.csv "$OUT"/e15_*.csv
python3 experiments/cub_cbm/run_cub_cbm.py \
    --artifact-dir "$ART" --subset awa2 \
    --concept-sources oracle predicted --detector-seeds $SEEDS \
    --experiments all --learned-depth 60 --output-dir "$OUT"

python3 experiments/cub_cbm/run_cub_cbm_extras.py \
    --artifact-dir "$ART" --subset awa2 --detector-seeds $SEEDS \
    --experiments all --learned-depth 60 \
    --betas 0.02 0.05 0.1 0.15 0.2 0.3 --output-dir "$OUT"

# Calibration materially changes the CUB intervention regime and is retained
# here as a pre-specified AwA2 comparison. E7 above already evaluates both
# variants internally; E10/E11 require the explicit calibrated invocation.
python3 experiments/cub_cbm/run_cub_cbm_extras.py \
    --artifact-dir "$ART" --subset awa2 --detector-seeds $SEEDS \
    --experiments e10 e11 --learned-depth 60 --calibrate-concepts \
    --output-dir "$OUT"

echo AWA2_CBM_DONE
