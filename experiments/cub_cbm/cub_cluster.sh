#!/bin/bash
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N cub_cbm
#$ -o logs/cub_cbm
#$ -q all.q

# Full FERL concept-bottleneck suite on CUB, regenerated cleanly from the
# existing concept artifacts (results/cub_cbm_artifacts/). CPU only: unlike
# AwA2, CUB's detector-predicted concepts already exist, so there is no GPU
# training step -- this just reruns the experiments (E1-E4 + extras E5-E8,
# raw and concept-calibrated) across all detector seeds on both subsets.
#
# Submit from the repo root:  qsub experiments/cub_cbm/cub_cluster.sh
# Long batch job (the full 200-class subset with the deep learned tree and the
# E8 cross-fitting dominates the wall time).

micromamba activate datasci

mkdir -p logs/cub_cbm

A=results/cub_cbm_artifacts
OUT=results/cub_cbm_perf
BETAS="0.02 0.05 0.1 0.15 0.2 0.3"

export PYTHONPATH=.
mkdir -p "$OUT"

# fail fast if the concept artifacts are absent. They are git-ignored data
# (only the small JSON manifests are tracked), so a fresh cluster checkout has
# NONE of the .npz files and cannot regenerate them (the CUB concept detector
# is out-of-repo). Copy them from the machine that produced them, e.g.:
#   rsync -avz <host>:.../results/cub_cbm_artifacts/ results/cub_cbm_artifacts/
for sub in 20 full; do
    if [ ! -f "$A/cub_koh112_$sub/oracle_train.npz" ]; then
        echo "ERROR: CUB artifacts missing at $A/cub_koh112_$sub/ (git-ignored data)."
        echo "  rsync them from the machine that has them (see comment above); aborting."
        exit 1
    fi
done

# clean the append-mode result tables so this is a fresh, non-duplicated run
# (PNG/txt dumps overwrite by name and need no cleaning)
rm -f "$OUT"/e1_*.csv "$OUT"/e2_*.csv "$OUT"/e3_*.csv "$OUT"/e4_*.csv \
      "$OUT"/e5_*.csv "$OUT"/e6_*.csv "$OUT"/e7_*.csv "$OUT"/e8*_*.csv

# subset  depth  detector-seeds
run_subset () {
    local subset="$1" depth="$2"; shift 2
    local seeds="$*"
    echo "=== CUB subset=$subset depth=$depth seeds=$seeds ==="
    python3 experiments/cub_cbm/run_cub_cbm.py \
        --artifact-dir "$A/cub_koh112_$subset" --subset "$subset" \
        --concept-sources oracle predicted --detector-seeds $seeds \
        --experiments all --learned-depth "$depth" --output-dir "$OUT"
    python3 experiments/cub_cbm/run_cub_cbm_extras.py \
        --artifact-dir "$A/cub_koh112_$subset" --subset "$subset" \
        --detector-seeds $seeds --experiments e5 e6 e7 e8 \
        --learned-depth "$depth" --betas $BETAS --output-dir "$OUT"
    python3 experiments/cub_cbm/run_cub_cbm_extras.py \
        --artifact-dir "$A/cub_koh112_$subset" --subset "$subset" \
        --detector-seeds $seeds --experiments e8 --calibrate-concepts \
        --learned-depth "$depth" --betas $BETAS --output-dir "$OUT"
}

run_subset 20   12 0 1 2
run_subset full 60 0 1 2 3 4

echo CUB_CBM_DONE
