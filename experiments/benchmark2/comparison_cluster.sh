#!/bin/bash
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N ferl_cmp
#$ -o logs/benchmark2
#$ -q all.q
#$ -t 1-122
#$ -tc 15

# CPU array for the comparison-paper methods. Full mapping:
#   1     SampledRuleList proxy on all 30 datasets
#   2     RRL on all 30 datasets
#   3-32  SamRuLe-OVR, one dataset per task
#   33-62 RL-Net, one dataset per task
#   63-92 FuzzyUCS-DS, one dataset per task
#   93-122 NeuRules, one dataset per task
#
# Submit all final runs from the repository root:
#   qsub experiments/benchmark2/comparison_cluster.sh
#
# Submit a six-task Iris smoke first:
#   qsub -v BENCHMARK_SMOKE=1 -t 1-6 experiments/benchmark2/comparison_cluster.sh
#
# Run ./make_comparison.sh once on the cluster login node before submission.

export PYTHONPATH=.
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${JOB_ID}-${SGE_TASK_ID}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export KEEL_DIR="${KEEL_DIR:-$HOME/Datasets/keel_datasets}"
export FERL_SAMRULE_TIMEOUT="${FERL_SAMRULE_TIMEOUT:-0}"   # 0 = no CORELS time limit
export FERL_SAMRULE_REPO="${FERL_SAMRULE_REPO:-$PWD/external/SamRuLe}"
export FERL_RRL_REPO="${FERL_RRL_REPO:-$PWD/external/rrl}"
export FERL_RLNET_REPO="${FERL_RLNET_REPO:-$PWD/external/RLNet}"

DATASETS=(
    magic penbased ring twonorm satimage optdigits texture phoneme banana spambase
    segment contraceptive german vowel vehicle mammographic pima australian wisconsin crx
    balance wdbc saheart bupa ionosphere ecoli spectfheart heart glass wine
)
METHODS=(SampledRuleList SamRuLe-OVR RRL RL-Net FuzzyUCS-DS NeuRules)

if [ -z "$SGE_TASK_ID" ]; then
    echo "ERROR: SGE_TASK_ID is unset; submit this file with qsub"
    exit 1
fi

if [ "$BENCHMARK_SMOKE" = "1" ]; then
    if [ "$SGE_TASK_ID" -gt 6 ]; then
        echo "ERROR: smoke mode requires qsub -t 1-6"
        exit 1
    fi
    METHOD=${METHODS[$((SGE_TASK_ID - 1))]}
    TASK_DATASETS=(iris)
else
    if [ "$SGE_TASK_ID" -eq 1 ]; then
        METHOD=SampledRuleList
        TASK_DATASETS=("${DATASETS[@]}")
    elif [ "$SGE_TASK_ID" -eq 2 ]; then
        METHOD=RRL
        TASK_DATASETS=("${DATASETS[@]}")
    elif [ "$SGE_TASK_ID" -le 32 ]; then
        METHOD=SamRuLe-OVR
        TASK_DATASETS=("${DATASETS[$((SGE_TASK_ID - 3))]}")
    elif [ "$SGE_TASK_ID" -le 62 ]; then
        METHOD=RL-Net
        TASK_DATASETS=("${DATASETS[$((SGE_TASK_ID - 33))]}")
    elif [ "$SGE_TASK_ID" -le 92 ]; then
        METHOD=FuzzyUCS-DS
        TASK_DATASETS=("${DATASETS[$((SGE_TASK_ID - 63))]}")
    else
        METHOD=NeuRules
        TASK_DATASETS=("${DATASETS[$((SGE_TASK_ID - 93))]}")
    fi
fi

case "$METHOD" in
    RRL|RL-Net|NeuRules)
        ENV_NAME=gpuenv
        ;;
    *)
        ENV_NAME=datasci
        ;;
esac

micromamba activate "$ENV_NAME"
if [ $? -ne 0 ]; then
    echo "ERROR: could not activate the $ENV_NAME environment"
    exit 1
fi

for DATASET in "${TASK_DATASETS[@]}"; do
    if [ ! -f "$KEEL_DIR/$DATASET/$DATASET.dat" ]; then
        echo "ERROR: KEEL dataset missing: $KEEL_DIR/$DATASET/$DATASET.dat"
        echo "Set KEEL_DIR with: qsub -v KEEL_DIR=/path/to/keel_datasets ..."
        exit 1
    fi
done

python3 -c "import numpy, sklearn"
if [ $? -ne 0 ]; then
    echo "ERROR: $ENV_NAME must provide numpy and scikit-learn"
    exit 1
fi

if [ "$METHOD" = "SampledRuleList" ]; then
    python3 -c "import imodels"
    if [ $? -ne 0 ]; then
        echo "ERROR: SampledRuleList requires imodels in the datasci environment"
        exit 1
    fi
fi

if [ "$METHOD" = "SamRuLe-OVR" ]; then
    if [ ! -f "$FERL_SAMRULE_REPO/src/corels" ]; then
        echo "ERROR: built SamRuLe CORELS not found at $FERL_SAMRULE_REPO/src/corels"
        echo "Clone VandinLab/SamRuLe and run: make -C external/SamRuLe/src corels NGMP=1"
        exit 1
    fi
    echo "SamRuLe commit: $(git -C "$FERL_SAMRULE_REPO" rev-parse HEAD 2>/dev/null)"
fi

if [ "$METHOD" = "RRL" ]; then
    if [ ! -f "$FERL_RRL_REPO/rrl/models.py" ]; then
        echo "ERROR: public RRL checkout not found at $FERL_RRL_REPO"
        echo "Clone https://github.com/12wang3/rrl before submitting."
        exit 1
    fi
    python3 -c "import torch"
    if [ $? -ne 0 ]; then
        echo "ERROR: RRL requires PyTorch in gpuenv"
        exit 1
    fi
    echo "RRL commit: $(git -C "$FERL_RRL_REPO" rev-parse HEAD 2>/dev/null)"
fi

if [ "$METHOD" = "NeuRules" ]; then
    python3 -c "import torch"
    if [ $? -ne 0 ]; then
        echo "ERROR: NeuRules requires PyTorch in gpuenv"
        exit 1
    fi
fi

if [ "$METHOD" = "RL-Net" ]; then
    if [ ! -f "$FERL_RLNET_REPO/networkTorch_multiClass.py" ]; then
        echo "ERROR: public RL-Net checkout not found at $FERL_RLNET_REPO"
        echo "Clone https://github.com/luciledierckx/RLNet before submitting."
        exit 1
    fi
    python3 -c "import torch"
    if [ $? -ne 0 ]; then
        echo "ERROR: RL-Net requires PyTorch in gpuenv"
        exit 1
    fi
    echo "RL-Net commit: $(git -C "$FERL_RLNET_REPO" rev-parse HEAD 2>/dev/null)"
fi

echo "=== task=$SGE_TASK_ID host=$HOSTNAME env=$ENV_NAME method=$METHOD datasets=${TASK_DATASETS[*]} ==="
python3 experiments/benchmark2/harness.py --strict --datasets "${TASK_DATASETS[@]}" --models "$METHOD"
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: harness exited with status $STATUS"
    exit "$STATUS"
fi

echo "COMPARISON_TASK_DONE method=$METHOD datasets=${TASK_DATASETS[*]}"
