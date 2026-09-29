#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
EXTERNAL="$ROOT/external"

SAMRULE_REV=450fb65af450dcd10a135e25ce37fc3a1e459f5e
RRL_REV=f8d0886b23c4e15f63c62c248b97d4eb73386ad1
RLNET_REV=2e7ad9504ff93f7e6c5320ddfa222ed1d8779289

clone_at_revision() {
    local label=$1
    local url=$2
    local destination=$3
    local revision=$4

    if [ -e "$destination" ] && [ ! -d "$destination/.git" ]; then
        echo "ERROR: $destination exists but is not a Git checkout"
        exit 1
    fi

    if [ ! -d "$destination/.git" ]; then
        echo "Cloning $label into $destination"
        git clone "$url" "$destination"
    fi

    if [ -n "$(git -C "$destination" status --porcelain --untracked-files=no)" ]; then
        echo "ERROR: $label has modified tracked files at $destination"
        echo "Commit or revert those changes before rerunning this setup."
        exit 1
    fi

    if ! git -C "$destination" cat-file -e "$revision^{commit}" 2>/dev/null; then
        echo "Fetching pinned $label revision $revision"
        git -C "$destination" fetch origin "$revision"
    fi
    git -C "$destination" checkout --detach "$revision"

    local actual
    actual=$(git -C "$destination" rev-parse HEAD)
    if [ "$actual" != "$revision" ]; then
        echo "ERROR: $label checkout is $actual, expected $revision"
        exit 1
    fi
    echo "$label ready at $actual"
}

command -v git >/dev/null || { echo "ERROR: git is required"; exit 1; }
command -v make >/dev/null || { echo "ERROR: make is required"; exit 1; }
command -v micromamba >/dev/null || { echo "ERROR: micromamba is required"; exit 1; }

mkdir -p "$EXTERNAL" "$ROOT/logs/benchmark2"

clone_at_revision \
    SamRuLe \
    https://github.com/VandinLab/SamRuLe.git \
    "$EXTERNAL/SamRuLe" \
    "$SAMRULE_REV"

clone_at_revision \
    RRL \
    https://github.com/12wang3/rrl.git \
    "$EXTERNAL/rrl" \
    "$RRL_REV"

clone_at_revision \
    RL-Net \
    https://github.com/luciledierckx/RLNet.git \
    "$EXTERNAL/RLNet" \
    "$RLNET_REV"

echo "Building the SamRuLe CORELS target"
make -C "$EXTERNAL/SamRuLe/src" corels NGMP=1

test -x "$EXTERNAL/SamRuLe/src/corels" || {
    echo "ERROR: SamRuLe CORELS build did not create src/corels"
    exit 1
}
test -f "$EXTERNAL/rrl/rrl/models.py" || {
    echo "ERROR: RRL checkout is missing rrl/models.py"
    exit 1
}
test -f "$EXTERNAL/RLNet/networkTorch_multiClass.py" || {
    echo "ERROR: RL-Net checkout is missing networkTorch_multiClass.py"
    exit 1
}

echo "Checking datasci dependencies"
micromamba run -n datasci python -c "import joblib, imodels, numpy, sklearn"

echo "Checking gpuenv dependencies"
micromamba run -n gpuenv python -c "import joblib, numpy, pandas, sklearn, torch"

echo
echo "Comparison dependencies are ready under $EXTERNAL"
echo "Datasets default to \$HOME/Datasets/keel_datasets; pass KEEL_DIR with qsub if needed."
echo "Iris smoke:"
echo "  qsub -v BENCHMARK_SMOKE=1 -t 1-6 experiments/benchmark2/comparison_cluster.sh"
echo "Final array:"
echo '  CMP_JOB=$(qsub -terse experiments/benchmark2/comparison_cluster.sh)'
echo '  qsub -hold_jid "${CMP_JOB%%.*}" experiments/benchmark2/comparison_score_cluster.sh'
