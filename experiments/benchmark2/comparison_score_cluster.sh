#!/bin/bash
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N ferl_cmp_score
#$ -o logs/benchmark2
#$ -q all.q

# Submit after the comparison array finishes, replacing <array-job-id>:
#   qsub -hold_jid <array-job-id> experiments/benchmark2/comparison_score_cluster.sh

micromamba activate gpuenv
if [ $? -ne 0 ]; then
    echo "ERROR: could not activate the gpuenv environment"
    exit 1
fi

export PYTHONPATH=.
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${JOB_ID}"

python3 experiments/benchmark2/validate_results.py
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: comparison artifacts are incomplete or malformed; scoring aborted"
    exit "$STATUS"
fi

python3 experiments/benchmark2/score.py
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: scoring exited with status $STATUS"
    exit "$STATUS"
fi

python3 experiments/benchmark2/submission_analysis.py --strict
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: submission statistical analysis exited with status $STATUS"
    exit "$STATUS"
fi

echo COMPARISON_SCORE_DONE
