#!/bin/bash
#$ -cwd
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q all.q
#$ -o ablation_$JOB_ID.log

# PBS/SGE Job Script for FuzzyCART Ablation Studies
# ==================================================
# This script runs comprehensive ablation studies on 10 datasets
# using the FuzzyCART algorithm

echo "======================================================================"
echo "FuzzyCART Ablation Studies - 10 Datasets"
echo "======================================================================"
echo "Job ID: $JOB_ID"
echo "Hostname: $HOSTNAME"
echo "Date: $(date)"
echo "======================================================================"
echo ""

# Load conda environment
echo "Activating conda environment: datasci"
conda activate datasci

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'datasci'"
    echo "Please ensure the environment exists: conda env list"
    exit 1
fi

echo "✓ Conda environment 'datasci' activated"
echo ""

# Verify Python and required packages
echo "Verifying Python environment..."
python --version
echo ""

python -c "import numpy, pandas, sklearn, scipy, matplotlib, seaborn, ex_fuzzy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Required packages not found in 'datasci' environment"
    echo "Please install: numpy pandas scikit-learn scipy matplotlib seaborn ex_fuzzy"
    exit 1
fi
echo "✓ Required packages verified"
echo ""

# Navigate to working directory
WORK_DIR=$PWD
echo "Working directory: $WORK_DIR"
echo ""

# Create results directory
mkdir -p results
echo "✓ Results directory created/verified"
echo ""

# Run ablation studies
echo "======================================================================"
echo "Starting Ablation Studies"
echo "======================================================================"
echo "This will run ablation studies on all 10 datasets:"
echo "  - iris, wine, vehicle, vowel, glass"
echo "  - ecoli, yeast, segment, pendigits, optdigits"
echo ""
echo "Studies performed:"
echo "  1. max_rules parameter impact"
echo "  2. coverage_threshold parameter impact"
echo "  3. min_improvement parameter impact"
echo "  4. pruning (ccp_alpha) parameter impact"
echo "  5. grid search for optimal hyperparameters"
echo ""
echo "Estimated runtime: 24-48 hours"
echo "======================================================================"
echo ""

# Record start time
START_TIME=$(date +%s)

# Run the ablation studies
python ablation_studies.py

EXIT_CODE=$?

# Record end time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "======================================================================"
echo "Ablation Studies Complete!"
echo "======================================================================"
echo "Exit code: $EXIT_CODE"
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Date: $(date)"
echo ""
echo "Results saved to: $WORK_DIR/results/"
echo ""
echo "Output files:"
echo "  Individual results:"
echo "    - ablation_<dataset>_max_rules.csv"
echo "    - ablation_<dataset>_coverage_threshold.csv"
echo "    - ablation_<dataset>_min_improvement.csv"
echo "    - ablation_<dataset>_pruning.csv"
echo "    - optimization_<dataset>.csv"
echo ""
echo "  Combined results:"
echo "    - ablation_ALL_DATASETS_max_rules.csv"
echo "    - ablation_ALL_DATASETS_coverage_threshold.csv"
echo "    - ablation_ALL_DATASETS_min_improvement.csv"
echo "    - ablation_ALL_DATASETS_pruning.csv"
echo ""
echo "  Summary visualizations:"
echo "    - ablation_SUMMARY_max_rules.png"
echo "    - ablation_SUMMARY_coverage_threshold.png"
echo "    - ablation_SUMMARY_min_improvement.png"
echo "    - ablation_SUMMARY_pruning.png"
echo ""
echo "======================================================================"

# Deactivate conda environment
conda deactivate

exit $EXIT_CODE
