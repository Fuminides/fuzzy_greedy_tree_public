# FGRT Experimental Test Suite

This directory contains comprehensive experimental code for testing and evaluating FGRT (Fuzzy Greedy Rule Tree), a fuzzy tree learning algorithm for classification.

## Overview

The test suite implements the experimental protocol described in the FGRT paper:
- 5-fold stratified cross-validation
- Comparison with CART, C4.5, and other baselines
- 10 UCI/Keel benchmark datasets
- Metrics: Accuracy, number of rules, number of conditions, training time
- Statistical testing: Friedman test + Post-hoc Nemenyi

## File Structure

```
├── test_runner.py              # Main orchestrator for all experiments
├── fuzzy_cart_experiments.py   # Core benchmark experiments
├── ablation_studies.py         # Parameter sensitivity analysis
├── performance_analysis.py     # Runtime and scalability analysis
└── README.md                   # This file
```

## Installation and Setup

### Prerequisites

```bash
# Required packages
pip install numpy pandas scikit-learn scipy scikit-posthocs matplotlib seaborn --break-system-packages

# The FGRT implementation should be in tree_learning.py
# The ex-fuzzy library is required for fuzzy set operations
```

### Directory Structure

Ensure the following directory structure:
```
/mnt/project/
    ├── tree_learning.py        # FGRT implementation
    └── (other project files)

/home/claude/
    ├── test_runner.py
    ├── fuzzy_cart_experiments.py
    ├── ablation_studies.py
    └── performance_analysis.py

/mnt/user-data/outputs/         # Results will be saved here
```

## Usage

### Quick Start - Run All Tests

```bash
python test_runner.py --all
```

This will:
1. Run basic functionality tests on Iris and Wine datasets
2. Run benchmark comparisons between FGRT, CART, and C4.5
3. Perform ablation studies on key hyperparameters
4. Analyze runtime performance and scalability
5. Generate a comprehensive HTML report

### Run Specific Test Suites

#### Basic Functionality Tests
```bash
python test_runner.py --basic
```
Tests FGRT on simple datasets to verify correct operation.

#### Benchmark Comparisons
```bash
python test_runner.py --benchmarks
```
Runs 5-fold CV comparisons with baseline methods.

#### Ablation Studies
```bash
python test_runner.py --ablation
```
Analyzes the impact of:
- max_rules parameter
- coverage_threshold parameter
- min_improvement parameter
- Pruning (ccp_alpha)

#### Performance Analysis
```bash
python test_runner.py --performance
```
Measures:
- Training time vs. dataset size
- Training time vs. number of features
- Prediction time analysis
- Memory consumption

### Individual Script Usage

Each script can also be run independently:

#### 1. Benchmark Experiments
```bash
python fuzzy_cart_experiments.py
```

Features:
- DatasetLoader: Loads UCI/Keel datasets
- Fuzzy partitions: Creates trapezoidal fuzzy partitions using ex_fuzzy
- ExperimentRunner: Runs 5-fold CV experiments
- StatisticalAnalyzer: Performs Friedman + Nemenyi tests

#### 2. Ablation Studies
```bash
python ablation_studies.py
```

Features:
- AblationStudy: Tests individual parameter impacts
- ParameterOptimizer: Grid search for optimal parameters
- VisualizationTools: Plots parameter impact

#### 3. Performance Analysis
```bash
python performance_analysis.py
```

Features:
- PerformanceAnalyzer: Measures training/prediction times
- PerformanceVisualizer: Creates performance plots
- Memory profiling

## Customization

### Adding New Datasets

To add datasets, modify `fuzzy_cart_experiments.py`:

```python
# In DatasetLoader class
DATASET_NAMES = [
    'iris', 'wine', 'your_dataset',
    # ... add more
]

def load_dataset(self, name):
    if name == 'your_dataset':
        X, y = load_your_data()
        return X, y
    # ... existing code
```

### Modifying Fuzzy Partitions

To change partition strategy, use ex_fuzzy utilities:

```python
from ex_fuzzy import utils
from ex_fuzzy import fuzzy_sets as fs

# Create custom partitions
fuzzy_partitions = utils.construct_partitions(
    X, 
    n_partitions=3,  # Number of linguistic terms
    fuzzy_type=fs.FUZZY_SETS.t1
)
```

### Adjusting Experimental Parameters

Modify in respective scripts:

```python
# Number of folds
runner = ExperimentRunner(n_folds=5)  # Change to 10 for more robust results

# FGRT configurations
configs = [
    {'max_rules': 15, 'coverage_threshold': 0.01, 'min_improvement': 0.01},
    {'max_rules': 20, 'coverage_threshold': 0.03, 'min_improvement': 0.01},
    # Add more configurations
]
```

## Output Files

All results are saved to `/mnt/user-data/outputs/`:

### Basic Tests
- `basic_tests.csv`: Results of functionality tests

### Benchmark Tests  
- `fuzzy_cart_results.csv`: Detailed 5-fold CV results
- `fuzzy_cart_summary.csv`: Average performance across datasets
- `benchmark_results.csv`: Complete comparison table
- `benchmark_summary.csv`: Summary statistics

### Ablation Studies
- `ablation_max_rules.csv`: Impact of max_rules
- `ablation_coverage.csv`: Impact of coverage_threshold
- `ablation_min_improvement.csv`: Impact of min_improvement
- `ablation_pruning.csv`: Impact of CCP alpha
- `parameter_optimization.csv`: Grid search results
- `ablation_*_plot.png`: Visualization plots

### Performance Analysis
- `performance_vs_samples.csv`: Training time vs. dataset size
- `performance_vs_features.csv`: Training time vs. features
- `prediction_time.csv`: Prediction time analysis
- `performance_*.png`: Performance plots

### Reports
- `test_report.html`: Comprehensive HTML report of all results

## Expected Results

Based on the paper, you should expect:

### Accuracy
- FGRT: ~79-80% average accuracy
- CART: ~81% average accuracy  
- C4.5: ~80% average accuracy

### Complexity
- FGRT: ~13-15 rules, ~2-3 conditions/rule
- CART: ~30-40 rules
- C4.5: ~130-140 rules

### Training Time
- FGRT: Similar to CART (O(n log n))
- Substantially faster than evolutionary approaches

## Statistical Testing

The suite includes:

### Friedman Test
Tests if there are significant differences between methods across datasets.
- H0: All methods have equal performance
- Reject H0 if p < 0.05

### Post-hoc Nemenyi Test  
Pairwise comparisons between methods.
- Shows which specific method pairs differ significantly
- Results presented as p-value matrix

## Troubleshooting

### Import Errors
```bash
# If ex-fuzzy is not found
pip install ex-fuzzy --break-system-packages

# Or ensure fuzzy_sets.py is in the path
export PYTHONPATH=$PYTHONPATH:/mnt/project
```

### Memory Issues
For large datasets, reduce:
```python
# In ablation_studies.py or performance_analysis.py
sample_sizes = [100, 500, 1000]  # Instead of [100, 500, 1000, 5000, 10000]
n_trials = 2  # Instead of 3 or 5
```

### Slow Execution
To speed up:
```python
# Reduce cross-validation folds
runner = ExperimentRunner(n_folds=3)  # Instead of 5

# Use fewer datasets
datasets_to_test = ['iris', 'wine']  # Instead of all 10

# Reduce parameter grid
param_grid = {
    'max_rules': [15],  # Instead of [10, 15, 20]
    'coverage_threshold': [0.01],  # Instead of [0.01, 0.03, 0.05]
}
```

## Extending the Test Suite

### Adding New Baselines

In `fuzzy_cart_experiments.py`:

```python
def _test_your_method(self, X_train, X_test, y_train, y_test, results_dict):
    """Test your baseline method."""
    start_time = time.time()
    
    your_model = YourClassifier()
    your_model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    y_pred = your_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    results_dict['accuracy'].append(accuracy)
    results_dict['train_time'].append(train_time)
    # Add complexity metrics as applicable
```

### Adding New Metrics

Modify the metrics tracking in `ExperimentRunner`:

```python
methods_results = {
    'FuzzyCART': {
        'accuracy': [], 
        'n_rules': [], 
        'your_metric': [],  # Add here
        # ...
    }
}
```

## Contact

For questions or issues:
- Check the FGRT implementation in `tree_learning.py`
- Review the paper: `_2026_WCCI__Fuzzy_CART_v3.pdf`
- See the FRR paper for additional context: `FRR__IEEE_TKD_2025_3.pdf`

## License

This test suite is provided as-is for research purposes. Refer to the main project license for details.
