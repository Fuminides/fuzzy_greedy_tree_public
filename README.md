# Fuzzy Greedy Rule Tree with Partition Optimization

This repository contains the implementation of Fuzzy Greedy Rule Tree (FGRT) with optimized fuzzy partition learning, including comprehensive experimental code for benchmarking and analysis.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

FGRT is a fuzzy decision tree algorithm that combines greedy rule learning with fuzzy logic. This implementation extends the base algorithm with automatic optimization of fuzzy partitions using an innovative interleaved encoding scheme and multiple search strategies.

Key features:
- Greedy rule-based decision tree learning
- Fuzzy membership functions with automatic partition optimization
- Three search strategies: grid search, coordinate descent, and hybrid
- Early stopping optimization for faster convergence
- Comprehensive experimental framework with statistical testing
- Support for both continuous and categorical features

## Installation

### Prerequisites

```bash
pip install numpy pandas scikit-learn scipy matplotlib ex-fuzzy
```

Optional dependencies for statistical testing:
```bash
pip install scikit-posthocs seaborn
```

**Recommended**: For 10-12x faster training, install Numba:
```bash
pip install numba
```

### Quick Setup

```bash
git clone https://github.com/Fuminides/fuzzy_greedy_tree_public.git
cd fuzzy_greedy_tree_public
pip install -r requirements.txt
```

## Repository Structure

```
├── tree_learning.py                    # Core FGRT implementation
├── partition_optimization.py           # Partition optimization framework
├── data_loaders.py                     # Dataset utilities
├── fuzzy_cart_experiments.py           # Benchmark experiments
├── ablation_studies.py                 # Parameter sensitivity analysis
├── performance_analysis.py             # Runtime and scalability analysis
├── test_runner.py                      # Main test orchestrator
├── run_experiments.sh                  # Automated experiment runner
├── regenerate_figures.py               # Generate paper figures
├── examples/
│   ├── minimal_example.py              # Basic usage example
│   └── strategy_comparison_example.py  # Strategy comparison
└── docs/
    ├── QUICKSTART.md                   # Quick start guide
    ├── TECHNICAL_REPORT_PARTITION_OPTIMIZATION.md
    ├── SEARCH_STRATEGIES.md            # Strategy comparison
    └── PARTITION_OPTIMIZATION_EXPERIMENTS.md
```

## Usage

### Basic Usage

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ex_fuzzy import utils
from tree_learning import FuzzyCART

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create default fuzzy partitions
fuzzy_partitions = utils.construct_partitions(X_train, n_partitions=3)

# Train model with performance optimizations
model = FuzzyCART(
    fuzzy_partitions=fuzzy_partitions,
    max_depth=5,
    max_rules=15,
    target_metric='purity',       # Use purity for Numba acceleration
    early_stop_threshold=0.05,    # Early stopping (1.3-4x speedup)
    use_numba=True                # Numba JIT compilation (up to 10x speedup)
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
```

### Usage with Partition Optimization

```python
from partition_optimization import optimize_partitions_for_gfrt

# Optimize fuzzy partitions
optimized_partitions = optimize_partitions_for_gfrt(
    X_train, y_train,
    initial_partitions=fuzzy_partitions,
    strategy='hybrid',  # Options: 'grid', 'coordinate', 'hybrid'
    verbose=True
)

# Train with optimized partitions
model = FuzzyCART(
    fuzzy_partitions=optimized_partitions,
    max_depth=5,
    max_rules=15,
    target_metric='purity',
    early_stop_threshold=0.05,
    use_numba=True
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Running Experiments

Run complete benchmark suite:
```bash
python test_runner.py --all
```

Run specific experiments:
```bash
python test_runner.py --benchmarks    # Benchmark comparisons
python test_runner.py --ablation      # Parameter studies
python test_runner.py --performance   # Performance analysis
```

Run automated experimental protocol:
```bash
bash run_experiments.sh
```

### Examples

Simple demonstration:
```bash
python examples/minimal_example.py
```

Compare optimization strategies:
```bash
python examples/strategy_comparison_example.py
```

## Partition Optimization

The framework implements three search strategies for optimizing fuzzy partition parameters:

### Grid Search (Exhaustive)
- Evaluates all combinations in a discrete grid
- Most thorough but computationally expensive
- 216 evaluations per feature (6^3 grid)
- Deterministic results

### Coordinate Descent (Fast)
- Optimizes one parameter at a time iteratively
- Adaptive convergence
- 30-84 evaluations per feature (avg: ~50)
- 3-4x speedup over grid search

### Hybrid (Balanced)
- Coarse grid search followed by coordinate refinement
- Best balance of thoroughness and speed
- ~120 evaluations per feature
- 2-3x speedup over grid search
- Recommended default for production use


## Performance Optimization

FuzzyCART includes built-in performance optimizations for faster training:

### Numba JIT Compilation

When Numba is installed, core computational loops are JIT-compiled for significant speedup:

```python
model = FuzzyCART(
    fuzzy_partitions=partitions,
    target_metric='purity',  # Required for Numba acceleration
    use_numba=True           # Enable Numba JIT (default: True)
)
```

### Early Stopping

Terminate split search early when a "good enough" improvement is found:

```python
model = FuzzyCART(
    fuzzy_partitions=partitions,
    early_stop_threshold=0.05  # Stop when improvement >= 5% (0.0 to disable)
)
```

### Training Speedup Benchmarks

| Optimization | Speedup | Notes |
|-------------|--------:|-------|
| Numba JIT only | **10x** | Requires `target_metric='purity'` |
| Early stopping only | **1.3-4x** | Works with any metric |
| Combined | **12x** | Maximum performance |

**Note**: First run includes Numba compilation overhead (~1-2 seconds). Subsequent runs use cached compiled code.


## Experimental Framework

The repository includes a comprehensive experimental suite for:

### Benchmark Comparisons
- 5-fold stratified cross-validation
- Comparison with CART, C4.5, and other baselines
- UCI/Keel benchmark datasets
- Statistical testing: Friedman + Nemenyi post-hoc tests

### Ablation Studies
- Parameter sensitivity analysis (max_rules, coverage_threshold, etc.)
- Impact of pruning (CCP alpha)
- Partition count optimization

### Performance Analysis
- Training time vs. dataset size
- Training time vs. number of features
- Prediction time analysis
- Memory consumption profiling

## Results

### Accuracy
Based on benchmark experiments:
- FGRT: 79-80% average accuracy across datasets
- CART: ~81% average accuracy
- C4.5: ~80% average accuracy

### Model Complexity
- FGRT: 13-15 rules, 2-3 conditions per rule
- CART: 30-40 rules
- C4.5: 130-140 rules

## Documentation

- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [TECHNICAL_REPORT_PARTITION_OPTIMIZATION.md](TECHNICAL_REPORT_PARTITION_OPTIMIZATION.md) - Detailed technical documentation
- [SEARCH_STRATEGIES.md](SEARCH_STRATEGIES.md) - Strategy comparison and selection guide
- [PARTITION_OPTIMIZATION_EXPERIMENTS.md](PARTITION_OPTIMIZATION_EXPERIMENTS.md) - Experimental protocol

## Testing

Run basic validation tests:
```bash
python test_optimization_experiment.py
python test_search_strategies.py
```

Generate visualizations:
```bash
python visualize_strategies.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{fumanal2025fast,
  title={A Fast Interpretable Fuzzy Tree Learner},
  author={Fumanal-Idocin, Javier and Fernandez-Peralta, Raquel and Andreu-Perez, Javier},
  journal={arXiv preprint arXiv:2512.11616},
  year={2025}
}

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.
