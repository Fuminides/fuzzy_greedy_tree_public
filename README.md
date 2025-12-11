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
fuzzy_vars = utils.construct_partitions(X_train, n_partitions=3)

# Train model
model = FuzzyCART(max_depth=5, max_rules=15)
model.fit(X_train, y_train, fuzzy_variables=fuzzy_vars)

# Predict
y_pred = model.predict(X_test, fuzzy_variables=fuzzy_vars)
```

### Usage with Partition Optimization

```python
from partition_optimization import optimize_partitions_for_gfrt

# Optimize fuzzy partitions
optimized_vars, metrics = optimize_partitions_for_gfrt(
    X_train, y_train,
    initial_fuzzy_vars=fuzzy_vars,
    strategy='hybrid',  # Options: 'grid', 'coordinate', 'hybrid'
    verbose=True
)

# Train with optimized partitions
model.fit(X_train, y_train, fuzzy_variables=optimized_vars)
y_pred = model.predict(X_test, fuzzy_variables=optimized_vars)
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

### Performance Comparison

| Strategy   | Evaluations/Feature | Relative Speed | Accuracy (Iris) |
|------------|--------------------:|---------------:|----------------:|
| Grid       | 216                 | 1.0x           | 97.8%           |
| Coordinate | 30-84               | 3.0x           | 97.8%           |
| Hybrid     | ~120                | 2.3x           | 97.8%           |

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

### Training Time
- FGRT: O(n log n) complexity, similar to CART
- Substantially faster than evolutionary approaches
- Partition optimization: 0.1-0.5 seconds per feature (hybrid strategy)

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

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas for contribution:
- Additional optimization strategies
- New partition metrics
- Parallel optimization implementations
- Additional benchmark datasets

## Citation

If you use this code in your research, please cite:

```bibtex
@article{fumanal2025fgrt,
  title={Fuzzy Greedy Rule Trees with Optimized Partitions},
  author={Fumanal-Idocin, Javier and others},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.
