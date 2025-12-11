# Fuzzy Greedy Decision Trees with Optimized Partitions

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive framework for fuzzy decision tree learning with **automatic fuzzy partition optimization**. This repository implements FuzzyCART with a novel interleaved encoding scheme that optimizes trapezoidal membership functions to maximize class separability while guaranteeing validity by construction.

## 🌟 Key Features

- **Novel Partition Optimization**: Automatic tuning of fuzzy membership functions using separability-based metrics
- **Multiple Search Strategies**: Grid search (exhaustive), coordinate descent (3× faster), and hybrid optimization
- **Validity Guarantees**: Interleaved encoding ensures all optimized partitions are valid and interpretable
- **Comprehensive Benchmarking**: Integrated experimental framework for comparing methods across datasets
- **Production-Ready**: Robust handling of categorical features, proper error handling, full documentation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Fuminides/fuzzy_greedy_tree_public.git
cd fuzzy_greedy_tree_public

# Install dependencies
pip install numpy pandas scikit-learn scipy matplotlib ex-fuzzy
pip install scikit-posthocs  # Optional: for statistical tests
```

### Basic Usage

```python
from sklearn.datasets import load_iris
from ex_fuzzy import utils
from tree_learning import FuzzyCART
from partition_optimization import optimize_partitions_for_gfrt

# Load data
X, y = load_iris(return_X_y=True)

# Create default partitions
default_partitions = utils.construct_partitions(X)

# Optimize partitions (optional)
optimized_partitions = optimize_partitions_for_gfrt(
    X, y,
    initial_partitions=default_partitions,
    method='separability',      # 'separability', 'gini', or 'fisher'
    strategy='hybrid',           # 'grid', 'coordinate', or 'hybrid'
    verbose=True
)

# Train classifier
model = FuzzyCART(fuzzy_partitions=optimized_partitions, max_depth=5)
model.fit(X, y)

# Predict
predictions = model.predict(X)
accuracy = (predictions == y).mean()
print(f"Accuracy: {accuracy:.2%}")
```

### Quick Test

```bash
# Run quick validation on Iris dataset (~2 minutes)
python test_optimization_experiment.py

# Compare all search strategies (~5 minutes)
python test_search_strategies.py

# Generate comparison visualization
python visualize_strategies.py
```

## 📊 Performance Comparison

Performance on Iris dataset (4 features, 150 samples):

| Strategy | Time | Evaluations/Feature | Speedup | Best Separability |
|----------|------|---------------------|---------|-------------------|
| **Grid** | 0.09s | 216 fixed | 1.0× | 89.56 |
| **Coordinate** | 0.03s | 30-84 adaptive | **3.0×** | **91.78** |
| **Hybrid** | 0.04s | ~120 | 2.3× | 91.78 |

**Key Result**: Coordinate descent is 3× faster and often finds better solutions than grid search!

## 🔬 Research Highlights

### Novel Interleaved Encoding

Our key innovation is an encoding scheme that guarantees partition validity by construction:

```
12-Parameter Interleaved Order (for 3 fuzzy terms):
Position 1:  Low[a]      Position 7:  Medium[g]
Position 2:  Low[b]      Position 8:  High[i]
Position 3:  Low[c]      Position 9:  Medium[h]
Position 4:  Medium[e]   Position 10: High[j]
Position 5:  Low[d]      Position 11: High[k]
Position 6:  Medium[f]   Position 12: High[l]
```

This interleaving ensures:
- ✅ All trapezoids satisfy a ≤ b ≤ c ≤ d automatically
- ✅ Monotonicity preserved during optimization
- ✅ Interpretable ordering maintained (Low < Medium < High)
- ✅ Full domain coverage guaranteed
- ✅ No constraint violations possible

### Three Search Strategies

1. **Grid Search** - Exhaustive exploration (best for reproducibility)
   - 216 evaluations per feature
   - Deterministic results
   - Baseline for comparison

2. **Coordinate Descent** - Fast adaptive search (best for speed)
   - 30-84 evaluations per feature (adaptive)
   - 3× faster than grid search
   - Often finds better solutions

3. **Hybrid** - Balanced approach (recommended default)
   - Coarse grid (27) + local refinement
   - 2.3× faster than grid search
   - Robust to poor initialization

See [SEARCH_STRATEGIES.md](SEARCH_STRATEGIES.md) for detailed comparison.

## 📖 Documentation

- **[TECHNICAL_REPORT_PARTITION_OPTIMIZATION.md](TECHNICAL_REPORT_PARTITION_OPTIMIZATION.md)** - Complete technical documentation (~1200 lines)
  - Detailed algorithm descriptions with pseudocode
  - Performance benchmarks on Iris
  - Validation results
  - Paper writing guide with suggested figures and tables

- **[SEARCH_STRATEGIES.md](SEARCH_STRATEGIES.md)** - Search strategy comparison
  - Implementation details for all three strategies
  - Usage recommendations
  - Performance analysis and trade-offs

- **[PARTITION_OPTIMIZATION_EXPERIMENTS.md](PARTITION_OPTIMIZATION_EXPERIMENTS.md)** - Experimental guide
  - Running benchmarks on KEEL datasets
  - Dataset selection
  - Output interpretation

- **[partition_optimization_guide.txt](partition_optimization_guide.txt)** - Quick reference

## 🧪 Running Experiments

### Full Benchmark Suite

```bash
# Standard experiments (default partitions only)
python fuzzy_cart_experiments.py

# With optimization comparison (~2-3x longer)
python fuzzy_cart_experiments.py --optimize-partitions
```

### Custom Dataset Selection

Edit `fuzzy_cart_experiments.py` (around line 980):
```python
datasets_to_test = ['iris', 'wine', 'pima', 'hepatitis', 'glass']
```

### Expected Outputs

- `fuzzy_cart_results.csv` - Per-dataset, per-method results
- `fuzzy_cart_summary.csv` - Aggregated statistics
- `partition_optimization_comparison.csv` - Default vs optimized comparison
- `partition_optimization_impact.pdf` - Visualization
- `search_strategy_comparison.png` - Strategy comparison plots

## 🔧 Core Modules

### `partition_optimization.py` (~1200 lines)
Core optimization module with three search strategies:
```python
from partition_optimization import FuzzyPartitionOptimizer

optimizer = FuzzyPartitionOptimizer(
    optimization_method='separability',  # 'separability', 'gini', 'fisher'
    search_strategy='hybrid',            # 'grid', 'coordinate', 'hybrid'
    verbose=True
)

optimized = optimizer.optimize_partitions(X, y, initial_partitions)
```

**Key Methods**:
- `optimize_partitions()` - Main entry point
- `_encode_partitions_direct()` - Interleaved encoding
- `_decode_partitions()` - Decode to valid trapezoids
- `_grid_search_encoded()` - Grid search over parameter space
- `_coordinate_descent_encoded()` - Adaptive local search
- `_hybrid_search_encoded()` - Two-phase optimization
- `_compute_separability_index()` - Primary metric

### `tree_learning.py` (~2600 lines)
FuzzyCART implementation with greedy optimization:
```python
from tree_learning import FuzzyCART

model = FuzzyCART(
    fuzzy_partitions=partitions,
    max_depth=5,           # Maximum tree depth
    max_rules=50,          # Maximum number of rules (leaves)
    coverage_threshold=0.0 # Minimum coverage for valid split
)
```

**Features**:
- Complete Classification Index (CCI) as splitting criterion
- Fuzzy membership-based splits
- Rule extraction and interpretation
- Optimized caching for membership computations

### `fuzzy_cart_experiments.py` (~1200 lines)
Comprehensive experimental framework:
```python
from fuzzy_cart_experiments import ExperimentRunner

runner = ExperimentRunner()
results = runner.run_all_experiments(
    datasets_dict,
    optimize_partitions=True  # Enable optimization comparison
)
```

**Capabilities**:
- 5-fold stratified cross-validation
- Multiple method comparison (FuzzyCART, FuzzyCART_Optimized, CART, C4.5)
- Statistical analysis (Friedman + Nemenyi tests)
- Automatic visualization generation
- Ablation studies for hyperparameters

## 📊 Datasets

Uses KEEL benchmark datasets. The framework automatically handles:
- ✅ Continuous features (optimized)
- ✅ Categorical features (preserved, not optimized)
- ✅ Missing values
- ✅ Imbalanced classes

Popular datasets included:
- **iris** - 150 samples, 4 features, 3 classes (clean, well-separated)
- **wine** - 178 samples, 13 features, 3 classes
- **pima** - 768 samples, 8 features, 2 classes (diabetes prediction)
- **glass** - 214 samples, 9 features, 7 classes
- **hepatitis** - 155 samples, 19 features (6 categorical), 2 classes

Run `python list_datasets.py` to see all available datasets.

## 🎯 When to Use Partition Optimization

**Optimization helps when**:
- Features have non-uniform class distributions
- Default quantile-based partitions are suboptimal
- You need simpler models (fewer rules)
- Interpretability is important
- Class boundaries don't align with quantiles

**Use default partitions when**:
- Data is already well-separated (like Iris)
- Speed is critical and good-enough is sufficient
- Quick baseline is needed
- Features are uniformly distributed

## 📈 Validation Results

Tested on Iris dataset (4 features, 5-fold CV):

**Separability Improvements**:
```
Feature 1: 85.15 → 91.78 (+7.8% with hybrid)
Feature 2: 70.50 → 72.47 (+2.8%)
Feature 3: 125.85 → 130.52 (+3.7%)
Feature 4: 126.47 → 133.89 (+5.9%)
```

**Model Complexity**:
- Default: 17.8 rules, 0.166s training
- Optimized: 10.0 rules, 0.060s training
- **Result**: Simpler, faster models with optimized partitions!

**Note**: On Iris, accuracy slightly decreases (95.33% → 94.67%) because the data is already very well-separated with default partitions. Optimization shows more benefit on complex datasets where default quantiles are suboptimal.

## 🏗️ Repository Structure

```
fuzzy_greedy_tree_public/
├── partition_optimization.py          # Core optimization module
├── tree_learning.py                   # FuzzyCART implementation
├── fuzzy_cart_experiments.py          # Benchmark framework
├── data_loaders.py                    # KEEL dataset utilities
├── test_optimization_experiment.py    # Quick validation
├── test_search_strategies.py          # Strategy comparison
├── visualize_strategies.py            # Generate comparison plots
├── list_datasets.py                   # Dataset enumeration
├── ablation_studies.py                # Hyperparameter analysis
├── performance_analysis.py            # Scalability tests
├── regenerate_figures.py              # Reproduce paper figures
├── partition_integration_example.py   # Usage examples
├── TECHNICAL_REPORT_PARTITION_OPTIMIZATION.md  # Complete docs
├── SEARCH_STRATEGIES.md               # Strategy guide
├── PARTITION_OPTIMIZATION_EXPERIMENTS.md      # Experiment guide
├── partition_optimization_guide.txt   # Quick reference
└── README.md                          # This file
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

**Algorithm Enhancements**:
- Additional optimization metrics (entropy-based, etc.)
- New search strategies (genetic algorithms, simulated annealing, Bayesian optimization)
- Support for regression tasks
- Multi-objective optimization

**Performance**:
- Parallel feature optimization
- GPU acceleration for metric computation
- Caching strategies for repeated experiments

**Usability**:
- More example notebooks
- Integration with scikit-learn pipelines
- Web-based visualization dashboard

Please open an issue or pull request!

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{fuzzy_greedy_tree_optimization,
  author = {Fumanal-Idocin, Javier},
  title = {Fuzzy Greedy Decision Trees with Optimized Partitions},
  year = {2025},
  url = {https://github.com/Fuminides/fuzzy_greedy_tree_public},
  note = {Interleaved encoding for validity-preserving fuzzy partition optimization}
}
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Javier Fumanal-Idocin** - *Original implementation and optimization framework*

## 🙏 Acknowledgments

- Built on the [ex_fuzzy](https://github.com/Fuminides/ex_fuzzy) library
- KEEL dataset repository for benchmark datasets
- Inspired by classical fuzzy decision tree research (FID3, FDT, etc.)

## 📞 Contact

- **Issues**: Please use the [GitHub issue tracker](https://github.com/Fuminides/fuzzy_greedy_tree_public/issues)
- **Pull Requests**: Contributions welcome!

## 📚 Related Work

This implementation is based on research in fuzzy decision trees and greedy rule learning:

- **Fuzzy Decision Trees**: FID3, Fuzzy CART, Fuzzy C4.5
- **Greedy Rule Learning**: RIPPER, PART, JRip
- **Fuzzy Partition Optimization**: Genetic fuzzy systems, neuro-fuzzy approaches

See [TECHNICAL_REPORT_PARTITION_OPTIMIZATION.md](TECHNICAL_REPORT_PARTITION_OPTIMIZATION.md) for detailed related work discussion.

---

**Status**: ✅ Production-ready | 🧪 Well-tested | 📚 Fully documented | 🚀 Ready for research and production

*Last updated: December 2025*
