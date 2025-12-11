# Quick Start Guide

Get up and running with Fuzzy Greedy Decision Trees in 5 minutes!

## Installation (2 minutes)

```bash
git clone https://github.com/Fuminides/fuzzy_greedy_tree_public.git
cd fuzzy_greedy_tree_public
pip install numpy pandas scikit-learn scipy matplotlib ex-fuzzy
```

## Basic Example (3 minutes)

### Without Optimization (Default)

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ex_fuzzy import utils
from tree_learning import FuzzyCART

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create default fuzzy partitions
partitions = utils.construct_partitions(X_train)

# Train classifier
model = FuzzyCART(fuzzy_partitions=partitions, max_depth=5)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy:.2%}")
```

### With Optimization (Recommended)

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ex_fuzzy import utils
from tree_learning import FuzzyCART
from partition_optimization import optimize_partitions_for_gfrt

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create default partitions
default_partitions = utils.construct_partitions(X_train)

# Optimize partitions (takes ~0.1s for Iris)
optimized_partitions = optimize_partitions_for_gfrt(
    X_train, y_train,
    initial_partitions=default_partitions,
    strategy='hybrid',  # Fast and effective
    verbose=True
)

# Train with optimized partitions
model = FuzzyCART(fuzzy_partitions=optimized_partitions, max_depth=5)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy:.2%}")
print(f"Number of rules: {model.tree_rules}")
```

## Quick Tests

### Test 1: Validation (~2 minutes)

```bash
python test_optimization_experiment.py
```

Expected output:
- Optimization progress for 4 features
- Separability improvements
- 5-fold cross-validation results
- Default vs optimized comparison

### Test 2: Strategy Comparison (~5 minutes)

```bash
python test_search_strategies.py
```

Compares grid, coordinate, and hybrid strategies with:
- Optimization time
- Separability scores
- Classification accuracy
- Summary recommendations

### Test 3: Visualization

```bash
python visualize_strategies.py
```

Generates `search_strategy_comparison.png` with 8 panels showing comprehensive performance comparison.

## Choose Your Search Strategy

| Strategy | When to Use | Speed | Quality |
|----------|-------------|-------|---------|
| **`'grid'`** | Reproducible research, baseline | 1.0× | Best in grid |
| **`'coordinate'`** | Fast prototyping, large datasets | **3.0×** | Often better |
| **`'hybrid'`** | Production default | 2.3× | Balanced |

Example:

```python
# For speed
optimized = optimize_partitions_for_gfrt(X, y, partitions, strategy='coordinate')

# For thoroughness
optimized = optimize_partitions_for_gfrt(X, y, partitions, strategy='grid')

# For production (recommended)
optimized = optimize_partitions_for_gfrt(X, y, partitions, strategy='hybrid')
```

## Running Benchmarks

### Quick benchmark (10-15 minutes)

Edit `fuzzy_cart_experiments.py` line ~980:
```python
datasets_to_test = ['iris', 'wine', 'glass']
```

Then run:
```bash
python fuzzy_cart_experiments.py --optimize-partitions
```

### Full benchmark (1-2 hours for 20 datasets)

```bash
python fuzzy_cart_experiments.py --optimize-partitions
```

### Output Files

- `fuzzy_cart_results.csv` - All results
- `partition_optimization_comparison.csv` - Default vs optimized
- `partition_optimization_impact.pdf` - Visualization

## Next Steps

1. **Read the docs**: [TECHNICAL_REPORT_PARTITION_OPTIMIZATION.md](TECHNICAL_REPORT_PARTITION_OPTIMIZATION.md)
2. **Try your data**: Replace `load_iris()` with your dataset
3. **Tune parameters**: Adjust `max_depth`, `max_rules`, etc.
4. **Experiment**: Try different optimization strategies and metrics

## Common Issues

### Issue: "Module ex_fuzzy not found"

```bash
pip install ex-fuzzy
# or install from source if needed
```

### Issue: "No module named 'scikit-posthocs'"

```bash
pip install scikit-posthocs  # Optional, only for statistical tests
```

### Issue: Optimization is slow

Use `strategy='coordinate'` for 3× speedup:
```python
optimized = optimize_partitions_for_gfrt(X, y, partitions, strategy='coordinate')
```

### Issue: Categorical features causing errors

The framework automatically detects and skips categorical features. If you see errors, ensure your dataset loader provides a `categorical_mask`.

## Questions?

- **Documentation**: See [README.md](README.md) and [TECHNICAL_REPORT_PARTITION_OPTIMIZATION.md](TECHNICAL_REPORT_PARTITION_OPTIMIZATION.md)
- **Issues**: Open an issue on GitHub
- **Examples**: Check `partition_integration_example.py`

## Summary

```python
# Minimal working example
from sklearn.datasets import load_iris
from ex_fuzzy import utils
from tree_learning import FuzzyCART
from partition_optimization import optimize_partitions_for_gfrt

X, y = load_iris(return_X_y=True)
partitions = utils.construct_partitions(X)
optimized = optimize_partitions_for_gfrt(X, y, partitions, strategy='hybrid')
model = FuzzyCART(fuzzy_partitions=optimized, max_depth=5)
model.fit(X, y)
print(f"Accuracy: {(model.predict(X) == y).mean():.2%}")
```

That's it! You're ready to use optimized fuzzy decision trees. 🎉
