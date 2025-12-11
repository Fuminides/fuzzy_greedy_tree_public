#!/usr/bin/env python
"""
Strategy Comparison Example
============================

Compares all three search strategies (grid, coordinate, hybrid)
on a single dataset with detailed output.

Run from repository root: python examples/strategy_comparison_example.py
"""

import sys
import os
# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import time
import numpy as np

from ex_fuzzy import utils
from tree_learning import FuzzyCART
from partition_optimization import optimize_partitions_for_gfrt

print("="*80)
print("Strategy Comparison Example: Grid vs Coordinate vs Hybrid")
print("="*80)

# Load data
print("\nLoading Wine dataset...")
X, y = load_wine(return_X_y=True)
print(f"  Samples: {len(X)}")
print(f"  Features: {X.shape[1]}")
print(f"  Classes: {len(set(y))}")

# Create default partitions
default_partitions = utils.construct_partitions(X)

# Test each strategy
strategies = {
    'grid': 'Grid Search (exhaustive)',
    'coordinate': 'Coordinate Descent (fast)',
    'hybrid': 'Hybrid (balanced)'
}

results = {}

for strategy_name, strategy_desc in strategies.items():
    print(f"\n{'='*80}")
    print(f"Testing Strategy: {strategy_desc}")
    print(f"{'='*80}")
    
    # Optimize
    print(f"\nOptimizing partitions with '{strategy_name}' strategy...")
    start_time = time.time()
    optimized_partitions = optimize_partitions_for_gfrt(
        X, y,
        initial_partitions=default_partitions,
        method='separability',
        strategy=strategy_name,
        verbose=True  # Show optimization progress
    )
    opt_time = time.time() - start_time
    print(f"Optimization time: {opt_time:.3f}s")
    
    # 5-fold cross-validation
    print(f"\nRunning 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scores_default = []
    scores_optimized = []
    rules_default = []
    rules_optimized = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Default
        model_def = FuzzyCART(fuzzy_partitions=default_partitions, max_depth=5)
        model_def.fit(X_train, y_train)
        scores_default.append(accuracy_score(y_test, model_def.predict(X_test)))
        rules_default.append(model_def.tree_rules)
        
        # Optimized
        model_opt = FuzzyCART(fuzzy_partitions=optimized_partitions, max_depth=5)
        model_opt.fit(X_train, y_train)
        scores_optimized.append(accuracy_score(y_test, model_opt.predict(X_test)))
        rules_optimized.append(model_opt.tree_rules)
        
        print(f"  Fold {fold}: Default={scores_default[-1]:.3f}, "
              f"Optimized={scores_optimized[-1]:.3f}")
    
    # Store results
    results[strategy_name] = {
        'opt_time': opt_time,
        'acc_default': np.array(scores_default),
        'acc_optimized': np.array(scores_optimized),
        'rules_default': np.array(rules_default),
        'rules_optimized': np.array(rules_optimized)
    }
    
    print(f"\nResults for {strategy_name}:")
    print(f"  Default:   {np.mean(scores_default):.4f} ± {np.std(scores_default):.4f}")
    print(f"  Optimized: {np.mean(scores_optimized):.4f} ± {np.std(scores_optimized):.4f}")
    print(f"  Improvement: {np.mean(scores_optimized) - np.mean(scores_default):+.4f}")
    print(f"  Rules (default): {np.mean(rules_default):.1f}")
    print(f"  Rules (optimized): {np.mean(rules_optimized):.1f}")

# Final comparison
print(f"\n{'='*80}")
print("FINAL COMPARISON")
print(f"{'='*80}")

print(f"\n{'Strategy':<15} {'Time (s)':<12} {'Default Acc':<15} {'Optimized Acc':<15} {'Improvement'}")
print("-"*80)

for strategy_name in ['grid', 'coordinate', 'hybrid']:
    r = results[strategy_name]
    print(f"{strategy_name.upper():<15} "
          f"{r['opt_time']:<12.3f} "
          f"{np.mean(r['acc_default']):.4f} ± {np.std(r['acc_default']):.4f}  "
          f"{np.mean(r['acc_optimized']):.4f} ± {np.std(r['acc_optimized']):.4f}  "
          f"{np.mean(r['acc_optimized']) - np.mean(r['acc_default']):+.4f}")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

# Find best by improvement
best_acc = max(results.items(), 
               key=lambda x: np.mean(x[1]['acc_optimized']))
print(f"\n✓ Best accuracy: {best_acc[0].upper()}")
print(f"  {np.mean(best_acc[1]['acc_optimized']):.4f} ± {np.std(best_acc[1]['acc_optimized']):.4f}")

# Find fastest
fastest = min(results.items(), key=lambda x: x[1]['opt_time'])
print(f"\n✓ Fastest: {fastest[0].upper()}")
print(f"  {fastest[1]['opt_time']:.3f}s (optimization time)")

# Find best improvement
best_imp = max(results.items(),
               key=lambda x: np.mean(x[1]['acc_optimized']) - np.mean(x[1]['acc_default']))
print(f"\n✓ Best improvement: {best_imp[0].upper()}")
improvement = np.mean(best_imp[1]['acc_optimized']) - np.mean(best_imp[1]['acc_default'])
print(f"  {improvement:+.4f} ({improvement*100:+.2f}%)")

print("\n" + "="*80)
print("CONCLUSIONS")
print("="*80)
print("""
- GRID: Most thorough, deterministic, but slowest
- COORDINATE: Fastest, adaptive, good for large datasets
- HYBRID: Best balance, recommended default for production

For this dataset:
""")
if fastest[0] == best_imp[0]:
    print(f"→ {fastest[0].upper()} is both fastest AND best - clear winner!")
else:
    print(f"→ Trade-off between speed ({fastest[0].upper()}) and quality ({best_imp[0].upper()})")

print(f"\nOptimization overhead: {results['grid']['opt_time']:.3f}s (grid) "
      f"to {results['coordinate']['opt_time']:.3f}s (coordinate)")
print(f"Speed-up: {results['grid']['opt_time'] / results['coordinate']['opt_time']:.1f}× "
      f"(coordinate vs grid)")

print("\n✓ Example completed successfully!")
