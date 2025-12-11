"""
Test and Compare Different Search Strategies
=============================================

This script compares grid, coordinate descent, and hybrid search strategies
for fuzzy partition optimization on the Iris dataset.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import time

from tree_learning import FuzzyCART
from partition_optimization import optimize_partitions_for_gfrt
from ex_fuzzy import utils

print("="*80)
print("Comparing Search Strategies for Fuzzy Partition Optimization")
print("="*80)

# Load Iris dataset
X, y = load_iris(return_X_y=True)

print(f"\nDataset: Iris")
print(f"Samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {len(np.unique(y))}")

# Create default partitions for comparison
print("\n" + "="*80)
print("Creating default fuzzy partitions...")
print("="*80)
default_partitions = utils.construct_partitions(X)

# Test each strategy
strategies = ['grid', 'coordinate', 'hybrid']
results = {}

for strategy in strategies:
    print(f"\n{'='*80}")
    print(f"Strategy: {strategy.upper()}")
    print(f"{'='*80}")
    
    # Optimize partitions
    start_time = time.time()
    optimized_partitions = optimize_partitions_for_gfrt(
        X, y,
        initial_partitions=default_partitions,
        method='separability',
        strategy=strategy,
        verbose=True
    )
    optimization_time = time.time() - start_time
    
    print(f"\n⏱️  Optimization time: {optimization_time:.2f}s")
    
    # Evaluate with cross-validation
    print(f"\nRunning 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scores_default = []
    scores_optimized = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Default partitions
        model_default = FuzzyCART(fuzzy_partitions=default_partitions, max_depth=5)
        model_default.fit(X_train, y_train)
        y_pred_default = model_default.predict(X_test)
        scores_default.append(accuracy_score(y_test, y_pred_default))
        
        # Optimized partitions
        model_optimized = FuzzyCART(fuzzy_partitions=optimized_partitions, max_depth=5)
        model_optimized.fit(X_train, y_train)
        y_pred_optimized = model_optimized.predict(X_test)
        scores_optimized.append(accuracy_score(y_test, y_pred_optimized))
    
    scores_default = np.array(scores_default)
    scores_optimized = np.array(scores_optimized)
    
    # Store results
    results[strategy] = {
        'optimization_time': optimization_time,
        'default_accuracy': np.mean(scores_default),
        'default_std': np.std(scores_default),
        'optimized_accuracy': np.mean(scores_optimized),
        'optimized_std': np.std(scores_optimized),
        'improvement': np.mean(scores_optimized) - np.mean(scores_default)
    }
    
    print(f"\n📊 Results for {strategy.upper()}:")
    print(f"   Default:   {results[strategy]['default_accuracy']:.4f} ± {results[strategy]['default_std']:.4f}")
    print(f"   Optimized: {results[strategy]['optimized_accuracy']:.4f} ± {results[strategy]['optimized_std']:.4f}")
    print(f"   Change:    {results[strategy]['improvement']:+.4f} ({results[strategy]['improvement']*100:+.2f}%)")

# Summary comparison
print("\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)
print(f"\n{'Strategy':<15} {'Time (s)':<12} {'Default Acc':<15} {'Optimized Acc':<15} {'Improvement'}")
print("-"*80)

for strategy in strategies:
    r = results[strategy]
    print(f"{strategy.upper():<15} "
          f"{r['optimization_time']:<12.2f} "
          f"{r['default_accuracy']:.4f} ± {r['default_std']:.4f}   "
          f"{r['optimized_accuracy']:.4f} ± {r['optimized_std']:.4f}   "
          f"{r['improvement']:+.4f}")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

# Find best strategy by improvement
best_by_improvement = max(results.items(), key=lambda x: x[1]['improvement'])
print(f"\n✓ Best improvement: {best_by_improvement[0].upper()} "
      f"({best_by_improvement[1]['improvement']:+.4f})")

# Find fastest strategy
fastest = min(results.items(), key=lambda x: x[1]['optimization_time'])
print(f"✓ Fastest: {fastest[0].upper()} "
      f"({fastest[1]['optimization_time']:.2f}s)")

# Find best trade-off (improvement per second)
best_tradeoff = max(results.items(), 
                    key=lambda x: x[1]['improvement'] / x[1]['optimization_time'] 
                    if x[1]['optimization_time'] > 0 else 0)
print(f"✓ Best time/improvement trade-off: {best_tradeoff[0].upper()}")

print("\n" + "="*80)
