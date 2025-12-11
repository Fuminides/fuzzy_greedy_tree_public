#!/usr/bin/env python
"""
Minimal Example: Fuzzy Decision Trees with Optimized Partitions
=================================================================

This is the simplest possible example showing how to use the
partition optimization framework.

Run from repository root: python examples/minimal_example.py
"""

import sys
import os
# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from ex_fuzzy import utils
from tree_learning import FuzzyCART
from partition_optimization import optimize_partitions_for_gfrt

print("="*70)
print("Fuzzy Decision Trees with Optimized Partitions - Minimal Example")
print("="*70)

# 1. Load data
print("\n[1/5] Loading Iris dataset...")
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"  Features: {X_train.shape[1]}")
print(f"  Classes: {len(set(y))}")

# 2. Create default partitions
print("\n[2/5] Creating default fuzzy partitions...")
default_partitions = utils.construct_partitions(X_train)
print(f"  Created {len(default_partitions)} fuzzy variables")

# 3. Train with default partitions
print("\n[3/5] Training FuzzyCART with default partitions...")
model_default = FuzzyCART(fuzzy_partitions=default_partitions, max_depth=5)
model_default.fit(X_train, y_train)
y_pred_default = model_default.predict(X_test)
acc_default = accuracy_score(y_test, y_pred_default)
print(f"  Accuracy: {acc_default:.2%}")
print(f"  Number of rules: {model_default.tree_rules}")

# 4. Optimize partitions
print("\n[4/5] Optimizing fuzzy partitions...")
print("  Strategy: hybrid (coarse grid + local refinement)")
optimized_partitions = optimize_partitions_for_gfrt(
    X_train, y_train,
    initial_partitions=default_partitions,
    method='separability',
    strategy='hybrid',
    verbose=False  # Set to True to see optimization details
)
print("  ✓ Optimization complete!")

# 5. Train with optimized partitions
print("\n[5/5] Training FuzzyCART with optimized partitions...")
model_optimized = FuzzyCART(fuzzy_partitions=optimized_partitions, max_depth=5)
model_optimized.fit(X_train, y_train)
y_pred_optimized = model_optimized.predict(X_test)
acc_optimized = accuracy_score(y_test, y_pred_optimized)
print(f"  Accuracy: {acc_optimized:.2%}")
print(f"  Number of rules: {model_optimized.tree_rules}")

# Summary
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"Default Partitions:")
print(f"  Accuracy:  {acc_default:.2%}")
print(f"  Rules:     {model_default.tree_rules}")
print(f"\nOptimized Partitions:")
print(f"  Accuracy:  {acc_optimized:.2%}")
print(f"  Rules:     {model_optimized.tree_rules}")
print(f"\nDifference:")
print(f"  Accuracy:  {acc_optimized - acc_default:+.2%}")
print(f"  Rules:     {model_optimized.tree_rules - model_default.tree_rules:+d}")

# Detailed classification report
print("\n" + "="*70)
print("CLASSIFICATION REPORT (Optimized)")
print("="*70)
print(classification_report(y_test, y_pred_optimized, 
                          target_names=['Setosa', 'Versicolor', 'Virginica']))

print("\n✓ Example completed successfully!")
print("\nNext steps:")
print("  - Try with your own dataset")
print("  - Experiment with different strategies: 'grid', 'coordinate', 'hybrid'")
print("  - Adjust max_depth and other FuzzyCART parameters")
print("  - See QUICKSTART.md for more examples")
