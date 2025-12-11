"""
Quick test of partition optimization experiments
"""

from fuzzy_cart_experiments import DatasetLoader, ExperimentRunner
import numpy as np

# Simple quick test with Iris
print("="*80)
print("Quick Test: Partition Optimization Impact")
print("="*80)

# Load a small dataset
from sklearn.datasets import load_iris
from ex_fuzzy import fuzzy_sets as fs, utils

X, y = load_iris(return_X_y=True)
print(f"\nDataset: Iris")
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")

# Create default partitions
categorical_mask = [False] * X.shape[1]
default_partitions = utils.construct_partitions(
    X, 
    fz_type_studied=fs.FUZZY_SETS.t1,
    categorical_mask=categorical_mask,
    n_partitions=3,
    shape='trapezoid'
)

# Optimize partitions
print("\nOptimizing partitions...")
from partition_optimization import optimize_partitions_for_gfrt
optimized_partitions = optimize_partitions_for_gfrt(
    X, y,
    initial_partitions=default_partitions,
    method='separability',
    strategy='grid',
    verbose=True
)

# Run experiments
print("\n" + "="*60)
print("Running Cross-Validation Experiments")
print("="*60)

runner = ExperimentRunner(n_folds=5, random_state=42)
results = runner.run_experiment(X, y, "iris", default_partitions, optimized_partitions)

# Display results
print("\n" + "="*60)
print("Results Summary")
print("="*60)

for method_name, metrics in results.items():
    print(f"\n{method_name}:")
    print(f"  Accuracy: {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")
    print(f"  Rules: {metrics['n_rules_mean']:.2f}")
    print(f"  Conditions: {metrics['n_conditions_mean']:.2f}")
    print(f"  Training Time: {metrics['train_time_mean']:.4f}s")

# Compare FuzzyCART with default vs optimized
if 'FuzzyCART' in results and 'FuzzyCART_Optimized' in results:
    default_acc = results['FuzzyCART']['accuracy_mean']
    optimized_acc = results['FuzzyCART_Optimized']['accuracy_mean']
    improvement = optimized_acc - default_acc
    
    print("\n" + "="*60)
    print("Partition Optimization Impact")
    print("="*60)
    print(f"Default Partitions:   {default_acc:.4f}")
    print(f"Optimized Partitions: {optimized_acc:.4f}")
    print(f"Improvement:          {improvement:+.4f} ({100*improvement/default_acc:+.2f}%)")
    
    if improvement > 0:
        print("✓ Partition optimization improved accuracy!")
    elif improvement < 0:
        print("✗ Partition optimization decreased accuracy")
    else:
        print("= No change in accuracy")

print("\n" + "="*60)
print("Test Complete!")
print("="*60)
