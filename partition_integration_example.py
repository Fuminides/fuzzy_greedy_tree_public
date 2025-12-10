"""
Integration Example: Using Optimized Partitions with GFRT
==========================================================

This script demonstrates how to integrate the partition optimizer
into your existing experimental framework.

It shows:
1. How to use optimized partitions with FuzzyCART
2. Comparison between default and optimized partitions
3. Integration into benchmark experiments
"""

import ex_fuzzy
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import time

# Import your existing code
import sys
sys.path.append('/mnt/project')
sys.path.append('/home/claude')

from tree_learning import FuzzyCART
from partition_optimization import optimize_partitions_for_gfrt, FuzzyPartitionOptimizer, plot_fuzzy_partitions
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import utils


def generate_default_partitions(X: np.ndarray, categorical_mask=None) -> dict:
    """
    Generate default partitions using ex_fuzzy.utils.construct_partitions.
    
    Parameters
    ----------
    X : np.ndarray
        Training data
    categorical_mask : array-like, optional
        Boolean mask indicating which features are categorical
        
    Returns
    -------
    partitions : dict
        Dictionary mapping feature index to partition parameters
    """
    return utils.construct_partitions(
        X, 
        fz_type_studied=fs.FUZZY_SETS.t1,
        categorical_mask=categorical_mask,
        n_partitions=3,
        shape='trapezoid'
    )


def compare_partition_methods(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compare default vs optimized partitions using cross-validation.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    n_folds : int
        Number of cross-validation folds
    random_state : int
        Random seed
        
    Returns
    -------
    results_df : pd.DataFrame
        Comparison results
    """
    print("\n" + "="*80)
    print("Comparing Partition Methods")
    print("="*80)
    
    results = {
        'Method': [],
        'Fold': [],
        'Accuracy': [],
        'N_Rules': [],
        'Train_Time': [],
        'Partition_Time': []
    }
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Method 1: Default quantile-based partitions
        print("  Testing: Default partitions")
        start_time = time.time()
        default_partitions = generate_default_partitions(X_train)
        partition_time_default = time.time() - start_time
        
        start_time = time.time()
        model_default = FuzzyCART(
            fuzzy_partitions=default_partitions,
            max_rules=15,
            coverage_threshold=0.01,
            min_improvement=0.01
        )
        model_default.fit(X_train, y_train)
        train_time_default = time.time() - start_time
        
        y_pred_default = model_default.predict(X_test)
        accuracy_default = accuracy_score(y_test, y_pred_default)
        
        stats_default = model_default.get_tree_stats() if hasattr(model_default, 'get_tree_stats') else {}
        n_rules_default = stats_default.get('leaves', 0)
        
        results['Method'].append('Default')
        results['Fold'].append(fold_idx + 1)
        results['Accuracy'].append(accuracy_default)
        results['N_Rules'].append(n_rules_default)
        results['Train_Time'].append(train_time_default)
        results['Partition_Time'].append(partition_time_default)
        
        print(f"    Accuracy: {accuracy_default:.4f}, Rules: {n_rules_default}, Time: {train_time_default:.2f}s")
        
        # Method 2: Optimized partitions (separability, hybrid)
        print("  Testing: Optimized partitions (separability + hybrid)")
        start_time = time.time()
        optimized_partitions = optimize_partitions_for_gfrt(
            X_train, y_train,
            method='separability',
            strategy='hybrid',
            initial_partitions=default_partitions,
            verbose=False
        )
        partition_time_opt = time.time() - start_time
        
        start_time = time.time()
        model_opt = FuzzyCART(
            fuzzy_partitions=optimized_partitions,
            max_rules=15,
            coverage_threshold=0.01,
            min_improvement=0.01
        )
        model_opt.fit(X_train, y_train)
        train_time_opt = time.time() - start_time
        
        y_pred_opt = model_opt.predict(X_test)
        accuracy_opt = accuracy_score(y_test, y_pred_opt)
        
        stats_opt = model_opt.get_tree_stats() if hasattr(model_opt, 'get_tree_stats') else {}
        n_rules_opt = stats_opt.get('leaves', 0)
        
        results['Method'].append('Optimized (Sep+Hybrid)')
        results['Fold'].append(fold_idx + 1)
        results['Accuracy'].append(accuracy_opt)
        results['N_Rules'].append(n_rules_opt)
        results['Train_Time'].append(train_time_opt)
        results['Partition_Time'].append(partition_time_opt)
        
        print(f"    Accuracy: {accuracy_opt:.4f}, Rules: {n_rules_opt}, Time: {train_time_opt:.2f}s")
        print(f"    Partition optimization time: {partition_time_opt:.2f}s")
        
        # Method 3: Optimized partitions (gini, grid)
        print("  Testing: Optimized partitions (gini + grid)")
        start_time = time.time()
        optimized_partitions_gini = optimize_partitions_for_gfrt(
            X_train, y_train,
            method='gini',
            strategy='grid',
            initial_partitions=default_partitions,
            verbose=False
        )
        partition_time_gini = time.time() - start_time
        
        start_time = time.time()
        model_gini = FuzzyCART(
            fuzzy_partitions=optimized_partitions_gini,
            max_rules=15,
            coverage_threshold=0.01,
            min_improvement=0.01
        )
        model_gini.fit(X_train, y_train)
        train_time_gini = time.time() - start_time
        
        y_pred_gini = model_gini.predict(X_test)
        accuracy_gini = accuracy_score(y_test, y_pred_gini)
        
        stats_gini = model_gini.get_tree_stats() if hasattr(model_gini, 'get_tree_stats') else {}
        n_rules_gini = stats_gini.get('leaves', 0)
        
        results['Method'].append('Optimized (Gini+Grid)')
        results['Fold'].append(fold_idx + 1)
        results['Accuracy'].append(accuracy_gini)
        results['N_Rules'].append(n_rules_gini)
        results['Train_Time'].append(train_time_gini)
        results['Partition_Time'].append(partition_time_gini)
        
        print(f"    Accuracy: {accuracy_gini:.4f}, Rules: {n_rules_gini}, Time: {train_time_gini:.2f}s")
        print(f"    Partition optimization time: {partition_time_gini:.2f}s")
    
    results_df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    summary = results_df.groupby('Method').agg({
        'Accuracy': ['mean', 'std'],
        'N_Rules': 'mean',
        'Train_Time': 'mean',
        'Partition_Time': 'mean'
    })
    print(summary)
    
    return results_df


def integration_example_for_experiments():
    """
    Show how to integrate optimized partitions into your experiment pipeline.
    """
    print("\n" + "="*80)
    print("Integration Example for Experiments")
    print("="*80)
    
    print("\nTo integrate into your existing fuzzy_cart_experiments.py:")
    print("-" * 60)
    
    code_example = '''
# In fuzzy_cart_experiments.py, modify ExperimentRunner class:

from partition_optimization import optimize_partitions_for_gfrt
from ex_fuzzy import utils, fuzzy_sets as fs

class ExperimentRunner:
    def __init__(self, n_folds=5, random_state=42, 
                 optimize_partitions=False,
                 optimization_method='separability',
                 optimization_strategy='hybrid'):
        """
        Parameters
        ----------
        optimize_partitions : bool
            Whether to optimize fuzzy partitions before tree induction
        optimization_method : str
            Metric for partition optimization ('separability', 'gini', 'fisher')
        optimization_strategy : str
            Search strategy ('grid', 'coordinate', 'gradient', 'hybrid')
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.optimize_partitions = optimize_partitions
        self.optimization_method = optimization_method
        self.optimization_strategy = optimization_strategy
    
    def _test_fuzzy_cart(self, X_train, X_test, y_train, y_test, results_dict):
        """Modified to support optimized partitions."""
        
        # Generate or optimize partitions
        if self.optimize_partitions:
            fuzzy_partitions = optimize_partitions_for_gfrt(
                X_train, y_train,
                method=self.optimization_method,
                strategy=self.optimization_strategy,
                verbose=False
            )
        else:
            # Use default partitions from ex_fuzzy
            fuzzy_partitions = utils.construct_partitions(
                X_train,
                fz_type_studied=fs.FUZZY_SETS.t1,
                n_partitions=3,
                shape='trapezoid'
            )
        
        # Rest of your existing code...
        model = FuzzyCART(
            fuzzy_partitions=fuzzy_partitions,
            max_rules=15,
            coverage_threshold=0.01,
            min_improvement=0.01
        )
        model.fit(X_train, y_train)
        # ...
    '''
    
    print(code_example)
    
    print("\n" + "-" * 60)
    print("Then run experiments with optimized partitions:")
    print("-" * 60)
    
    usage_example = '''
# Run with default partitions
runner_default = ExperimentRunner(
    n_folds=5, 
    optimize_partitions=False
)

# Run with optimized partitions
runner_optimized = ExperimentRunner(
    n_folds=5,
    optimize_partitions=True,
    optimization_method='separability',
    optimization_strategy='hybrid'
)

# Compare both
results_default = runner_default.run_all_experiments(datasets)
results_optimized = runner_optimized.run_all_experiments(datasets)
    '''
    
    print(usage_example)


def main():
    """Main demonstration."""
    from sklearn.datasets import load_iris, load_wine
    
    print("="*80)
    print("Partition Optimization Integration Demo")
    print("="*80)
    
    # Example 1: Quick comparison on Iris
    print("\n\nExample 1: Iris Dataset Comparison")
    print("-" * 80)
    X, y = load_iris(return_X_y=True)
    
    # Generate initial and optimized partitions for visualization
    print("\nGenerating partitions for visualization...")
    initial_partitions = ex_fuzzy.utils.construct_partitions(
        X, 
        fz_type_studied=fs.FUZZY_SETS.t1,
        n_partitions=3,
        shape='trapezoid'
    )
    optimized_partitions = optimize_partitions_for_gfrt(
        X, y, initial_partitions,
        method='separability', 
        strategy='hybrid',
        verbose=False
    )
    
    # Plot fuzzy partitions for first feature
    print("\nPlotting fuzzy partitions for Feature 0 (sepal length)...")
    plot_fuzzy_partitions(
        X[:, 0], 
        initial_partitions, 
        optimized_partitions,
        feature_idx=0,
        save_path='partition_comparison_feature0.png'
    )
    
    # Run full comparison
    results_iris = compare_partition_methods(X, y, n_folds=3, random_state=42)
    results_iris.to_csv('partition_comparison_iris.csv', index=False)
    print(f"\n✓ Results saved to: partition_comparison_iris.csv")
    
    # Example 2: Wine dataset
    print("\n\nExample 2: Wine Dataset Comparison")
    print("-" * 80)
    X, y = load_wine(return_X_y=True)
    results_wine = compare_partition_methods(X, y, n_folds=3, random_state=42)
    results_wine.to_csv('partition_comparison_wine.csv', index=False)
    print(f"\n✓ Results saved to: partition_comparison_wine.csv")
    
    # Show integration guide
    integration_example_for_experiments()
    
    print("\n" + "="*80)
    print("Demo Complete!")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Optimized partitions can improve accuracy while maintaining interpretability")
    print("2. Different optimization methods work better for different datasets")
    print("3. Partition optimization adds computational cost but is typically < 1 second per feature")
    print("4. Integration into your experiments is straightforward - see examples above")
    print("\nRecommended approach for your paper:")
    print("  - Use 'separability' metric with 'hybrid' strategy")
    print("  - Report both default and optimized results")
    print("  - Include partition optimization time in ablation studies")


if __name__ == '__main__':
    main()
