"""
Ablation Studies and Parameter Sensitivity Analysis for FuzzyCART
==================================================================

This module implements ablation studies to analyze the impact of:
1. Different hyperparameters (max_rules, coverage_threshold, min_improvement)
2. Different splitting metrics (Gini vs. Entropy)
3. Pruning strategies (CCP alpha values)
4. Number of linguistic terms
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
import sys
sys.path.append('/mnt/project')
from tree_learning import FuzzyCART
try:
    from ex_fuzzy import fuzzy_sets as fs
    from ex_fuzzy import utils
except ImportError:
    import fuzzy_sets as fs


class AblationStudy:
    """
    Conducts ablation studies to understand the impact of different components.
    """
    
    def __init__(self, X, y, fuzzy_partitions, n_folds=5, random_state=42):
        """
        Initialize ablation study.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        fuzzy_partitions : list
            Fuzzy partitions for features
        n_folds : int
            Number of CV folds
        random_state : int
            Random seed
        """
        self.X = X
        self.y = y
        self.fuzzy_partitions = fuzzy_partitions
        self.n_folds = n_folds
        self.random_state = random_state
        
    def test_max_rules_impact(self, max_rules_values=None):
        """
        Test impact of maximum number of rules.
        
        Parameters
        ----------
        max_rules_values : list, optional
            List of max_rules values to test
            
        Returns
        -------
        pd.DataFrame
            Results for different max_rules values
        """
        if max_rules_values is None:
            max_rules_values = [5, 10, 15, 20, 25, 30]
        
        print("\n" + "="*60)
        print("Testing Impact of max_rules Parameter")
        print("="*60)
        
        results = []
        
        for max_rules in max_rules_values:
            print(f"\nTesting max_rules={max_rules}")
            
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                 random_state=self.random_state)
            
            fold_accuracies = []
            fold_n_rules = []
            fold_complexities = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y)):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                
                try:
                    fuzzy_cart = FuzzyCART(
                        fuzzy_partitions=self.fuzzy_partitions,
                        max_rules=max_rules,
                        coverage_threshold=0.01,
                        min_improvement=0.01
                    )
                    
                    fuzzy_cart.fit(X_train, y_train)
                    y_pred = fuzzy_cart.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    stats = fuzzy_cart.get_tree_stats() if hasattr(fuzzy_cart, 'get_tree_stats') else {}
                    
                    fold_accuracies.append(accuracy)
                    fold_n_rules.append(stats.get('leaves', 0))
                    fold_complexities.append(stats.get('total_conditions', 0))
                    
                except Exception as e:
                    print(f"  Fold {fold_idx} failed: {e}")
                    fold_accuracies.append(0.0)
                    fold_n_rules.append(0)
                    fold_complexities.append(0)
            
            results.append({
                'max_rules': max_rules,
                'accuracy_mean': np.mean(fold_accuracies),
                'accuracy_std': np.std(fold_accuracies),
                'n_rules_mean': np.mean(fold_n_rules),
                'complexity_mean': np.mean(fold_complexities)
            })
            
            print(f"  Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
            print(f"  Avg Rules: {np.mean(fold_n_rules):.2f}")
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def test_coverage_threshold_impact(self, threshold_values=None):
        """
        Test impact of coverage threshold parameter.
        
        Parameters
        ----------
        threshold_values : list, optional
            List of coverage threshold values to test
            
        Returns
        -------
        pd.DataFrame
            Results for different threshold values
        """
        if threshold_values is None:
            threshold_values = [0.0, 0.01, 0.03, 0.05, 0.1]
        
        print("\n" + "="*60)
        print("Testing Impact of coverage_threshold Parameter")
        print("="*60)
        
        results = []
        
        for threshold in threshold_values:
            print(f"\nTesting coverage_threshold={threshold}")
            
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                 random_state=self.random_state)
            
            fold_accuracies = []
            fold_n_rules = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y)):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                
                try:
                    fuzzy_cart = FuzzyCART(
                        fuzzy_partitions=self.fuzzy_partitions,
                        max_rules=15,
                        coverage_threshold=threshold,
                        min_improvement=0.01
                    )
                    
                    fuzzy_cart.fit(X_train, y_train)
                    y_pred = fuzzy_cart.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    stats = fuzzy_cart.get_tree_stats() if hasattr(fuzzy_cart, 'get_tree_stats') else {}
                    
                    fold_accuracies.append(accuracy)
                    fold_n_rules.append(stats.get('leaves', 0))
                    
                except Exception as e:
                    print(f"  Fold {fold_idx} failed: {e}")
                    fold_accuracies.append(0.0)
                    fold_n_rules.append(0)
            
            results.append({
                'coverage_threshold': threshold,
                'accuracy_mean': np.mean(fold_accuracies),
                'accuracy_std': np.std(fold_accuracies),
                'n_rules_mean': np.mean(fold_n_rules)
            })
            
            print(f"  Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def test_min_improvement_impact(self, improvement_values=None):
        """
        Test impact of minimum improvement parameter.
        
        Parameters
        ----------
        improvement_values : list, optional
            List of min_improvement values to test
            
        Returns
        -------
        pd.DataFrame
            Results for different min_improvement values
        """
        if improvement_values is None:
            improvement_values = [0.0, 0.005, 0.01, 0.02, 0.05]
        
        print("\n" + "="*60)
        print("Testing Impact of min_improvement Parameter")
        print("="*60)
        
        results = []
        
        for min_improvement in improvement_values:
            print(f"\nTesting min_improvement={min_improvement}")
            
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                 random_state=self.random_state)
            
            fold_accuracies = []
            fold_n_rules = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y)):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                
                try:
                    fuzzy_cart = FuzzyCART(
                        fuzzy_partitions=self.fuzzy_partitions,
                        max_rules=15,
                        coverage_threshold=0.01,
                        min_improvement=min_improvement
                    )
                    
                    fuzzy_cart.fit(X_train, y_train)
                    y_pred = fuzzy_cart.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    stats = fuzzy_cart.get_tree_stats() if hasattr(fuzzy_cart, 'get_tree_stats') else {}
                    
                    fold_accuracies.append(accuracy)
                    fold_n_rules.append(stats.get('leaves', 0))
                    
                except Exception as e:
                    print(f"  Fold {fold_idx} failed: {e}")
                    fold_accuracies.append(0.0)
                    fold_n_rules.append(0)
            
            results.append({
                'min_improvement': min_improvement,
                'accuracy_mean': np.mean(fold_accuracies),
                'accuracy_std': np.std(fold_accuracies),
                'n_rules_mean': np.mean(fold_n_rules)
            })
            
            print(f"  Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def test_pruning_impact(self, ccp_alpha_values=None):
        """
        Test impact of pruning (CCP alpha parameter).
        
        Parameters
        ----------
        ccp_alpha_values : list, optional
            List of ccp_alpha values to test
            
        Returns
        -------
        pd.DataFrame
            Results for different ccp_alpha values
        """
        if ccp_alpha_values is None:
            ccp_alpha_values = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
        
        print("\n" + "="*60)
        print("Testing Impact of Pruning (ccp_alpha)")
        print("="*60)
        
        results = []
        
        for ccp_alpha in ccp_alpha_values:
            print(f"\nTesting ccp_alpha={ccp_alpha}")
            
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                 random_state=self.random_state)
            
            fold_accuracies = []
            fold_n_rules = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y)):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                
                try:
                    fuzzy_cart = FuzzyCART(
                        fuzzy_partitions=self.fuzzy_partitions,
                        max_rules=20,
                        coverage_threshold=0.01,
                        min_improvement=0.01,
                        ccp_alpha=ccp_alpha
                    )
                    
                    fuzzy_cart.fit(X_train, y_train)
                    y_pred = fuzzy_cart.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    stats = fuzzy_cart.get_tree_stats() if hasattr(fuzzy_cart, 'get_tree_stats') else {}
                    
                    fold_accuracies.append(accuracy)
                    fold_n_rules.append(stats.get('leaves', 0))
                    
                except Exception as e:
                    print(f"  Fold {fold_idx} failed: {e}")
                    fold_accuracies.append(0.0)
                    fold_n_rules.append(0)
            
            results.append({
                'ccp_alpha': ccp_alpha,
                'accuracy_mean': np.mean(fold_accuracies),
                'accuracy_std': np.std(fold_accuracies),
                'n_rules_mean': np.mean(fold_n_rules)
            })
            
            print(f"  Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
            print(f"  Avg Rules: {np.mean(fold_n_rules):.2f}")
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def run_all_ablation_studies(self):
        """
        Run all ablation studies and save results.
        
        Returns
        -------
        dict
            Dictionary containing all ablation study results
        """
        print("\n" + "="*80)
        print("Running Complete Ablation Study Suite")
        print("="*80)
        
        results = {}
        
        # Test max_rules
        results['max_rules'] = self.test_max_rules_impact()
        
        # Test coverage_threshold
        results['coverage_threshold'] = self.test_coverage_threshold_impact()
        
        # Test min_improvement
        results['min_improvement'] = self.test_min_improvement_impact()
        
        # Test pruning
        results['pruning'] = self.test_pruning_impact()
        
        return results


class ParameterOptimizer:
    """
    Performs grid search to find optimal hyperparameters.
    """
    
    def __init__(self, X, y, fuzzy_partitions, n_folds=5, random_state=42):
        """
        Initialize parameter optimizer.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        fuzzy_partitions : list
            Fuzzy partitions for features
        n_folds : int
            Number of CV folds
        random_state : int
            Random seed
        """
        self.X = X
        self.y = y
        self.fuzzy_partitions = fuzzy_partitions
        self.n_folds = n_folds
        self.random_state = random_state
    
    def grid_search(self, param_grid=None):
        """
        Perform grid search over hyperparameter space.
        
        Parameters
        ----------
        param_grid : dict, optional
            Dictionary of parameters to search
            
        Returns
        -------
        dict
            Best parameters and corresponding results
        """
        if param_grid is None:
            param_grid = {
                'max_rules': [10, 15, 20],
                'coverage_threshold': [0.01, 0.03, 0.05],
                'min_improvement': [0.005, 0.01, 0.02]
            }
        
        print("\n" + "="*60)
        print("Grid Search for Optimal Hyperparameters")
        print("="*60)
        print(f"Parameter grid: {param_grid}")
        
        best_score = 0
        best_params = None
        all_results = []
        
        # Generate all parameter combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        for param_combination in product(*param_values):
            params = dict(zip(param_names, param_combination))
            
            print(f"\nTesting parameters: {params}")
            
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                 random_state=self.random_state)
            
            fold_scores = []
            
            for train_idx, test_idx in skf.split(self.X, self.y):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                
                try:
                    fuzzy_cart = FuzzyCART(
                        fuzzy_partitions=self.fuzzy_partitions,
                        **params
                    )
                    
                    fuzzy_cart.fit(X_train, y_train)
                    y_pred = fuzzy_cart.predict(X_test)
                    score = accuracy_score(y_test, y_pred)
                    fold_scores.append(score)
                    
                except Exception as e:
                    print(f"  Failed: {e}")
                    fold_scores.append(0.0)
            
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            print(f"  Mean accuracy: {mean_score:.4f} ± {std_score:.4f}")
            
            all_results.append({
                **params,
                'mean_score': mean_score,
                'std_score': std_score
            })
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        
        print("\n" + "="*60)
        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score:.4f}")
        print("="*60)
        
        results_df = pd.DataFrame(all_results)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results_df
        }


class VisualizationTools:
    """
    Visualization tools for ablation study results.
    """
    
    @staticmethod
    def plot_parameter_impact(results_df, param_name, metric='accuracy_mean'):
        """
        Plot impact of a parameter on performance.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results from ablation study
        param_name : str
            Name of parameter being varied
        metric : str
            Metric to plot
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(results_df[param_name], results_df[metric], 'o-', linewidth=2)
        
        if 'accuracy_std' in results_df.columns:
            plt.fill_between(
                results_df[param_name],
                results_df[metric] - results_df['accuracy_std'],
                results_df[metric] + results_df['accuracy_std'],
                alpha=0.3
            )
        
        plt.xlabel(param_name.replace('_', ' ').title(), fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.title(f'Impact of {param_name.replace("_", " ").title()} on Performance',
                 fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def plot_accuracy_vs_complexity(results_df):
        """
        Plot accuracy vs model complexity trade-off.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results containing accuracy and complexity metrics
        """
        plt.figure(figsize=(10, 6))
        
        if 'n_rules_mean' in results_df.columns:
            plt.scatter(results_df['n_rules_mean'], results_df['accuracy_mean'],
                       s=100, alpha=0.6)
            plt.xlabel('Number of Rules', fontsize=12)
        elif 'complexity_mean' in results_df.columns:
            plt.scatter(results_df['complexity_mean'], results_df['accuracy_mean'],
                       s=100, alpha=0.6)
            plt.xlabel('Total Conditions', fontsize=12)
        
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Accuracy vs. Model Complexity Trade-off', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()


def main():
    """
    Main function to run ablation studies on all 10 datasets.
    """
    import os
    from fuzzy_cart_experiments import DatasetLoader
    
    print("="*80)
    print("FuzzyCART Ablation Studies - All Datasets")
    print("="*80)
    
    # Initialize dataset loader
    loader = DatasetLoader()
    
    # Define which datasets to test
    datasets_to_test = [
        'iris', 'wine', 'vehicle', 'vowel', 'glass', 
        'ecoli', 'yeast', 'segment', 'pendigits', 'optdigits'
    ]
    
    # Load all datasets
    print("\nLoading datasets...")
    datasets_dict = loader.load_all_datasets(datasets_to_test)
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Store all results across datasets
    all_dataset_results = {
        'max_rules': [],
        'coverage_threshold': [],
        'min_improvement': [],
        'pruning': []
    }
    
    # Run ablation studies for each dataset
    for dataset_name, (X, y, categorical_mask) in datasets_dict.items():
        print("\n" + "="*80)
        print(f"Dataset: {dataset_name}")
        print("="*80)
        print(f"  Samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Classes: {len(np.unique(y))}")
        
        # Generate fuzzy partitions using ex_fuzzy.utils
        fuzzy_partitions = utils.construct_partitions(
            X, 
            n_partitions=3, 
            fz_type_studied=fs.FUZZY_SETS.t1
        )
        
        # Run ablation studies with fewer folds for efficiency
        ablation = AblationStudy(X, y, fuzzy_partitions, n_folds=3)
        ablation_results = ablation.run_all_ablation_studies()
        
        # Save results for this dataset
        for study_name, results_df in ablation_results.items():
            # Add dataset name to results
            results_df['dataset'] = dataset_name
            
            # Append to overall results
            all_dataset_results[study_name].append(results_df)
            
            # Save individual dataset results
            filename = f'results/ablation_{dataset_name}_{study_name}.csv'
            results_df.to_csv(filename, index=False)
            print(f"  Saved {study_name} results to: {filename}")
        
        # Run parameter optimization
        print(f"\n  Running parameter optimization for {dataset_name}...")
        optimizer = ParameterOptimizer(X, y, fuzzy_partitions, n_folds=3)
        optimization_results = optimizer.grid_search()
        
        # Save optimization results
        opt_df = optimization_results['all_results']
        opt_df['dataset'] = dataset_name
        opt_df.to_csv(f'results/optimization_{dataset_name}.csv', index=False)
        
        print(f"  Best params for {dataset_name}: {optimization_results['best_params']}")
        print(f"  Best score: {optimization_results['best_score']:.4f}")
    
    # Combine results across all datasets
    print("\n" + "="*80)
    print("Combining results across all datasets...")
    print("="*80)
    
    for study_name, df_list in all_dataset_results.items():
        if df_list:
            combined_df = pd.concat(df_list, ignore_index=True)
            filename = f'results/ablation_ALL_DATASETS_{study_name}.csv'
            combined_df.to_csv(filename, index=False)
            print(f"Saved combined {study_name} results: {filename}")
    
    # Generate summary visualizations
    print("\nGenerating summary visualizations...")
    viz = VisualizationTools()
    
    for study_name, df_list in all_dataset_results.items():
        if df_list and len(df_list) > 0:
            combined_df = pd.concat(df_list, ignore_index=True)
            
            if len(combined_df) > 1:
                # Find parameter column name
                param_cols = [col for col in combined_df.columns 
                             if col not in ['accuracy_mean', 'accuracy_std', 
                                           'n_rules_mean', 'complexity_mean', 'dataset']]
                
                if param_cols:
                    param_name = param_cols[0]
                    
                    # Create plot with average across datasets
                    grouped = combined_df.groupby(param_name).agg({
                        'accuracy_mean': 'mean',
                        'accuracy_std': 'mean'
                    }).reset_index()
                    
                    fig = viz.plot_parameter_impact(grouped, param_name)
                    fig.savefig(f'results/ablation_SUMMARY_{study_name}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"  Saved summary plot: ablation_SUMMARY_{study_name}.png")
    
    print("\n" + "="*80)
    print("Ablation Studies Complete!")
    print("="*80)
    print(f"Results saved to: results/")
    print(f"  - Individual dataset results: ablation_<dataset>_<study>.csv")
    print(f"  - Combined results: ablation_ALL_DATASETS_<study>.csv")
    print(f"  - Summary plots: ablation_SUMMARY_<study>.png")


if __name__ == '__main__':
    main()
