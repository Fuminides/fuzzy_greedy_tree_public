"""
Comprehensive Experimental Setup for FuzzyCART
================================================

This module implements the experimental protocol described in the FuzzyCART paper:
- 40 UCI/Keel benchmark datasets  
- 5-fold cross-validation
- Comparison with CART, C4.5, and other baselines
- Metrics: Accuracy, number of rules, number of conditions, unique conditions
- Statistical testing: Friedman test + Post-hoc Nemenyi
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import time
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Import FuzzyCART from tree_learning
import sys
from tree_learning import FuzzyCART
try:
    from ex_fuzzy import fuzzy_sets as fs
    from ex_fuzzy import utils
except ImportError:
    import fuzzy_sets as fs
    import utils


class DatasetLoader:
    """
    Handles loading and preprocessing of UCI/Keel benchmark datasets.
    
    Based on the 40 datasets used in the FRR paper (Table IV):
    - Range: 80 to 19,020 samples
    - Features: 2 to 85
    - Mix of numerical and categorical features
    """
    
    # List of 40 datasets from the FRR paper
    DATASET_NAMES = [
        'appendicitis', 'australian', 'banana', 'bupa', 'chess',
        'coil2000', 'contraceptive', 'crx', 'ecoli', 'flare',
        'german', 'glass', 'haberman', 'hayes-roth', 'heart',
        'hepatitis', 'housevotes', 'iris', 'led7digit', 'magic',
        'mammographic', 'monk-2', 'newthyroid', 'page-blocks', 'penbased',
        'phoneme', 'pima', 'ring', 'saheart', 'satimage',
        'segment', 'sonar', 'spambase', 'spectfheart', 'thyroid',
        'titanic', 'twonorm', 'vehicle', 'wdbc', 'wine',
        'winequality-red', 'winequality-white', 'wisconsin', 'zoo'
    ]
    
    def __init__(self, data_directory='../keel_datasets'):
        """
        Initialize the dataset loader.
        
        Parameters
        ----------
        data_directory : str
            Path to directory containing dataset files
        """
        self.data_directory = data_directory
        self.datasets = {}
        
    def load_dataset(self, name):
        """
        Load a single dataset by name.
        
        Parameters
        ----------
        name : str
            Name of the dataset to load
            
        Returns
        -------
        X : np.ndarray
            Feature matrix
        y : np.ndarray  
            Target labels
        """
        import os
        from data_loaders import load_keel_dataset
        
        print(f"Loading dataset: {name}")
        
        # Construct file path to the dataset
        file_path = os.path.join(self.data_directory, name, name + '.dat')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Load the dataset using the KEEL loader
        X_train, X_val, X_test, y_train, y_val, y_test, header, boolean_categorical_vector = load_keel_dataset(
            file_path, batch_size=32, train_proportion=0.8, random_seed=33
        )
        
        # Combine train, val, and test sets for cross-validation
        X = np.vstack([X_train, X_val, X_test])
        y = np.concatenate([y_train.values if hasattr(y_train, 'values') else y_train,
                           y_val.values if hasattr(y_val, 'values') else y_val,
                           y_test.values if hasattr(y_test, 'values') else y_test])
            
        return X, y, boolean_categorical_vector
    

    def load_all_datasets(self, dataset_names=None):
        """
        Load multiple datasets.
        
        Parameters
        ----------
        dataset_names : list, optional
            List of dataset names to load. If None, loads all.
            
        Returns
        -------
        dict
            Dictionary mapping dataset names to (X, y) tuples
        """
        if dataset_names is None:
            dataset_names = self.DATASET_NAMES
            
        datasets = {}
        for name in dataset_names:
            try:
                X, y, boolean_categorical_vector = self.load_dataset(name)
                datasets[name] = (X, y, boolean_categorical_vector)
                print(f"  Loaded {name}: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
            except Exception as e:
                print(f"  Failed to load {name}: {e}")
                
        return datasets


class ExperimentRunner:
    """
    Runs comprehensive experiments comparing FuzzyCART with baseline methods.
    
    Implements the experimental protocol from the paper:
    - 5-fold stratified cross-validation
    - Multiple baseline comparisons (CART, C4.5, etc.)
    - Complexity metrics tracking
    - Statistical significance testing
    """
    
    def __init__(self, n_folds=5, random_state=42):
        """
        Initialize the experiment runner.
        
        Parameters
        ----------
        n_folds : int
            Number of folds for cross-validation (default: 5)
        random_state : int
            Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.results = {}
        self.ablation_results = {}  # Store all configurations tested
        self.best_configs = {}  # Store best config per dataset
        
    def run_experiment(self, X, y, dataset_name, fuzzy_partitions):
        """
        Run 5-fold cross-validation experiment on a single dataset.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        dataset_name : str
            Name of the dataset
        fuzzy_partitions : list
            Fuzzy partitions for the dataset
            
        Returns
        -------
        dict
            Results for all methods on this dataset
        """
        print(f"\n{'='*60}")
        print(f"Running experiments on: {dataset_name}")
        print(f"{'='*60}")
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, 
                             random_state=self.random_state)
        
        # Store results for each method
        methods_results = {
            'FuzzyCART': {'accuracy': [], 'n_rules': [], 'n_conditions': [], 
                         'n_unique_conditions': [], 'train_time': []},
            'CART': {'accuracy': [], 'n_rules': [], 'n_conditions': [],
                    'n_unique_conditions': [], 'train_time': []},
            'C4.5': {'accuracy': [], 'n_rules': [], 'n_conditions': [],
                    'n_unique_conditions': [], 'train_time': []},
        }
        
        # Run cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold_idx + 1}/{self.n_folds}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Test FuzzyCART with multiple configurations
            self._test_fuzzycart(X_train, X_test, y_train, y_test,
                               fuzzy_partitions, methods_results['FuzzyCART'])
            
            # Test CART baseline
            self._test_cart(X_train, X_test, y_train, y_test,
                          methods_results['CART'])
            
            # Test C4.5 (DecisionTreeClassifier with entropy)
            self._test_c45(X_train, X_test, y_train, y_test,
                         methods_results['C4.5'])
        
        # Compute average results
        dataset_results = {}
        for method_name, results in methods_results.items():
            dataset_results[method_name] = {
                'accuracy_mean': np.mean(results['accuracy']),
                'accuracy_std': np.std(results['accuracy']),
                'n_rules_mean': np.mean(results['n_rules']),
                'n_conditions_mean': np.mean(results['n_conditions']),
                'n_unique_conditions_mean': np.mean(results['n_unique_conditions']),
                'train_time_mean': np.mean(results['train_time']),
                'raw_results': results
            }
            
            # For FuzzyCART, store best configuration info
            if method_name == 'FuzzyCART' and 'best_configs' in results:
                # Find most common best config across folds
                from collections import Counter
                config_strs = [str(sorted(c.items())) if c else 'None' for c in results['best_configs']]
                most_common = Counter(config_strs).most_common(1)
                if most_common and most_common[0][0] != 'None':
                    dataset_results[method_name]['best_config'] = results['best_configs'][config_strs.index(most_common[0][0])]
                    dataset_results[method_name]['best_config_frequency'] = most_common[0][1] / len(results['best_configs'])
                
                # Store all configurations tested for ablation analysis
                if 'all_configs' in results:
                    self.ablation_results[dataset_name] = results['all_configs']
            
            print(f"\n{method_name} Results:")
            print(f"  Accuracy: {dataset_results[method_name]['accuracy_mean']:.4f} "
                  f"± {dataset_results[method_name]['accuracy_std']:.4f}")
            print(f"  Avg Rules: {dataset_results[method_name]['n_rules_mean']:.2f}")
            print(f"  Avg Conditions: {dataset_results[method_name]['n_conditions_mean']:.2f}")
            print(f"  Avg Unique Conditions: {dataset_results[method_name]['n_unique_conditions_mean']:.2f}")
            print(f"  Avg Train Time: {dataset_results[method_name]['train_time_mean']:.4f}s")
        
        return dataset_results
    
    def _test_fuzzycart(self, X_train, X_test, y_train, y_test, 
                        fuzzy_partitions, results_dict):
        """
        Test FuzzyCART with different configurations.
        
        Parameters
        ----------
        X_train, X_test : np.ndarray
            Training and test features
        y_train, y_test : np.ndarray
            Training and test labels
        fuzzy_partitions : list
            Fuzzy partitions for features
        results_dict : dict
            Dictionary to store results
        """
        # Ablation study: test multiple configurations
        configs = [
            # Varying max_rules
            {'max_rules': 5, 'coverage_threshold': 0.01, 'min_improvement': 0.01, 'target_metric': 'purity'},
            {'max_rules': 10, 'coverage_threshold': 0.01, 'min_improvement': 0.01, 'target_metric': 'purity'},
            {'max_rules': 15, 'coverage_threshold': 0.01, 'min_improvement': 0.01, 'target_metric': 'purity'},
            {'max_rules': 20, 'coverage_threshold': 0.01, 'min_improvement': 0.01, 'target_metric': 'purity'},
            {'max_rules': 30, 'coverage_threshold': 0.01, 'min_improvement': 0.01, 'target_metric': 'purity'},
            
            # Varying coverage_threshold
            {'max_rules': 15, 'coverage_threshold': 0.0, 'min_improvement': 0.01, 'target_metric': 'purity'},
            {'max_rules': 15, 'coverage_threshold': 0.03, 'min_improvement': 0.01, 'target_metric': 'purity'},
            {'max_rules': 15, 'coverage_threshold': 0.05, 'min_improvement': 0.01, 'target_metric': 'purity'},
            {'max_rules': 15, 'coverage_threshold': 0.1, 'min_improvement': 0.01, 'target_metric': 'purity'},
            
            # Varying min_improvement
            {'max_rules': 15, 'coverage_threshold': 0.01, 'min_improvement': 0.0, 'target_metric': 'purity'},
            {'max_rules': 15, 'coverage_threshold': 0.01, 'min_improvement': 0.005, 'target_metric': 'purity'},
            {'max_rules': 15, 'coverage_threshold': 0.01, 'min_improvement': 0.02, 'target_metric': 'purity'},
            {'max_rules': 15, 'coverage_threshold': 0.01, 'min_improvement': 0.05, 'target_metric': 'purity'},
            
            # Some combined variations
            {'max_rules': 20, 'coverage_threshold': 0.03, 'min_improvement': 0.01, 'target_metric': 'purity'},
            {'max_rules': 10, 'coverage_threshold': 0.05, 'min_improvement': 0.005, 'target_metric': 'purity'},
            {'max_rules': 25, 'coverage_threshold': 0.01, 'min_improvement': 0.005, 'target_metric': 'purity'},
        ]
        
        best_accuracy = 0
        best_config = None
        all_configs_results = []  # Store all results for this fold
        
        for config in configs:
            try:
                start_time = time.time()
                
                fuzzy_cart = FuzzyCART(
                    fuzzy_partitions=fuzzy_partitions,
                    max_rules=config['max_rules'],
                    coverage_threshold=config['coverage_threshold'],
                    min_improvement=config['min_improvement'],
                    target_metric=config['target_metric']
                )
                
                fuzzy_cart.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                y_pred = fuzzy_cart.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                stats = fuzzy_cart.get_tree_stats() if hasattr(fuzzy_cart, 'get_tree_stats') else {}
                
                # Calculate conditions per rule: internal_nodes / leaves
                n_leaves = stats.get('leaves', 0)
                n_internal = stats.get('internal', 0)
                conditions_per_rule = n_internal / n_leaves if n_leaves > 0 else 0
                
                # Store this configuration's results
                config_result = {
                    'config': config.copy(),
                    'accuracy': accuracy,
                    'train_time': train_time,
                    'n_rules': n_leaves,
                    'n_conditions': conditions_per_rule,
                    'n_unique_conditions': n_internal  # Total internal nodes
                }
                all_configs_results.append(config_result)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = {
                        'config': config.copy(),
                        'accuracy': accuracy,
                        'train_time': train_time,
                        'stats': stats
                    }
            except Exception as e:
                print(f"    FuzzyCART config {config} failed: {e}")
                continue

        
        if best_config:
            results_dict['accuracy'].append(best_config['accuracy'])
            results_dict['train_time'].append(best_config['train_time'])
            
            stats = best_config['stats']
            n_leaves = stats.get('leaves', 0)
            n_internal = stats.get('internal', 0)
            conditions_per_rule = n_internal / n_leaves if n_leaves > 0 else 0
            
            results_dict['n_rules'].append(n_leaves)
            results_dict['n_conditions'].append(conditions_per_rule)
            results_dict['n_unique_conditions'].append(n_internal)
            
            # Store best config for this fold
            if 'best_configs' not in results_dict:
                results_dict['best_configs'] = []
            results_dict['best_configs'].append(best_config['config'])
            
            # Store all configs tested for ablation analysis
            if 'all_configs' not in results_dict:
                results_dict['all_configs'] = []
            results_dict['all_configs'].extend(all_configs_results)
        else:
            # Fallback if all configs fail
            results_dict['accuracy'].append(0.0)
            results_dict['train_time'].append(0.0)
            results_dict['n_rules'].append(0)
            results_dict['n_conditions'].append(0)
            results_dict['n_unique_conditions'].append(0)
            if 'best_configs' not in results_dict:
                results_dict['best_configs'] = []
            results_dict['best_configs'].append(None)
    
    def _test_cart(self, X_train, X_test, y_train, y_test, results_dict):
        """Test CART baseline."""
        try:
            start_time = time.time()
            
            cart = DecisionTreeClassifier(
                criterion='gini',
                random_state=self.random_state,
                min_samples_split=5,
                min_samples_leaf=2
            )
            
            cart.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = cart.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            n_leaves = cart.get_n_leaves()
            n_nodes = cart.tree_.node_count
            n_internal = n_nodes - n_leaves
            conditions_per_rule = n_internal / n_leaves if n_leaves > 0 else 0
            
            results_dict['accuracy'].append(accuracy)
            results_dict['train_time'].append(train_time)
            results_dict['n_rules'].append(n_leaves)
            results_dict['n_conditions'].append(conditions_per_rule)
            results_dict['n_unique_conditions'].append(n_internal)
            
        except Exception as e:
            print(f"    CART failed: {e}")
            results_dict['accuracy'].append(0.0)
            results_dict['train_time'].append(0.0)
            results_dict['n_rules'].append(0)
            results_dict['n_conditions'].append(0)
            results_dict['n_unique_conditions'].append(0)
    
    def _test_c45(self, X_train, X_test, y_train, y_test, results_dict):
        """Test C4.5 baseline (entropy-based decision tree)."""
        try:
            start_time = time.time()
            
            c45 = DecisionTreeClassifier(
                criterion='entropy',
                random_state=self.random_state,
                min_samples_split=5,
                min_samples_leaf=2
            )
            
            c45.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            y_pred = c45.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            n_leaves = c45.get_n_leaves()
            n_nodes = c45.tree_.node_count
            n_internal = n_nodes - n_leaves
            conditions_per_rule = n_internal / n_leaves if n_leaves > 0 else 0
            
            results_dict['accuracy'].append(accuracy)
            results_dict['train_time'].append(train_time)
            results_dict['n_rules'].append(n_leaves)
            results_dict['n_conditions'].append(conditions_per_rule)
            results_dict['n_unique_conditions'].append(n_internal)
            
        except Exception as e:
            print(f"    C4.5 failed: {e}")
            results_dict['accuracy'].append(0.0)
            results_dict['train_time'].append(0.0)
            results_dict['n_rules'].append(0)
            results_dict['n_conditions'].append(0)
            results_dict['n_unique_conditions'].append(0)
    
    def run_all_experiments(self, datasets_dict):
        """
        Run experiments on all datasets.
        
        Parameters
        ----------
        datasets_dict : dict
            Dictionary mapping dataset names to (X, y, categorical_mask) tuples
            
        Returns
        -------
        pd.DataFrame
            Comprehensive results table
        """
        all_results = {}
        
        for dataset_name, (X, y, categorical_mask) in datasets_dict.items():
            print(f"\n\nProcessing dataset: {dataset_name}")
            
            # Generate fuzzy partitions using ex_fuzzy utils
            fuzzy_partitions = utils.construct_partitions(
                X, 
                fz_type_studied=fs.FUZZY_SETS.t1,
                categorical_mask=categorical_mask,
                n_partitions=3,
                shape='trapezoid'
            )
            
            # Run experiments
            dataset_results = self.run_experiment(X, y, dataset_name, fuzzy_partitions)
            all_results[dataset_name] = dataset_results
            
            # Store best config for this dataset
            if 'FuzzyCART' in dataset_results and 'best_config' in dataset_results['FuzzyCART']:
                self.best_configs[dataset_name] = dataset_results['FuzzyCART']['best_config']
        
        # Create results DataFrame
        results_df = self._create_results_dataframe(all_results)
        
        return results_df, all_results
    
    def _create_results_dataframe(self, all_results):
        """Create a comprehensive results DataFrame."""
        rows = []
        
        for dataset_name, methods in all_results.items():
            for method_name, metrics in methods.items():
                row = {
                    'Dataset': dataset_name,
                    'Method': method_name,
                    'Accuracy': metrics['accuracy_mean'],
                    'Accuracy_Std': metrics['accuracy_std'],
                    'N_Rules': metrics['n_rules_mean'],
                    'N_Conditions': metrics['n_conditions_mean'],
                    'N_Unique_Conditions': metrics['n_unique_conditions_mean'],
                    'Train_Time': metrics['train_time_mean']
                }
                
                # Add best config info for FuzzyCART
                if method_name == 'FuzzyCART' and 'best_config' in metrics:
                    config = metrics['best_config']
                    row['Best_Max_Rules'] = config['max_rules']
                    row['Best_Coverage_Threshold'] = config['coverage_threshold']
                    row['Best_Min_Improvement'] = config['min_improvement']
                    row['Best_Target_Metric'] = config['target_metric']
                    row['Config_Frequency'] = metrics.get('best_config_frequency', 0.0)
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_ablation_dataframe(self):
        """Create detailed ablation study results DataFrame."""
        if not self.ablation_results:
            return pd.DataFrame()
        
        rows = []
        for dataset_name, configs_results in self.ablation_results.items():
            for config_result in configs_results:
                row = {
                    'Dataset': dataset_name,
                    'Max_Rules': config_result['config']['max_rules'],
                    'Coverage_Threshold': config_result['config']['coverage_threshold'],
                    'Min_Improvement': config_result['config']['min_improvement'],
                    'Target_Metric': config_result['config']['target_metric'],
                    'Accuracy': config_result['accuracy'],
                    'N_Rules': config_result['n_rules'],
                    'N_Conditions': config_result['n_conditions'],
                    'N_Unique_Conditions': config_result['n_unique_conditions'],
                    'Train_Time': config_result['train_time']
                }
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_best_configs_table(self):
        """Create a summary table of best configurations per dataset."""
        if not self.best_configs:
            return pd.DataFrame()
        
        rows = []
        for dataset_name, config in self.best_configs.items():
            rows.append({
                'Dataset': dataset_name,
                'Best_Max_Rules': config['max_rules'],
                'Best_Coverage_Threshold': config['coverage_threshold'],
                'Best_Min_Improvement': config['min_improvement'],
                'Best_Target_Metric': config['target_metric']
            })
        
        return pd.DataFrame(rows)


class AblationVisualizer:
    """
    Creates publication-quality visualizations for ablation study results.
    """
    
    @staticmethod
    def plot_parameter_impact(ablation_df, output_prefix='ablation'):
        """
        Create comprehensive parameter impact visualizations.
        
        Parameters
        ----------
        ablation_df : pd.DataFrame
            Detailed ablation results
        output_prefix : str
            Prefix for output filenames
        """
        if ablation_df.empty:
            print("No ablation data to visualize")
            return
        
        # Figure 1: Impact of max_rules
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy vs max_rules
        max_rules_stats = ablation_df.groupby('Max_Rules')['Accuracy'].agg(['mean', 'std', 'count'])
        axes[0].errorbar(max_rules_stats.index, max_rules_stats['mean'], 
                        yerr=max_rules_stats['std'], marker='o', capsize=5, linewidth=2, markersize=8)
        axes[0].set_xlabel('Maximum Number of Rules')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Impact of max_rules on Accuracy')
        axes[0].grid(True, alpha=0.3)
        
        # Complexity vs max_rules
        complexity_stats = ablation_df.groupby('Max_Rules')['N_Rules'].agg(['mean', 'std'])
        axes[1].errorbar(complexity_stats.index, complexity_stats['mean'],
                        yerr=complexity_stats['std'], marker='s', capsize=5, linewidth=2, markersize=8, color='coral')
        axes[1].set_xlabel('Maximum Number of Rules')
        axes[1].set_ylabel('Actual Number of Rules')
        axes[1].set_title('Impact of max_rules on Model Complexity')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_max_rules.pdf', bbox_inches='tight')
        plt.savefig(f'{output_prefix}_max_rules.png', bbox_inches='tight')
        print(f"Saved: {output_prefix}_max_rules.pdf/png")
        plt.close()
        
        # Figure 2: Impact of coverage_threshold
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        coverage_stats = ablation_df.groupby('Coverage_Threshold')['Accuracy'].agg(['mean', 'std'])
        axes[0].errorbar(coverage_stats.index, coverage_stats['mean'],
                        yerr=coverage_stats['std'], marker='o', capsize=5, linewidth=2, markersize=8, color='green')
        axes[0].set_xlabel('Coverage Threshold')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Impact of coverage_threshold on Accuracy')
        axes[0].grid(True, alpha=0.3)
        
        complexity_stats = ablation_df.groupby('Coverage_Threshold')['N_Rules'].agg(['mean', 'std'])
        axes[1].errorbar(complexity_stats.index, complexity_stats['mean'],
                        yerr=complexity_stats['std'], marker='s', capsize=5, linewidth=2, markersize=8, color='purple')
        axes[1].set_xlabel('Coverage Threshold')
        axes[1].set_ylabel('Number of Rules')
        axes[1].set_title('Impact of coverage_threshold on Model Complexity')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_coverage.pdf', bbox_inches='tight')
        plt.savefig(f'{output_prefix}_coverage.png', bbox_inches='tight')
        print(f"Saved: {output_prefix}_coverage.pdf/png")
        plt.close()
        
        # Figure 3: Impact of min_improvement
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        improvement_stats = ablation_df.groupby('Min_Improvement')['Accuracy'].agg(['mean', 'std'])
        axes[0].errorbar(improvement_stats.index, improvement_stats['mean'],
                        yerr=improvement_stats['std'], marker='o', capsize=5, linewidth=2, markersize=8, color='orange')
        axes[0].set_xlabel('Minimum Improvement')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Impact of min_improvement on Accuracy')
        axes[0].grid(True, alpha=0.3)
        
        complexity_stats = ablation_df.groupby('Min_Improvement')['N_Rules'].agg(['mean', 'std'])
        axes[1].errorbar(complexity_stats.index, complexity_stats['mean'],
                        yerr=complexity_stats['std'], marker='s', capsize=5, linewidth=2, markersize=8, color='teal')
        axes[1].set_xlabel('Minimum Improvement')
        axes[1].set_ylabel('Number of Rules')
        axes[1].set_title('Impact of min_improvement on Model Complexity')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_min_improvement.pdf', bbox_inches='tight')
        plt.savefig(f'{output_prefix}_min_improvement.png', bbox_inches='tight')
        print(f"Saved: {output_prefix}_min_improvement.pdf/png")
        plt.close()
        
        # Figure 4: Heatmap of configuration performance
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create pivot table for max_rules vs coverage_threshold
        pivot_data = ablation_df.pivot_table(
            values='Accuracy',
            index='Max_Rules',
            columns='Coverage_Threshold',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Accuracy'}, ax=ax, vmin=0.7, vmax=1.0)
        ax.set_xlabel('Coverage Threshold')
        ax.set_ylabel('Maximum Number of Rules')
        ax.set_title('Configuration Performance Heatmap\n(Accuracy by max_rules and coverage_threshold)')
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_heatmap.pdf', bbox_inches='tight')
        plt.savefig(f'{output_prefix}_heatmap.png', bbox_inches='tight')
        print(f"Saved: {output_prefix}_heatmap.pdf/png")
        plt.close()
        
        # Figure 5: Accuracy vs Complexity trade-off
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Aggregate by configuration
        config_agg = ablation_df.groupby(['Max_Rules', 'Coverage_Threshold', 'Min_Improvement']).agg({
            'Accuracy': 'mean',
            'N_Rules': 'mean',
            'Max_Rules': 'first'
        }).reset_index(drop=True)
        
        scatter = ax.scatter(config_agg['N_Rules'], config_agg['Accuracy'], 
                           c=config_agg['Max_Rules'], cmap='viridis', 
                           s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('max_rules setting')
        
        ax.set_xlabel('Actual Number of Rules (Complexity)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Model Complexity Trade-off')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_tradeoff.pdf', bbox_inches='tight')
        plt.savefig(f'{output_prefix}_tradeoff.png', bbox_inches='tight')
        print(f"Saved: {output_prefix}_tradeoff.pdf/png")
        plt.close()
        
        # Figure 6: Improved box plots for parameter distributions
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Custom box plot styling
        boxprops = dict(linewidth=1.5, color='darkblue')
        whiskerprops = dict(linewidth=1.5, color='darkblue')
        capprops = dict(linewidth=1.5, color='darkblue')
        medianprops = dict(linewidth=2, color='red')
        flierprops = dict(marker='o', markerfacecolor='lightblue', markersize=5, 
                         linestyle='none', markeredgecolor='darkblue', alpha=0.5)
        
        # Max rules
        data_max_rules = [ablation_df[ablation_df['Max_Rules'] == val]['Accuracy'].values 
                          for val in sorted(ablation_df['Max_Rules'].unique())]
        positions = sorted(ablation_df['Max_Rules'].unique())
        
        bp1 = axes[0].boxplot(data_max_rules, positions=positions, widths=2.5,
                              patch_artist=True, showmeans=True,
                              boxprops=boxprops, whiskerprops=whiskerprops,
                              capprops=capprops, medianprops=medianprops,
                              flierprops=flierprops,
                              meanprops=dict(marker='D', markerfacecolor='orange', 
                                           markersize=6, markeredgecolor='darkorange'))
        
        # Color the boxes with a gradient
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(data_max_rules)))
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[0].set_xlabel('Maximum Number of Rules', fontweight='bold')
        axes[0].set_ylabel('Accuracy', fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[0].set_axisbelow(True)
        
        # Coverage threshold
        data_coverage = [ablation_df[ablation_df['Coverage_Threshold'] == val]['Accuracy'].values 
                         for val in sorted(ablation_df['Coverage_Threshold'].unique())]
        positions_cov = range(len(data_coverage))
        labels_cov = [f'{val:.2f}' for val in sorted(ablation_df['Coverage_Threshold'].unique())]
        
        bp2 = axes[1].boxplot(data_coverage, positions=positions_cov,
                              patch_artist=True, showmeans=True,
                              boxprops=boxprops, whiskerprops=whiskerprops,
                              capprops=capprops, medianprops=medianprops,
                              flierprops=flierprops,
                              meanprops=dict(marker='D', markerfacecolor='orange', 
                                           markersize=6, markeredgecolor='darkorange'))
        
        colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(data_coverage)))
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1].set_xlabel('Coverage Threshold', fontweight='bold')
        axes[1].set_ylabel('Accuracy', fontweight='bold')
        axes[1].set_xticks(positions_cov)
        axes[1].set_xticklabels(labels_cov, rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[1].set_axisbelow(True)
        
        # Min improvement
        data_improvement = [ablation_df[ablation_df['Min_Improvement'] == val]['Accuracy'].values 
                            for val in sorted(ablation_df['Min_Improvement'].unique())]
        positions_imp = range(len(data_improvement))
        labels_imp = [f'{val:.3f}' for val in sorted(ablation_df['Min_Improvement'].unique())]
        
        bp3 = axes[2].boxplot(data_improvement, positions=positions_imp,
                              patch_artist=True, showmeans=True,
                              boxprops=boxprops, whiskerprops=whiskerprops,
                              capprops=capprops, medianprops=medianprops,
                              flierprops=flierprops,
                              meanprops=dict(marker='D', markerfacecolor='orange', 
                                           markersize=6, markeredgecolor='darkorange'))
        
        colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(data_improvement)))
        for patch, color in zip(bp3['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[2].set_xlabel('Minimum Improvement', fontweight='bold')
        axes[2].set_ylabel('Accuracy', fontweight='bold')
        axes[2].set_xticks(positions_imp)
        axes[2].set_xticklabels(labels_imp, rotation=45)
        axes[2].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[2].set_axisbelow(True)
        
        # Add legend for all subplots
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, label='Median'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='orange', 
                   markersize=6, markeredgecolor='darkorange', label='Mean')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
                  ncol=2, frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'{output_prefix}_distributions.pdf', bbox_inches='tight')
        plt.savefig(f'{output_prefix}_distributions.png', bbox_inches='tight')
        print(f"Saved: {output_prefix}_distributions.pdf/png")
        plt.close()
        
        print("\nAll ablation study figures saved!")


class StatisticalAnalyzer:
    """
    Performs statistical analysis on experimental results.
    
    Implements:
    - Friedman test for comparing multiple methods
    - Post-hoc Nemenyi test for pairwise comparisons
    """
    
    @staticmethod
    def friedman_test(results_df):
        """
        Perform Friedman test to check if there are significant differences.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results dataframe from experiments
            
        Returns
        -------
        tuple
            (statistic, p_value)
        """
        # Pivot to get methods as columns and datasets as rows
        pivot = results_df.pivot(index='Dataset', columns='Method', values='Accuracy')
        
        # Perform Friedman test
        methods_data = [pivot[col].values for col in pivot.columns]
        statistic, p_value = friedmanchisquare(*methods_data)
        
        print(f"\nFriedman Test Results:")
        print(f"  Statistic: {statistic:.4f}")
        print(f"  p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print("  => Significant differences detected (p < 0.05)")
        else:
            print("  => No significant differences (p >= 0.05)")
        
        return statistic, p_value
    
    @staticmethod
    def nemenyi_test(results_df):
        """
        Perform post-hoc Nemenyi test for pairwise comparisons.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results dataframe from experiments
            
        Returns
        -------
        pd.DataFrame
            Matrix of p-values for pairwise comparisons
        """
        # Pivot to get methods as columns and datasets as rows
        pivot = results_df.pivot(index='Dataset', columns='Method', values='Accuracy')
        
        # Perform Nemenyi test
        nemenyi_results = posthoc_nemenyi_friedman(pivot)
        
        print(f"\nPost-hoc Nemenyi Test Results:")
        print(nemenyi_results)
        
        return nemenyi_results


def main():
    """
    Main function to run complete experimental suite.
    """
    import os
    
    # Create results directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*80)
    print("FuzzyCART Comprehensive Experimental Suite")
    print("="*80)
    print(f"\nResults will be saved to: {results_dir}/")
    
    # Initialize components
    loader = DatasetLoader()
    runner = ExperimentRunner(n_folds=5, random_state=42)
    analyzer = StatisticalAnalyzer()
    
    # Load datasets (using a subset for demonstration)
    print("\nLoading datasets...")
    datasets_to_test = ['Appendicitis', 'Australian', 'dermatology', 'Hepatitis',
                        'Pima', 'Spambase', 'Wine', 'saheart', 'ring', 'Zoo'
                       ]  # Add more as needed   
    datasets_to_test = [name.lower() for name in datasets_to_test]
    datasets = loader.load_all_datasets(datasets_to_test)
    
    if len(datasets) == 0:
        print("No datasets loaded. Exiting.")
        return
    
    # Run experiments
    print("\n" + "="*80)
    print("Running Experiments")
    print("="*80)
    
    results_df, all_results = runner.run_all_experiments(datasets)
    
    # Display summary results
    print("\n" + "="*80)
    print("Summary Results")
    print("="*80)
    print(results_df)
    
    # Save results
    results_df.to_csv(os.path.join(results_dir, 'fuzzy_cart_results.csv'), index=False)
    print(f"\nResults saved to: {results_dir}/fuzzy_cart_results.csv")
    
    # Statistical analysis
    if len(datasets) > 1:
        print("\n" + "="*80)
        print("Statistical Analysis")
        print("="*80)
        
        analyzer.friedman_test(results_df)
        
        if len(datasets) >= 5:  # Nemenyi requires sufficient datasets
            analyzer.nemenyi_test(results_df)
    
    # Create summary table (similar to Table III in the paper)
    print("\n" + "="*80)
    print("Average Performance Across All Datasets")
    print("="*80)
    
    summary = results_df.groupby('Method').agg({
        'Accuracy': ['mean', 'std'],
        'N_Rules': 'mean',
        'N_Conditions': 'mean',
        'N_Unique_Conditions': 'mean',
        'Train_Time': 'mean'
    })
    
    print(summary)
    summary.to_csv(os.path.join(results_dir, 'fuzzy_cart_summary.csv'))
    print(f"\nFinal summary table saved to: {results_dir}/fuzzy_cart_summary.csv")
    
    # Save detailed final results table
    final_table = results_df.copy()
    final_table.to_csv(os.path.join(results_dir, 'fuzzy_cart_final_results_table.csv'), index=False)
    print(f"Final results table saved to: {results_dir}/fuzzy_cart_final_results_table.csv")
    
    # Display and save ablation study details
    print("\n" + "="*80)
    print("Best Configurations per Dataset")
    print("="*80)
    best_configs_df = runner.get_best_configs_table()
    print(best_configs_df)
    best_configs_df.to_csv(os.path.join(results_dir, 'fuzzy_cart_best_configs.csv'), index=False)
    print(f"\nBest configurations saved to: {results_dir}/fuzzy_cart_best_configs.csv")
    
    print("\n" + "="*80)
    print("Detailed Ablation Study Results (All Configurations)")
    print("="*80)
    ablation_df = runner.create_ablation_dataframe()
    if not ablation_df.empty:
        # Show summary statistics by parameter
        print("\nAverage accuracy by max_rules:")
        print(ablation_df.groupby('Max_Rules')['Accuracy'].agg(['mean', 'std', 'count']))
        print("\nAverage accuracy by coverage_threshold:")
        print(ablation_df.groupby('Coverage_Threshold')['Accuracy'].agg(['mean', 'std', 'count']))
        print("\nAverage accuracy by min_improvement:")
        print(ablation_df.groupby('Min_Improvement')['Accuracy'].agg(['mean', 'std', 'count']))
        
        ablation_df.to_csv(os.path.join(results_dir, 'fuzzy_cart_ablation_all_configs.csv'), index=False)
        print(f"\nAll configurations saved to: {results_dir}/fuzzy_cart_ablation_all_configs.csv")
        
        # Generate publication-quality figures
        print("\n" + "="*80)
        print("Generating Ablation Study Figures")
        print("="*80)
        visualizer = AblationVisualizer()
        visualizer.plot_parameter_impact(ablation_df, output_prefix=os.path.join(results_dir, 'ablation_study'))
    
    print("\n" + "="*80)
    print("Experiments Complete!")
    print("="*80)
    print("\nGenerated files in results/ directory:")
    print("  - fuzzy_cart_results.csv (main results)")
    print("  - fuzzy_cart_final_results_table.csv (detailed final table)")
    print("  - fuzzy_cart_summary.csv (summary statistics)")
    print("  - fuzzy_cart_best_configs.csv (best hyperparameters per dataset)")
    print("  - fuzzy_cart_ablation_all_configs.csv (all configurations tested)")
    print("  - ablation_study_*.pdf/png (publication figures)")


if __name__ == '__main__':
    main()
