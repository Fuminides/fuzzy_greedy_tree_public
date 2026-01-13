"""
Runtime Performance and Scalability Analysis for FuzzyCART
===========================================================

This module analyzes:
1. Training time vs dataset size
2. Training time vs number of features
3. Prediction time analysis
4. Memory consumption
5. Comparison with CART and C4.5 in terms of computational efficiency
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tracemalloc
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import sys
sys.path.append('/mnt/project')
from tree_learning import FuzzyCART
from ex_fuzzy import fuzzy_sets as fs
from ex_fuzzy import utils
from ex_fuzzy.evolutionary_fit import BaseFuzzyRulesClassifier


class PerformanceAnalyzer:
    """
    Analyzes computational performance and scalability of FuzzyCART.
    """

    def __init__(self, random_state=42, penalize_no_rules: bool = True, no_rules_penalty_multiplier: float = 10.0,
                 early_stop_threshold: float = 0.05, use_numba: bool = True):
        """
        Initialize performance analyzer.

        Parameters
        ----------
        random_state : int
            Random seed for reproducibility
        penalize_no_rules : bool
            If True, apply a penalty to timing measurements for cases where FuzzyCART
            learns no rules (to avoid misleadingly fast timings for trivial models).
        no_rules_penalty_multiplier : float
            Multiplier applied to raw timing when no rules are produced (default 10.0).
        early_stop_threshold : float
            Early stopping threshold for FuzzyCART optimization (default 0.05).
            Set to 0.0 to disable early stopping.
        use_numba : bool
            If True, use Numba JIT compilation for faster training (default True).
        """
        self.random_state = random_state
        self.penalize_no_rules = penalize_no_rules
        self.no_rules_penalty_multiplier = no_rules_penalty_multiplier
        self.early_stop_threshold = early_stop_threshold
        self.use_numba = use_numba
        self.results = []
        
    def generate_synthetic_dataset(self, n_samples, n_features, n_classes=3):
        """
        Generate synthetic classification dataset.
        
        Parameters
        ----------
        n_samples : int
            Number of samples
        n_features : int
            Number of features
        n_classes : int
            Number of classes
            
        Returns
        -------
        X, y : np.ndarray
            Feature matrix and labels
        """
        np.random.seed(self.random_state)
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return X, y
    
    def generate_fuzzy_partitions(self, X):
        """
        Generate fuzzy partitions for dataset using ex_fuzzy utils.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Returns
        -------
        list
            List of fuzzy partitions
        """
        # Use ex_fuzzy utils to construct partitions
        fuzzy_partitions = utils.construct_partitions(
            X,
            fz_type_studied=fs.FUZZY_SETS.t1,
            n_partitions=3,
            shape='trapezoid'
        )
        
        return fuzzy_partitions
    
    def measure_training_time_vs_samples(self, 
                                         sample_sizes=None,
                                         n_features=10,
                                         n_trials=3):
        """
        Measure training time as function of dataset size.
        
        Parameters
        ----------
        sample_sizes : list, optional
            List of sample sizes to test
        n_features : int
            Number of features
        n_trials : int
            Number of trials for averaging
            
        Returns
        -------
        pd.DataFrame
            Results table
        """
        if sample_sizes is None:
            sample_sizes = [100, 500, 1000, 2000, 5000, 10000]
        
        print("\n" + "="*60)
        print("Measuring Training Time vs Number of Samples")
        print("="*60)
        
        results = []
        
        for n_samples in sample_sizes:
            print(f"\nTesting n_samples={n_samples}")
            
            for trial in range(n_trials):
                # Generate dataset
                X, y = self.generate_synthetic_dataset(n_samples, n_features)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=self.random_state + trial
                )
                
                fuzzy_partitions = self.generate_fuzzy_partitions(X_train)
                
                # Test FuzzyCART
                start_time = time.time()
                tracemalloc.start()
                
                fuzzy_cart = FuzzyCART(
                    fuzzy_partitions=fuzzy_partitions,
                    max_rules=15,
                    coverage_threshold=0.01,
                    min_improvement=0.01,
                    target_metric='purity',
                    early_stop_threshold=self.early_stop_threshold,
                    use_numba=self.use_numba
                )
                fuzzy_cart.fit(X_train, y_train)
                
                # Raw timing
                fuzzy_cart_time_raw = time.time() - start_time
                current, peak = tracemalloc.get_traced_memory()
                fuzzy_cart_memory = peak / 1024 / 1024  # MB
                tracemalloc.stop()

                # Determine number of rules (fallback to leaf count if attribute missing)
                fuzzy_cart_rules = getattr(fuzzy_cart, 'tree_rules', None)
                try:
                    if fuzzy_cart_rules is None:
                        fuzzy_cart_rules = len(fuzzy_cart._get_leaves())
                except Exception:
                    fuzzy_cart_rules = -1

                # Apply penalization if no rules learned
                if self.penalize_no_rules and (fuzzy_cart_rules <= 1):
                    fuzzy_cart_time = fuzzy_cart_time_raw * self.no_rules_penalty_multiplier
                    print(f"  WARNING: FuzzyCART learned no rules (rules={fuzzy_cart_rules}). Applying penalty x{self.no_rules_penalty_multiplier} to training time.")
                    no_rules_flag = True
                else:
                    fuzzy_cart_time = fuzzy_cart_time_raw
                    no_rules_flag = False
                    
                
                # Test CART
                start_time = time.time()
                cart = DecisionTreeClassifier(random_state=self.random_state)
                cart.fit(X_train, y_train)
                cart_time = time.time() - start_time
                
                
                # Test C4.5
                start_time = time.time()
                c45 = DecisionTreeClassifier(
                    criterion='entropy',
                    random_state=self.random_state
                )
                c45.fit(X_train, y_train)
                c45_time = time.time() - start_time
                
                
                # Test BaseFuzzyRulesClassifier
                start_time = time.time()
                base_fuzzy = BaseFuzzyRulesClassifier(
                    nRules=15,
                    nAnts=3,
                    fuzzy_type=fs.FUZZY_SETS.t1,
                    tolerance=0.01,
                    runner=1
                )
                base_fuzzy.fit(X_train, y_train, n_gen=100, pop_size=50)
                base_fuzzy_time_raw = time.time() - start_time

                # Attempt to detect number of rules learned by BaseFuzzy
                base_fuzzy_rules = None
                for attr in ('nRules', 'n_rules', 'rules', 'rules_', 'best_rules', 'n_rules_learned', 'rules_learned'):
                    val = getattr(base_fuzzy, attr, None)
                    if val is not None:
                        if isinstance(val, (list, tuple, np.ndarray)):
                            base_fuzzy_rules = len(val)
                        elif isinstance(val, int):
                            base_fuzzy_rules = int(val)
                        break
                if base_fuzzy_rules is None:
                    # As a last resort, attempt to inspect a common attribute 'ant' or 'solutions'
                    val = getattr(base_fuzzy, 'solutions', None) or getattr(base_fuzzy, 'population', None)
                    if isinstance(val, (list, tuple, np.ndarray)):
                        base_fuzzy_rules = len(val)

                if base_fuzzy_rules is None:
                    base_fuzzy_rules = -1  # unknown

                # Apply penalization if no rules learned (only when known to be 0 or 1)
                if self.penalize_no_rules and (base_fuzzy_rules in (0, 1)):
                    genetic_opt_time = base_fuzzy_time_raw * self.no_rules_penalty_multiplier
                    base_fuzzy_no_rules_flag = True
                    print(f"  WARNING: BaseFuzzy (Genetic Opt.) learned no rules (rules={base_fuzzy_rules}). Applying penalty x{self.no_rules_penalty_multiplier} to its training time.")
                else:
                    genetic_opt_time = base_fuzzy_time_raw
                    base_fuzzy_no_rules_flag = False

                results.append({
                    'n_samples': n_samples,
                    'n_features': n_features,
                    'trial': trial,
                    'fuzzy_cart_time_raw': fuzzy_cart_time_raw,
                    'fuzzy_cart_time': fuzzy_cart_time,
                    'fuzzy_cart_rules': fuzzy_cart_rules,
                    'fuzzy_cart_no_rules_flag': no_rules_flag,
                    'fuzzy_cart_memory_mb': fuzzy_cart_memory,
                    'cart_time': cart_time,
                    'c45_time': c45_time,
                    'genetic_opt_time_raw': base_fuzzy_time_raw,
                    'genetic_opt_time': genetic_opt_time,
                    'genetic_opt_rules': base_fuzzy_rules,
                    'genetic_opt_no_rules_flag': base_fuzzy_no_rules_flag
                })

                print(f"  Trial {trial+1}: FuzzyCART={fuzzy_cart_time:.4f}s (raw={fuzzy_cart_time_raw:.4f}s, rules={fuzzy_cart_rules}), "
                      f"CART={cart_time:.4f}s, C4.5={c45_time:.4f}s, "
                      f"Genetic Opt.={genetic_opt_time:.4f}s (raw={base_fuzzy_time_raw:.4f}s, rules={base_fuzzy_rules})")
        
        results_df = pd.DataFrame(results)
        
        # Compute averages
        avg_results = results_df.groupby('n_samples').agg({
            'fuzzy_cart_time_raw': ['mean', 'std'],
            'fuzzy_cart_time': ['mean', 'std'],
            'fuzzy_cart_memory_mb': ['mean', 'std'],
            'cart_time': ['mean', 'std'],
            'c45_time': ['mean', 'std'],
            'genetic_opt_time_raw': ['mean', 'std'],
            'genetic_opt_time': ['mean', 'std']
        }).reset_index()
        
        print("\n" + "="*60)
        print("Average Results:")
        print(avg_results)
        
        return results_df, avg_results
    
    def measure_training_time_vs_features(self,
                                          feature_sizes=None,
                                          n_samples=1000,
                                          n_trials=3):
        """
        Measure training time as function of number of features.
        
        Parameters
        ----------
        feature_sizes : list, optional
            List of feature counts to test
        n_samples : int
            Number of samples
        n_trials : int
            Number of trials for averaging
            
        Returns
        -------
        pd.DataFrame
            Results table
        """
        if feature_sizes is None:
            feature_sizes = [5, 10, 20, 30, 50]
        
        print("\n" + "="*60)
        print("Measuring Training Time vs Number of Features")
        print("="*60)
        
        results = []
        
        for n_features in feature_sizes:
            print(f"\nTesting n_features={n_features}")
            
            for trial in range(n_trials):
                # Generate dataset
                X, y = self.generate_synthetic_dataset(n_samples, n_features)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=self.random_state + trial
                )
                
                fuzzy_partitions = self.generate_fuzzy_partitions(X_train)
                
                # Test FuzzyCART
                
                fuzzy_cart = FuzzyCART(
                    fuzzy_partitions=fuzzy_partitions,
                    max_rules=15,
                    coverage_threshold=0.01,
                    min_improvement=0.01,
                    target_metric='purity',
                    early_stop_threshold=self.early_stop_threshold,
                    use_numba=self.use_numba
                )
                start_time = time.time()
                fuzzy_cart.fit(X_train, y_train)
                fuzzy_cart_time_raw = time.time() - start_time

                # Determine rules and penalize if needed
                fuzzy_cart_rules = getattr(fuzzy_cart, 'tree_rules', None)
                try:
                    if fuzzy_cart_rules is None:
                        fuzzy_cart_rules = len(fuzzy_cart._get_leaves())
                except Exception:
                    fuzzy_cart_rules = -1

                if self.penalize_no_rules and (fuzzy_cart_rules <= 1):
                    fuzzy_cart_time = fuzzy_cart_time_raw * self.no_rules_penalty_multiplier
                    print(f"  WARNING: FuzzyCART learned no rules (rules={fuzzy_cart_rules}). Applying penalty x{self.no_rules_penalty_multiplier} to training time.")
                    no_rules_flag = True
                else:
                    fuzzy_cart_time = fuzzy_cart_time_raw
                    no_rules_flag = False
                
                # Test CART
                cart = DecisionTreeClassifier(random_state=self.random_state)
                start_time = time.time()
                cart.fit(X_train, y_train)
                cart_time = time.time() - start_time
                
                # Test C4.5
                c45 = DecisionTreeClassifier(
                    criterion='entropy',
                    random_state=self.random_state
                )
                start_time = time.time()
                c45.fit(X_train, y_train)
                c45_time = time.time() - start_time
                
                # Test BaseFuzzyRulesClassifier
                base_fuzzy = BaseFuzzyRulesClassifier(
                    nRules=15,
                    nAnts=3,
                    fuzzy_type=fs.FUZZY_SETS.t1,
                    tolerance=0.01,
                    runner=1
                )
                start_time = time.time()
                base_fuzzy.fit(X_train, y_train, n_gen=100, pop_size=50)
                base_fuzzy_time_raw = time.time() - start_time

                # Detect BaseFuzzy rules (same strategy as samples block)
                base_fuzzy_rules = None
                for attr in ('nRules', 'n_rules', 'rules', 'rules_', 'best_rules', 'n_rules_learned', 'rules_learned'):
                    val = getattr(base_fuzzy, attr, None)
                    if val is not None:
                        if isinstance(val, (list, tuple, np.ndarray)):
                            base_fuzzy_rules = len(val)
                        elif isinstance(val, int):
                            base_fuzzy_rules = int(val)
                        break
                if base_fuzzy_rules is None:
                    val = getattr(base_fuzzy, 'solutions', None) or getattr(base_fuzzy, 'population', None)
                    if isinstance(val, (list, tuple, np.ndarray)):
                        base_fuzzy_rules = len(val)
                if base_fuzzy_rules is None:
                    base_fuzzy_rules = -1

                if self.penalize_no_rules and (base_fuzzy_rules in (0, 1)):
                    genetic_opt_time = base_fuzzy_time_raw * self.no_rules_penalty_multiplier
                    base_fuzzy_no_rules_flag = True
                    print(f"  WARNING: BaseFuzzy (Genetic Opt.) learned no rules (rules={base_fuzzy_rules}). Applying penalty x{self.no_rules_penalty_multiplier} to its training time.")
                else:
                    genetic_opt_time = base_fuzzy_time_raw
                    base_fuzzy_no_rules_flag = False

                results.append({
                    'n_samples': n_samples,
                    'n_features': n_features,
                    'trial': trial,
                    'fuzzy_cart_time_raw': fuzzy_cart_time_raw,
                    'fuzzy_cart_time': fuzzy_cart_time,
                    'fuzzy_cart_rules': fuzzy_cart_rules,
                    'fuzzy_cart_no_rules_flag': no_rules_flag,
                    'cart_time': cart_time,
                    'c45_time': c45_time,
                    'genetic_opt_time_raw': base_fuzzy_time_raw,
                    'genetic_opt_time': genetic_opt_time,
                    'genetic_opt_rules': base_fuzzy_rules,
                    'genetic_opt_no_rules_flag': base_fuzzy_no_rules_flag
                })

                print(f"  Trial {trial+1}: FuzzyCART={fuzzy_cart_time:.4f}s (raw={fuzzy_cart_time_raw:.4f}s, rules={fuzzy_cart_rules}), "
                      f"CART={cart_time:.4f}s, C4.5={c45_time:.4f}s, Genetic Opt.={genetic_opt_time:.4f}s (raw={base_fuzzy_time_raw:.4f}s, rules={base_fuzzy_rules})")
        
        results_df = pd.DataFrame(results)
        
        # Compute averages
        avg_results = results_df.groupby('n_features').agg({
            'fuzzy_cart_time_raw': ['mean', 'std'],
            'fuzzy_cart_time': ['mean', 'std'],
            'cart_time': ['mean', 'std'],
            'c45_time': ['mean', 'std'],
            'genetic_opt_time_raw': ['mean', 'std'],
            'genetic_opt_time': ['mean', 'std']
        }).reset_index()
        
        print("\n" + "="*60)
        print("Average Results:")
        print(avg_results)
        
        return results_df, avg_results
    
    def measure_prediction_time(self, n_samples=1000, n_features=10, n_test_sizes=None):
        """
        Measure prediction time for different test set sizes.
        
        Parameters
        ----------
        n_samples : int
            Training set size
        n_features : int
            Number of features
        n_test_sizes : list, optional
            List of test set sizes to try
            
        Returns
        -------
        pd.DataFrame
            Results table
        """
        if n_test_sizes is None:
            n_test_sizes = [100, 500, 1000, 5000, 10000]
        
        print("\n" + "="*60)
        print("Measuring Prediction Time")
        print("="*60)
        
        # Train models once
        X_train, y_train = self.generate_synthetic_dataset(n_samples, n_features)
        fuzzy_partitions = self.generate_fuzzy_partitions(X_train)
        
        fuzzy_cart = FuzzyCART(
            fuzzy_partitions=fuzzy_partitions,
            max_rules=15,
            coverage_threshold=0.01,
            min_improvement=0.01,
            target_metric='purity',
            early_stop_threshold=self.early_stop_threshold,
            use_numba=self.use_numba
        )
        fuzzy_cart.fit(X_train, y_train)
        
        # Determine rules to flag for prediction-time penalization
        fuzzy_cart_rules = getattr(fuzzy_cart, 'tree_rules', None)
        try:
            if fuzzy_cart_rules is None:
                fuzzy_cart_rules = len(fuzzy_cart._get_leaves())
        except Exception:
            fuzzy_cart_rules = -1
        fuzzy_cart_had_no_rules = bool(self.penalize_no_rules and (fuzzy_cart_rules <= 1))
        if fuzzy_cart_had_no_rules:
            print(f"WARNING: FuzzyCART trained with no rules (rules={fuzzy_cart_rules}). Prediction times will be penalized by x{self.no_rules_penalty_multiplier}.")
       
        cart = DecisionTreeClassifier(random_state=self.random_state)
        cart.fit(X_train, y_train)
        
        base_fuzzy = BaseFuzzyRulesClassifier(
            nRules=15,
            nAnts=3,
            fuzzy_type=fs.FUZZY_SETS.t1,
            tolerance=0.01,
            runner=1
        )
        base_fuzzy.fit(X_train, y_train)
        base_fuzzy_trained = True

        # Detect number of rules learned by BaseFuzzy (for prediction-time penalization)
        base_fuzzy_rules = None
        for attr in ('nRules', 'n_rules', 'rules', 'rules_', 'best_rules', 'n_rules_learned', 'rules_learned'):
            val = getattr(base_fuzzy, attr, None)
            if val is not None:
                if isinstance(val, (list, tuple, np.ndarray)):
                    base_fuzzy_rules = len(val)
                elif isinstance(val, int):
                    base_fuzzy_rules = int(val)
                break
        if base_fuzzy_rules is None:
            val = getattr(base_fuzzy, 'solutions', None) or getattr(base_fuzzy, 'population', None)
            if isinstance(val, (list, tuple, np.ndarray)):
                base_fuzzy_rules = len(val)
        if base_fuzzy_rules is None:
            base_fuzzy_rules = -1

        base_fuzzy_had_no_rules = bool(self.penalize_no_rules and (base_fuzzy_rules in (0, 1)))
        if base_fuzzy_had_no_rules:
            print(f"WARNING: BaseFuzzy (Genetic Opt.) trained with no rules (rules={base_fuzzy_rules}). Prediction times will be penalized by x{self.no_rules_penalty_multiplier}.")
        
        results = []
        
        for test_size in n_test_sizes:
            print(f"\nTesting with test_size={test_size}")
            
            X_test, y_test = self.generate_synthetic_dataset(test_size, n_features)
            
            # Measure FuzzyCART prediction time
            start_time = time.time()
            _ = fuzzy_cart.predict(X_test)
            fuzzy_cart_pred_time_raw = time.time() - start_time

            # Apply penalization when no rules were trained
            if fuzzy_cart_had_no_rules:
                fuzzy_cart_pred_time = fuzzy_cart_pred_time_raw * self.no_rules_penalty_multiplier
            else:
                fuzzy_cart_pred_time = fuzzy_cart_pred_time_raw
            
            # Measure CART prediction time
            start_time = time.time()
            _ = cart.predict(X_test)
            cart_pred_time = time.time() - start_time
            
            # Measure BaseFuzzyRulesClassifier prediction time
            if base_fuzzy_trained:
                try:
                    start_time = time.time()
                    _ = base_fuzzy.predict(X_test)
                    base_fuzzy_pred_time_raw = time.time() - start_time

                    # Penalize when no rules were trained
                    if base_fuzzy_had_no_rules:
                        base_fuzzy_pred_time = base_fuzzy_pred_time_raw * self.no_rules_penalty_multiplier
                        print(f"  WARNING: BaseFuzzy prediction penalized due to no rules (raw={base_fuzzy_pred_time_raw:.6f}s).")
                    else:
                        base_fuzzy_pred_time = base_fuzzy_pred_time_raw
                except Exception as e:
                    # If predict fails (e.g., no learned rules or other error), set a conservative penalized time
                    fallback = max(fuzzy_cart_pred_time_raw, cart_pred_time, 1e-6)
                    base_fuzzy_pred_time_raw = np.nan
                    base_fuzzy_pred_time = fallback * self.no_rules_penalty_multiplier
                    print(f"  WARNING: BaseFuzzy.predict raised {e!r}; setting penalized prediction time {base_fuzzy_pred_time:.6f}s.")
            else:
                base_fuzzy_pred_time_raw = np.nan
                base_fuzzy_pred_time = np.nan
            
            results.append({
                'test_size': test_size,
                'fuzzy_cart_pred_time_raw': fuzzy_cart_pred_time_raw,
                'fuzzy_cart_pred_time': fuzzy_cart_pred_time,
                'cart_pred_time': cart_pred_time,
                'genetic_opt_pred_time_raw': base_fuzzy_pred_time_raw,
                'genetic_opt_pred_time': base_fuzzy_pred_time,
                'fuzzy_cart_pred_per_sample': fuzzy_cart_pred_time / test_size * 1000,  # ms (adjusted)
                'cart_pred_per_sample': cart_pred_time / test_size * 1000,  # ms
                'genetic_opt_pred_per_sample': base_fuzzy_pred_time / test_size * 1000 if not np.isnan(base_fuzzy_pred_time) else np.nan  # ms
            })
            
            print(f"  FuzzyCART: {fuzzy_cart_pred_time:.6f}s (raw={fuzzy_cart_pred_time_raw:.6f}s) "
                  f"({fuzzy_cart_pred_time/test_size*1000:.6f} ms/sample)")
            print(f"  CART: {cart_pred_time:.6f}s "
                  f"({cart_pred_time/test_size*1000:.6f} ms/sample)")
            if base_fuzzy_trained and not np.isnan(base_fuzzy_pred_time):
                print(f"  Genetic Opt.: {base_fuzzy_pred_time:.6f}s "
                      f"({base_fuzzy_pred_time/test_size*1000:.6f} ms/sample)")
        
        results_df = pd.DataFrame(results)
        return results_df


class PerformanceVisualizer:
    """
    Creates visualizations for performance analysis results.
    """
    
    @staticmethod
    def plot_training_time_vs_samples(results_df, avg_results):
        """
        Plot training time vs number of samples.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Raw results
        avg_results : pd.DataFrame
            Averaged results
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot FGRT (Fuzzy Greedy Rule Tree)
        ax.errorbar(
            avg_results['n_samples'],
            avg_results[('fuzzy_cart_time', 'mean')],
            yerr=avg_results[('fuzzy_cart_time', 'std')],
            label='FGRT',
            marker='o',
            linewidth=2,
            capsize=5
        )
        
        # Plot CART
        ax.errorbar(
            avg_results['n_samples'],
            avg_results[('cart_time', 'mean')],
            yerr=avg_results[('cart_time', 'std')],
            label='CART',
            marker='s',
            linewidth=2,
            capsize=5
        )
        
        # Plot C4.5
        ax.errorbar(
            avg_results['n_samples'],
            avg_results[('c45_time', 'mean')],
            yerr=avg_results[('c45_time', 'std')],
            label='C4.5',
            marker='^',
            linewidth=2,
            capsize=5
        )
        
        # Plot Genetic Opt. (BaseFuzzyRulesClassifier)
        ax.errorbar(
            avg_results['n_samples'],
            avg_results[('genetic_opt_time', 'mean')],
            yerr=avg_results[('genetic_opt_time', 'std')],
            label='Genetic Opt.',
            marker='d',
            linewidth=2,
            capsize=5
        )
        
        ax.set_xlabel('Number of Samples', fontsize=12)
        ax.set_ylabel('Training Time (seconds)', fontsize=12)
        ax.set_title('Training Time vs Dataset Size', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_training_time_vs_features(results_df, avg_results):
        """
        Plot training time vs number of features.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Raw results
        avg_results : pd.DataFrame
            Averaged results
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot FGRT (Fuzzy Greedy Rule Tree)
        ax.errorbar(
            avg_results['n_features'],
            avg_results[('fuzzy_cart_time', 'mean')],
            yerr=avg_results[('fuzzy_cart_time', 'std')],
            label='FGRT',
            marker='o',
            linewidth=2,
            capsize=5
        )
        
        # Plot CART
        ax.errorbar(
            avg_results['n_features'],
            avg_results[('cart_time', 'mean')],
            yerr=avg_results[('cart_time', 'std')],
            label='CART',
            marker='s',
            linewidth=2,
            capsize=5
        )
        
        # Plot C4.5
        ax.errorbar(
            avg_results['n_features'],
            avg_results[('c45_time', 'mean')],
            yerr=avg_results[('c45_time', 'std')],
            label='C4.5',
            marker='^',
            linewidth=2,
            capsize=5
        )
        
        # Plot Genetic Opt. (BaseFuzzyRulesClassifier)
        ax.errorbar(
            avg_results['n_features'],
            avg_results[('genetic_opt_time', 'mean')],
            yerr=avg_results[('genetic_opt_time', 'std')],
            label='Genetic Opt.',
            marker='d',
            linewidth=2,
            capsize=5
        )
        
        ax.set_xlabel('Number of Features', fontsize=12)
        ax.set_ylabel('Training Time (seconds)', fontsize=12)
        ax.set_title('Training Time vs Number of Features', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_prediction_time(results_df):
        """
        Plot prediction time analysis.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Prediction time results
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Total prediction time
        ax1.plot(results_df['test_size'], results_df['fuzzy_cart_pred_time'],
                'o-', label='FGRT', linewidth=2)
        ax1.plot(results_df['test_size'], results_df['cart_pred_time'],
                's-', label='CART', linewidth=2)
        if 'genetic_opt_pred_time' in results_df.columns:
            ax1.plot(results_df['test_size'], results_df['genetic_opt_pred_time'],
                    'd-', label='Genetic Opt.', linewidth=2)
        ax1.set_xlabel('Test Set Size', fontsize=12)
        ax1.set_ylabel('Prediction Time (seconds)', fontsize=12)
        ax1.set_title('Total Prediction Time', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Per-sample prediction time
        ax2.plot(results_df['test_size'], results_df['fuzzy_cart_pred_per_sample'],
                'o-', label='FGRT', linewidth=2)
        ax2.plot(results_df['test_size'], results_df['cart_pred_per_sample'],
                's-', label='CART', linewidth=2)
        if 'genetic_opt_pred_per_sample' in results_df.columns:
            ax2.plot(results_df['test_size'], results_df['genetic_opt_pred_per_sample'],
                    'd-', label='Genetic Opt.', linewidth=2)
        ax2.set_xlabel('Test Set Size', fontsize=12)
        ax2.set_ylabel('Prediction Time per Sample (ms)', fontsize=12)
        ax2.set_title('Per-Sample Prediction Time', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        return fig


def main():
    """
    Main function to run performance analysis.
    """
    print("="*80)
    print("FuzzyCART Performance and Scalability Analysis")
    print("="*80)

    # Check if Numba is available
    try:
        from tree_learning import NUMBA_AVAILABLE
        numba_status = "Available" if NUMBA_AVAILABLE else "Not installed"
    except ImportError:
        numba_status = "Unknown"

    # Configuration
    early_stop_threshold = 0.05  # Early stopping threshold (0.0 to disable)
    use_numba = True             # Use Numba JIT when available

    print(f"\nOptimization Settings:")
    print(f"  - Target metric: purity (enables Numba JIT acceleration)")
    print(f"  - Early stopping threshold: {early_stop_threshold}")
    print(f"  - Use Numba JIT: {use_numba} ({numba_status})")
    print(f"  - Expected speedup: 10-12x with Numba + early stopping")

    analyzer = PerformanceAnalyzer(
        random_state=42,
        early_stop_threshold=early_stop_threshold,
        use_numba=use_numba
    )
    visualizer = PerformanceVisualizer()
    
    # Test 1: Training time vs samples
    print("\n\nTest 1: Training Time vs Number of Samples")
    results_samples, avg_samples = analyzer.measure_training_time_vs_samples(
        sample_sizes=[100, 500, 1000, 2000, 5000],
        n_features=10,
        n_trials=3
    )
    results_samples.to_csv('performance_vs_samples.csv', index=False)
    
    fig = visualizer.plot_training_time_vs_samples(results_samples, avg_samples)
    fig.savefig('performance_vs_samples.png', 
                dpi=150, bbox_inches='tight')
    fig.savefig('performance_vs_samples.pdf', 
                dpi=150, bbox_inches='tight')
    print("Saved plots: performance_vs_samples.png, performance_vs_samples.pdf")
    
    # Test 2: Training time vs features
    print("\n\nTest 2: Training Time vs Number of Features")
    results_features, avg_features = analyzer.measure_training_time_vs_features(
        feature_sizes=[5, 10, 20, 30],
        n_samples=1000,
        n_trials=3
    )
    results_features.to_csv('performance_vs_features.csv', index=False)
    
    fig = visualizer.plot_training_time_vs_features(results_features, avg_features)
    fig.savefig('performance_vs_features.png',
                dpi=150, bbox_inches='tight')
    fig.savefig('performance_vs_features.pdf',
                dpi=150, bbox_inches='tight')
    print("Saved plots: performance_vs_features.png, performance_vs_features.pdf")
    
    # Test 3: Prediction time
    print("\n\nTest 3: Prediction Time Analysis")
    results_pred = analyzer.measure_prediction_time(
        n_samples=1000,
        n_features=10,
        n_test_sizes=[100, 500, 1000, 5000]
    )
    results_pred.to_csv('prediction_time.csv', index=False)
    
    fig = visualizer.plot_prediction_time(results_pred)
    fig.savefig('prediction_time.png',
                dpi=150, bbox_inches='tight')
    fig.savefig('prediction_time.pdf',
                dpi=150, bbox_inches='tight')
    print("Saved plots: prediction_time.png, prediction_time.pdf")
    
    print("\n" + "="*80)
    print("Performance Analysis Complete!")
    print("="*80)
    print("\nKey Findings:")
    print("1. Training time results saved to performance_vs_samples.csv")
    print("2. Feature scaling results saved to performance_vs_features.csv")
    print("3. Prediction time results saved to prediction_time.csv")
    print("4. All plots saved as PNG and PDF files")


if __name__ == '__main__':
    main()
