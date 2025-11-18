"""
Main Test Runner for FuzzyCART Experiments
==========================================

This script orchestrates all experimental components:
1. Basic functionality tests
2. Benchmark comparisons
3. Ablation studies
4. Performance analysis
5. Statistical analysis

Usage:
    python test_runner.py --all                # Run all tests
    python test_runner.py --benchmarks         # Run benchmark comparisons only
    python test_runner.py --ablation           # Run ablation studies only
    python test_runner.py --performance        # Run performance analysis only
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project directory to path
sys.path.append('/mnt/project')
sys.path.append('/home/claude')

from tree_learning import FuzzyCART
try:
    from ex_fuzzy import fuzzy_sets as fs
except ImportError:
    import fuzzy_sets as fs


class TestRunner:
    """
    Main test runner that orchestrates all FuzzyCART experiments.
    """
    
    def __init__(self, output_dir='/mnt/user-data/outputs'):
        """
        Initialize test runner.
        
        Parameters
        ----------
        output_dir : str
            Directory to save all outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        
    def run_basic_tests(self):
        """
        Run basic functionality tests to ensure FuzzyCART works.
        """
        print("\n" + "="*80)
        print("Running Basic Functionality Tests")
        print("="*80)
        
        from sklearn.datasets import load_iris, load_wine
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        test_results = []
        
        # Test 1: Iris dataset
        print("\nTest 1: Iris Dataset")
        data = load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Generate fuzzy partitions
        fuzzy_partitions = self._generate_partitions(X_train)
        
        try:
            fuzzy_cart = FuzzyCART(
                fuzzy_partitions=fuzzy_partitions,
                max_rules=10,
                coverage_threshold=0.01,
                min_improvement=0.01
            )
            
            fuzzy_cart.fit(X_train, y_train)
            y_pred = fuzzy_cart.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            stats = fuzzy_cart.get_tree_stats() if hasattr(fuzzy_cart, 'get_tree_stats') else {}
            
            print(f"  ✓ Training successful")
            print(f"  ✓ Accuracy: {accuracy:.4f}")
            print(f"  ✓ Number of rules: {stats.get('leaves', 'N/A')}")
            
            test_results.append({
                'test': 'iris',
                'status': 'PASS',
                'accuracy': accuracy,
                'n_rules': stats.get('leaves', 0)
            })
            
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            test_results.append({
                'test': 'iris',
                'status': 'FAIL',
                'error': str(e)
            })
        
        # Test 2: Wine dataset
        print("\nTest 2: Wine Dataset")
        data = load_wine()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        fuzzy_partitions = self._generate_partitions(X_train)
        
        try:
            fuzzy_cart = FuzzyCART(
                fuzzy_partitions=fuzzy_partitions,
                max_rules=15,
                coverage_threshold=0.01,
                min_improvement=0.01
            )
            
            fuzzy_cart.fit(X_train, y_train)
            y_pred = fuzzy_cart.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            stats = fuzzy_cart.get_tree_stats() if hasattr(fuzzy_cart, 'get_tree_stats') else {}
            
            print(f"  ✓ Training successful")
            print(f"  ✓ Accuracy: {accuracy:.4f}")
            print(f"  ✓ Number of rules: {stats.get('leaves', 'N/A')}")
            
            test_results.append({
                'test': 'wine',
                'status': 'PASS',
                'accuracy': accuracy,
                'n_rules': stats.get('leaves', 0)
            })
            
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            test_results.append({
                'test': 'wine',
                'status': 'FAIL',
                'error': str(e)
            })
        
        # Save results
        results_df = pd.DataFrame(test_results)
        results_df.to_csv(self.output_dir / 'basic_tests.csv', index=False)
        
        print("\n" + "="*60)
        print("Basic Tests Summary:")
        print(results_df)
        
        self.results['basic_tests'] = results_df
        
        return results_df
    
    def run_benchmark_comparisons(self):
        """
        Run full benchmark comparisons as described in the paper.
        """
        print("\n" + "="*80)
        print("Running Benchmark Comparisons")
        print("="*80)
        
        from fuzzy_cart_experiments import (
            DatasetLoader, ExperimentRunner, StatisticalAnalyzer
        )
        
        # Use a subset of datasets for demonstration
        test_datasets = ['iris', 'wine']
        
        loader = DatasetLoader()
        datasets = loader.load_all_datasets(test_datasets)
        
        if len(datasets) == 0:
            print("No datasets loaded. Skipping benchmark comparisons.")
            return None
        
        runner = ExperimentRunner(n_folds=5, random_state=42)
        results_df, all_results = runner.run_all_experiments(datasets)
        
        # Save results
        results_df.to_csv(self.output_dir / 'benchmark_results.csv', index=False)
        
        # Statistical analysis
        if len(datasets) > 1:
            analyzer = StatisticalAnalyzer()
            analyzer.friedman_test(results_df)
        
        # Create summary
        summary = results_df.groupby('Method').agg({
            'Accuracy': ['mean', 'std'],
            'N_Rules': 'mean',
            'N_Conditions': 'mean'
        })
        
        summary.to_csv(self.output_dir / 'benchmark_summary.csv')
        print("\n" + "="*60)
        print("Benchmark Summary:")
        print(summary)
        
        self.results['benchmarks'] = results_df
        
        return results_df
    
    def run_ablation_studies(self):
        """
        Run ablation studies to analyze parameter sensitivity.
        """
        print("\n" + "="*80)
        print("Running Ablation Studies")
        print("="*80)
        
        from ablation_studies import AblationStudy, ParameterOptimizer
        from sklearn.datasets import load_iris
        
        # Load test dataset
        data = load_iris()
        X, y = data.data, data.target
        
        fuzzy_partitions = self._generate_partitions(X)
        
        # Run ablation studies
        ablation = AblationStudy(X, y, fuzzy_partitions, n_folds=3)
        
        # Test max_rules
        print("\n" + "-"*60)
        results_max_rules = ablation.test_max_rules_impact(
            max_rules_values=[5, 10, 15, 20]
        )
        results_max_rules.to_csv(self.output_dir / 'ablation_max_rules.csv', index=False)
        
        # Test coverage_threshold
        print("\n" + "-"*60)
        results_coverage = ablation.test_coverage_threshold_impact(
            threshold_values=[0.0, 0.01, 0.03, 0.05]
        )
        results_coverage.to_csv(self.output_dir / 'ablation_coverage.csv', index=False)
        
        # Parameter optimization
        print("\n" + "-"*60)
        optimizer = ParameterOptimizer(X, y, fuzzy_partitions, n_folds=3)
        opt_results = optimizer.grid_search(param_grid={
            'max_rules': [10, 15],
            'coverage_threshold': [0.01, 0.03],
            'min_improvement': [0.01, 0.02]
        })
        
        opt_results['all_results'].to_csv(
            self.output_dir / 'parameter_optimization.csv', index=False
        )
        
        print("\n" + "="*60)
        print(f"Best Parameters: {opt_results['best_params']}")
        print(f"Best Score: {opt_results['best_score']:.4f}")
        
        self.results['ablation'] = {
            'max_rules': results_max_rules,
            'coverage': results_coverage,
            'optimization': opt_results
        }
        
        return self.results['ablation']
    
    def run_performance_analysis(self):
        """
        Run performance and scalability analysis.
        """
        print("\n" + "="*80)
        print("Running Performance Analysis")
        print("="*80)
        
        from performance_analysis import PerformanceAnalyzer, PerformanceVisualizer
        
        analyzer = PerformanceAnalyzer(random_state=42)
        visualizer = PerformanceVisualizer()
        
        # Test 1: Scalability with sample size
        print("\n" + "-"*60)
        results_samples, avg_samples = analyzer.measure_training_time_vs_samples(
            sample_sizes=[100, 500, 1000, 2000],
            n_features=10,
            n_trials=2
        )
        results_samples.to_csv(self.output_dir / 'performance_vs_samples.csv', index=False)
        
        fig = visualizer.plot_training_time_vs_samples(results_samples, avg_samples)
        fig.savefig(self.output_dir / 'performance_vs_samples.png',
                   dpi=150, bbox_inches='tight')
        
        # Test 2: Scalability with features
        print("\n" + "-"*60)
        results_features, avg_features = analyzer.measure_training_time_vs_features(
            feature_sizes=[5, 10, 20],
            n_samples=1000,
            n_trials=2
        )
        results_features.to_csv(self.output_dir / 'performance_vs_features.csv', index=False)
        
        fig = visualizer.plot_training_time_vs_features(results_features, avg_features)
        fig.savefig(self.output_dir / 'performance_vs_features.png',
                   dpi=150, bbox_inches='tight')
        
        print("\n" + "="*60)
        print("Performance Analysis Complete!")
        print(f"Results saved to: {self.output_dir}")
        
        self.results['performance'] = {
            'samples': results_samples,
            'features': results_features
        }
        
        return self.results['performance']
    
    def _generate_partitions(self, X):
        """Helper to generate fuzzy partitions."""
        from fuzzy_cart_experiments import FuzzyPartitionGenerator
        return FuzzyPartitionGenerator.generate_partitions(X)
    
    def generate_report(self):
        """
        Generate comprehensive HTML report of all results.
        """
        print("\n" + "="*80)
        print("Generating Comprehensive Report")
        print("="*80)
        
        report_lines = []
        report_lines.append("<html><head><title>FuzzyCART Test Report</title>")
        report_lines.append("<style>")
        report_lines.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        report_lines.append("h1 { color: #333; }")
        report_lines.append("h2 { color: #666; border-bottom: 2px solid #ccc; }")
        report_lines.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
        report_lines.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        report_lines.append("th { background-color: #4CAF50; color: white; }")
        report_lines.append("</style></head><body>")
        
        report_lines.append("<h1>FuzzyCART Experimental Results Report</h1>")
        
        # Basic tests section
        if 'basic_tests' in self.results:
            report_lines.append("<h2>1. Basic Functionality Tests</h2>")
            report_lines.append(self.results['basic_tests'].to_html(index=False))
        
        # Benchmarks section
        if 'benchmarks' in self.results:
            report_lines.append("<h2>2. Benchmark Comparisons</h2>")
            report_lines.append(self.results['benchmarks'].to_html(index=False))
        
        # Ablation section
        if 'ablation' in self.results:
            report_lines.append("<h2>3. Ablation Studies</h2>")
            if 'max_rules' in self.results['ablation']:
                report_lines.append("<h3>3.1 Impact of max_rules</h3>")
                report_lines.append(self.results['ablation']['max_rules'].to_html(index=False))
            if 'coverage' in self.results['ablation']:
                report_lines.append("<h3>3.2 Impact of coverage_threshold</h3>")
                report_lines.append(self.results['ablation']['coverage'].to_html(index=False))
        
        report_lines.append("</body></html>")
        
        # Save report
        report_path = self.output_dir / 'test_report.html'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\nReport saved to: {report_path}")
        
        return report_path


def main():
    """
    Main entry point for test runner.
    """
    parser = argparse.ArgumentParser(
        description='Run FuzzyCART experiments'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Run all tests'
    )
    parser.add_argument(
        '--basic', action='store_true',
        help='Run basic functionality tests'
    )
    parser.add_argument(
        '--benchmarks', action='store_true',
        help='Run benchmark comparisons'
    )
    parser.add_argument(
        '--ablation', action='store_true',
        help='Run ablation studies'
    )
    parser.add_argument(
        '--performance', action='store_true',
        help='Run performance analysis'
    )
    parser.add_argument(
        '--output-dir', type=str, default='/mnt/user-data/outputs',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # If no specific test is selected, run all
    if not any([args.all, args.basic, args.benchmarks, args.ablation, args.performance]):
        args.all = True
    
    print("="*80)
    print("FuzzyCART Test Runner")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    
    runner = TestRunner(output_dir=args.output_dir)
    
    # Run selected tests
    if args.all or args.basic:
        runner.run_basic_tests()
    
    if args.all or args.benchmarks:
        runner.run_benchmark_comparisons()
    
    if args.all or args.ablation:
        runner.run_ablation_studies()
    
    if args.all or args.performance:
        runner.run_performance_analysis()
    
    # Generate comprehensive report
    runner.generate_report()
    
    print("\n" + "="*80)
    print("All Tests Complete!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"View the report at: {args.output_dir}/test_report.html")


if __name__ == '__main__':
    main()
