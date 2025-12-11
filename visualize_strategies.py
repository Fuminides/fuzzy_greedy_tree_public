"""
Visualize Search Strategy Comparison
=====================================

Creates a comprehensive visualization comparing the three search strategies.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import time

from tree_learning import FuzzyCART
from partition_optimization import optimize_partitions_for_gfrt
from ex_fuzzy import utils

# Load data
X, y = load_iris(return_X_y=True)
default_partitions = utils.construct_partitions(X)

# Test strategies
strategies = ['grid', 'coordinate', 'hybrid']
results = {}

print("Running optimization tests...")
for strategy in strategies:
    print(f"  Testing {strategy}...")
    
    start_time = time.time()
    optimized_partitions = optimize_partitions_for_gfrt(
        X, y,
        initial_partitions=default_partitions,
        method='separability',
        strategy=strategy,
        verbose=False
    )
    optimization_time = time.time() - start_time
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_default = []
    scores_optimized = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model_default = FuzzyCART(fuzzy_partitions=default_partitions, max_depth=5)
        model_default.fit(X_train, y_train)
        scores_default.append(accuracy_score(y_test, model_default.predict(X_test)))
        
        model_optimized = FuzzyCART(fuzzy_partitions=optimized_partitions, max_depth=5)
        model_optimized.fit(X_train, y_train)
        scores_optimized.append(accuracy_score(y_test, model_optimized.predict(X_test)))
    
    results[strategy] = {
        'time': optimization_time,
        'default': np.array(scores_default),
        'optimized': np.array(scores_optimized),
        'improvement': np.mean(scores_optimized) - np.mean(scores_default)
    }

# Create visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Title
fig.suptitle('Fuzzy Partition Optimization: Search Strategy Comparison', 
             fontsize=18, fontweight='bold', y=0.98)

# 1. Optimization Time Comparison
ax1 = fig.add_subplot(gs[0, 0])
times = [results[s]['time'] for s in strategies]
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax1.bar(strategies, times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_title('Optimization Time', fontsize=13, fontweight='bold', pad=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
for bar, time_val in zip(bars, times):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{time_val:.3f}s', ha='center', va='bottom', fontweight='bold')

# 2. Accuracy Comparison
ax2 = fig.add_subplot(gs[0, 1])
x_pos = np.arange(len(strategies))
default_means = [np.mean(results[s]['default']) for s in strategies]
optimized_means = [np.mean(results[s]['optimized']) for s in strategies]
default_stds = [np.std(results[s]['default']) for s in strategies]
optimized_stds = [np.std(results[s]['optimized']) for s in strategies]

width = 0.35
ax2.bar(x_pos - width/2, default_means, width, label='Default', 
        yerr=default_stds, capsize=5, color='#95a5a6', alpha=0.7, edgecolor='black')
ax2.bar(x_pos + width/2, optimized_means, width, label='Optimized',
        yerr=optimized_stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax2.set_title('Cross-Validation Accuracy', fontsize=13, fontweight='bold', pad=10)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(strategies)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim([0.85, 1.0])

# 3. Improvement Percentage
ax3 = fig.add_subplot(gs[0, 2])
improvements = [results[s]['improvement'] * 100 for s in strategies]
bars = ax3.bar(strategies, improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
ax3.set_title('Accuracy Change', fontsize=13, fontweight='bold', pad=10)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
for bar, imp in zip(bars, improvements):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{imp:+.2f}%', ha='center', va='bottom' if imp > 0 else 'top', fontweight='bold')

# 4-6. Fold-by-fold comparison for each strategy
for idx, strategy in enumerate(strategies):
    ax = fig.add_subplot(gs[1, idx])
    folds = np.arange(1, 6)
    ax.plot(folds, results[strategy]['default'], 'o-', label='Default', 
            linewidth=2, markersize=8, color='#95a5a6')
    ax.plot(folds, results[strategy]['optimized'], 's-', label='Optimized',
            linewidth=2, markersize=8, color=colors[idx])
    ax.set_xlabel('Fold', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title(f'{strategy.upper()}: Fold-by-Fold', fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.85, 1.0])
    ax.set_xticks(folds)

# 7. Time/Improvement Trade-off
ax7 = fig.add_subplot(gs[2, 0])
for idx, strategy in enumerate(strategies):
    ax7.scatter(results[strategy]['time'], results[strategy]['improvement'] * 100,
               s=300, color=colors[idx], alpha=0.7, edgecolor='black', linewidth=2)
    ax7.text(results[strategy]['time'], results[strategy]['improvement'] * 100,
            strategy.upper(), ha='center', va='center', fontweight='bold', fontsize=9)
ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax7.set_xlabel('Optimization Time (s)', fontsize=12, fontweight='bold')
ax7.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
ax7.set_title('Time vs. Improvement Trade-off', fontsize=13, fontweight='bold', pad=10)
ax7.grid(True, alpha=0.3)

# 8. Box plot comparison
ax8 = fig.add_subplot(gs[2, 1:])
data_to_plot = []
labels = []
for strategy in strategies:
    data_to_plot.append(results[strategy]['default'])
    labels.append(f'{strategy.upper()}\n(Default)')
    data_to_plot.append(results[strategy]['optimized'])
    labels.append(f'{strategy.upper()}\n(Optimized)')

bp = ax8.boxplot(data_to_plot, labels=labels, patch_artist=True)
for idx, patch in enumerate(bp['boxes']):
    if idx % 2 == 0:  # Default
        patch.set_facecolor('#95a5a6')
        patch.set_alpha(0.5)
    else:  # Optimized
        patch.set_facecolor(colors[idx // 2])
        patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

ax8.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax8.set_title('Distribution Comparison (5-Fold CV)', fontsize=13, fontweight='bold', pad=10)
ax8.grid(axis='y', alpha=0.3, linestyle='--')
ax8.set_ylim([0.85, 1.0])
plt.setp(ax8.get_xticklabels(), fontsize=9)

# Add summary text
summary_text = f"""Dataset: Iris (150 samples, 4 features, 3 classes)
Optimization Metric: Separability Index
Cross-Validation: 5-Fold Stratified

Key Findings:
• Fastest: {min(strategies, key=lambda s: results[s]['time']).upper()} ({min(times):.3f}s)
• Best Accuracy: {max(strategies, key=lambda s: np.mean(results[s]['optimized'])).upper()} ({max(optimized_means):.4f})
• Most Consistent: Default partitions ({np.mean([np.std(results[s]['default']) for s in strategies]):.4f} avg std)
"""

fig.text(0.02, 0.02, summary_text, fontsize=10, family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Save figure
plt.savefig('search_strategy_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as: search_strategy_comparison.png")

plt.show()
