"""
Regenerate Ablation Study Figures from CSV Files
=================================================

This script loads previously saved ablation study results and regenerates
publication-quality figures without re-running the experiments.

Usage:
    python regenerate_figures.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

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


def plot_parameter_impact(ablation_df, output_prefix='ablation_study'):
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
    
    print("\nGenerating ablation study figures...")
    
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
    print(f"✓ Saved: {output_prefix}_max_rules.pdf/png")
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
    print(f"✓ Saved: {output_prefix}_coverage.pdf/png")
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
    print(f"✓ Saved: {output_prefix}_min_improvement.pdf/png")
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
    print(f"✓ Saved: {output_prefix}_heatmap.pdf/png")
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
    print(f"✓ Saved: {output_prefix}_tradeoff.pdf/png")
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
    print(f"✓ Saved: {output_prefix}_distributions.pdf/png")
    plt.close()
    
    print("\n✓ All ablation study figures regenerated successfully!")


def main():
    """Main function to regenerate figures from saved CSV files."""
    import os
    
    # Set results directory
    results_dir = 'results'
    
    print("="*80)
    print("Regenerating Ablation Study Figures")
    print("="*80)
    print(f"\nReading data from: {results_dir}/")
    print(f"Saving figures to: {results_dir}/\n")
    
    # Check if ablation CSV exists
    ablation_file = os.path.join(results_dir, 'fuzzy_cart_ablation_all_configs.csv')
    try:
        ablation_df = pd.read_csv(ablation_file)
        print(f"\n✓ Loaded ablation data: {len(ablation_df)} configurations")
        print(f"  Datasets: {ablation_df['Dataset'].nunique()}")
        print(f"  Parameters tested:")
        print(f"    - Max_Rules: {sorted(ablation_df['Max_Rules'].unique())}")
        print(f"    - Coverage_Threshold: {sorted(ablation_df['Coverage_Threshold'].unique())}")
        print(f"    - Min_Improvement: {sorted(ablation_df['Min_Improvement'].unique())}")
    except FileNotFoundError:
        print(f"\n✗ Error: {ablation_file} not found!")
        print("  Please run fuzzy_cart_experiments.py first to generate the data.")
        print(f"  Expected location: {results_dir}/fuzzy_cart_ablation_all_configs.csv")
        sys.exit(1)
    
    # Generate all figures
    plot_parameter_impact(ablation_df, output_prefix=os.path.join(results_dir, 'ablation_study'))
    
    print("\n" + "="*80)
    print("Figure Regeneration Complete!")
    print("="*80)
    print(f"\nGenerated files in {results_dir}/ directory:")
    print("  - ablation_study_max_rules.pdf/png")
    print("  - ablation_study_coverage.pdf/png")
    print("  - ablation_study_min_improvement.pdf/png")
    print("  - ablation_study_heatmap.pdf/png")
    print("  - ablation_study_tradeoff.pdf/png")
    print("  - ablation_study_distributions.pdf/png (improved box plots)")


if __name__ == '__main__':
    main()
