# CLAUDE — Understanding this Repository

Purpose
-------
This file is written to help Claude (or other LLMs) quickly understand the repository scope and how to help with code, experiments, and reproducibility. The project implements a fuzzy decision-tree algorithm (FuzzyCART), partition optimization strategies (grid / coordinate / hybrid), and a performance analysis suite used for a research paper.

Key concepts to know
---------------------
- FuzzyCART: A fuzzy extension of CART that uses fuzzy partitions for splits and evaluates splits using metrics such as CCI (Complete Classification Index) and weighted Gini.
- Partition optimization: Search strategies (`grid`, `coordinate`, `hybrid`) for tuning fuzzy partition parameters to improve separability.
- Baselines included: CART (DecisionTreeClassifier from scikit-learn), C4.5 (entropy criterion), and a genetic/evolutionary fuzzy rules baseline (named in code as `BaseFuzzyRulesClassifier` and displayed as "Genetic Opt.").

Where timing / runtime comparisons happen
---------------------------------------
- `examples/strategy_comparison_example.py`: computes optimization times (`opt_time`) per strategy and prints speed-up ratios.
- `visualize_strategies.py`: plots optimization times and time vs improvement trade-offs (bar + scatter plots).
- `performance_analysis.py`: `PerformanceAnalyzer` measures training time vs samples, training time vs features, and prediction time. This is the authoritative place for runtime and memory analysis.

Important files
---------------
- `tree_learning.py` — FuzzyCART implementation (training, splitting, prediction). Hotspots: split evaluation loops, membership computations, prediction loops.
- `partition_optimization.py` — partition encoding and search strategies (grid, coordinate, hybrid).
- `performance_analysis.py` — performance tests, measures, and plotting utilities.
- `visualize_strategies.py` and `examples/strategy_comparison_example.py` — reproducible examples and visuals.
- `requirements.txt` — Python dependencies.
- `results/` — generated CSVs and saved plots used in manuscript tables / figures.

How to run common tasks
-----------------------
- Quick example: `python examples/strategy_comparison_example.py`
- Performance suite: `python performance_analysis.py` (generates CSVs and plots)
- Visualizations: `python visualize_strategies.py` or run scripts that call the visualizer
- Tests: `pytest -q` (project includes unit tests)

Reproducibility notes
---------------------
- Many scripts use `random_state` in calls and the `PerformanceAnalyzer` supports `n_trials` to average times.
- For large datasets, `FuzzyCART` supports sampling of candidate splits (`sample_for_splits`, `sample_size`).

Suggested naming conventions & current mapping
---------------------------------------------
- `BaseFuzzyRulesClassifier` → displayed as **Genetic Opt.** in plots and results (so CSV headers, legends, and prints were updated).


Security, privacy, and license
------------------------------
- No user PII or private data is included in the repo. Example datasets use scikit-learn or synthetic data.
- The project uses the repository license (`LICENSE` file). For public distribution, ensure code and figures follow the license terms.



_File created to help Claude quickly be effective at reviewing, debugging, and optimizing this research codebase._
