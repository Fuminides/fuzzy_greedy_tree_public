# CLAUDE — Understanding this Repository

Purpose
-------
This is the public code of the paper "Evidential Rule Learning for Interpretable Classification with Abstention" (FERL, Fast Evidential Rule Learning). It holds the library, the experiment harness, the result files behind the tables, and the scripts that rebuild every table and figure.

The code of the earlier paper (FGRT with partition optimization, arXiv:2512.11616) is kept under the git tag `fgrt-arxiv-2512.11616`.

Key concepts to know
---------------------
- FERL: a fuzzy rule tree whose fired rules are belief masses for Dempster-Shafer evidence. Outputs: point label, belief/plausibility, set-valued prediction (abstention), near-OOD score.
- Three paper variants: FERL-compact (`make("ferl-compact")`), FERL-medium (`make("ferl-medium")`), FERL-deep (`make("ferl-deep")`, a `LearnedFuzzyTree`). `tests/test_variant_matrix.py` checks them against Table 1 of the paper.
- `FERL-enhanced` / `ferl-enhanced` is a configuration that is not a paper variant.

Important files
---------------
- `ferl/core/tree_learning.py` — `FuzzyCART` (compact and medium variants).
- `ferl/core/learned_tree.py` — `LearnedFuzzyTree`, `LearnedFuzzyTreeCV` (deep variant).
- `ferl/pipeline/ferl_pipeline.py` — `make()` and the `CONFIGS` registry.
- `ferl_fast/` — optional Cython kernels (`python ferl_fast/setup.py build_ext --inplace`).
- `experiments/benchmark2/` — main benchmark: `harness.py` fits, `score.py` scores, `models.py` is the model registry.
- `experiments/paper_assets/` — table and figure builders. They read `results/` and write `paper/generated/` and `paper/figures/`.
- `REPRODUCE.md` — the command and result file behind each table and figure.

How to run common tasks
-----------------------
- Tests: `make test`, or `pytest -q tests`
- Tables and figures from stored results: `make tables`
- Rerun experiments: see the targets in `Makefile` (`tabular`, `ablations`, `ood`, ...)
- Run every script from the repository root. Datasets are read from `KEEL_DIR` (default `../keel_datasets`; `make data` downloads them).

Reproducibility notes
---------------------
- The stored result files use the benchmark keys `FERL-compact`, `FERL-medium`, `FERL-deep`. Do not rename keys in the code without renaming them in `results/`.
- `make tables` must give the same `paper/generated/*.tex` as the committed files.

Security, privacy, and license
------------------------------
- No user PII or private data is included in the repo.
- MIT license (`LICENSE` file).
