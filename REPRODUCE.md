# Reproducing the FERL paper

Everything below runs from the repository root on a CPU. The three variants of the
paper are:

| Paper | Code |
|---|---|
| FERL-compact | `ferl.pipeline.make("ferl-compact")` (benchmark key `FERL-compact`) |
| FERL-medium | `make("ferl-medium")` (benchmark key `FERL-medium`) |
| FERL-deep | `ferl.core.learned_tree.LearnedFuzzyTree()` (`make("ferl-deep")`, benchmark key `FERL-deep`) |

`tests/test_variant_matrix.py` checks that these configurations match Table 1 of the paper.

## 1. Environment and data

```bash
make env                       # Python 3.12 venv with the pinned requirements-lock.txt (CPU torch)
make data KEEL_DIR=../keel_datasets   # downloads the 30 KEEL datasets and verifies SHA-256 checksums
export KEEL_DIR=../keel_datasets
make test PY=.venv/bin/python  # variant table == code
```

`ex-fuzzy` is pinned to the commit used for the runs (a git URL in
`requirements-lock.txt`); PyPI's `ex-fuzzy==3.1.0` is equivalent for FERL's use.
The concept-bottleneck experiments need the detector outputs in
`results/cub_cbm_artifacts/` and `results/awa2_cbm_artifacts/`. They are not in this
repository (about 900 MB). The scripts `experiments/cub_cbm/*_cluster.sh` and
`*_gpu.sh` regenerate them; training the detectors needs a GPU.

The compiled kernels in `ferl_fast/` are optional (they are required only for the
runtime table): `python ferl_fast/setup.py build_ext --inplace`.

## 2. Rebuild the tables and figures from the stored results (minutes)

```bash
make tables PY=.venv/bin/python    # tables and macros in paper/generated/, figures in paper/figures/
```

Every number in the text is either in a generated table or a macro in
`paper/generated/*.tex`. No figure or table is edited by hand, except the variant
and protocol tables (Tables 1 and 3), which the test and the scripts below document.
This repository ships the result files that `make tables` reads. The per-fold
predictions, split indices and manifests (`results/bench*/`) are not in this
repository; `make tabular` regenerates them.

## 3. Rerun the experiments (CPU hours)

| Paper item | Command | Result file(s) | Table builder |
|---|---|---|---|
| Table 4 (accuracy, AURC), Fig. 2 (CD), Table C2, Table D3, Table 5 (sets) | `make tabular` | `results/benchmark2_per_fold.csv`; per-fold predictions, indices and manifests in `results/bench/` (FERL-compact, its DS read-out and FERL-medium in `results/bench_rev/`) | `make_results_assets.py`, `make_revision_assets.py` |
| FERL-deep, tuned width (Tables 4, 5) | `python experiments/benchmark2/harness.py --models FERL-deep-tuned --out-dir results/bench_tuned`, then `python experiments/benchmark2/score.py --bench-dir results/bench_tuned --replace-models FERL-deep-tuned` | rows `FERL-deep-tuned` in `results/benchmark2_per_fold.csv`; chosen widths in the manifests | `make_results_assets.py`, `make_revision_assets.py` |
| Appendix (encoding of nominal attributes) | `FERL_NOMINAL=onehot python experiments/benchmark2/harness.py --datasets german australian crx --models <methods> --out-dir results/bench_oh`, then `python experiments/benchmark2/score.py --bench-dir results/bench_oh --per-fold-csv results/benchmark2_onehot_per_fold.csv --summary-csv results/benchmark2_onehot_summary.csv` | `results/benchmark2_onehot_per_fold.csv` | `make_revision_assets.py` |
| Fig. 3 (matched budgets) | `python experiments/reliability/ds_ignorance_semantics.py --study depth` | `results/ignorance_depth.csv` | `make_depth_figure.py` |
| Table 2 (stability, explanation size) | `python experiments/reliability/ds_stability_constants.py` | `results/stability_constants.csv` | `make_revision_assets.py` |
| Table 6 (tabular near-OOD, EDL) | `make ood` | `results/ds_ood_residual_{ferl-compact,ferl-medium,ferl-deep}.csv` | `make_revision_assets.py` |
| Table 7 (CBM accuracy) | `experiments/cub_cbm/cub_cluster.sh`, `awa2_cluster.sh` (E7) | `results/{cub,awa2}_cbm_perf/e7_calibration.csv` | `make_results_assets.py` |
| Table 8 (open-world CBM) | `experiments/cub_cbm/open_world_*_gpu.sh`, then `open_world_eval_cluster.sh` | `results/cbm_open_world/e16_open_world_ood.csv` | `make_cbm_open_world_assets.py` |
| Table 9 (concept errors) | `make cbm` | `results/cub_cbm_perf/concept_error_robustness_{full,20}.csv` | `make_revision_assets.py` |
| Fig. 4 (CUB case study) | `python experiments/paper_assets/make_cbm_case_study.py` (needs the CUB images; set `CUB_ROOT`) | `results/paper_figures/cbm_case_study.pdf` | -- |
| Table 10 (traced wine predictions) | `python experiments/paper_assets/make_traced_example.py` | `results/traced_example.json` | same script |
| Table 11 (read-out ablation) | `python experiments/reliability/ds_decision_rule_ablation.py` | `results/decision_rule_ablation.csv` | `make_revision_assets.py` |
| Table 12 (band width) | `python experiments/reliability/ds_width_ablation.py` | `results/width_ablation.csv` | `make_revision_assets.py` |
| Table 13 (bounded support) | `python experiments/reliability/ds_bounded_support_ablation.py` | `results/bounded_support_{id,ood}.csv` | `make_bounded_support_ablation.py` |
| Table 14 (CCI vs Gini) | `python experiments/paper_assets/make_cci_gini_learned_ablation.py` | `results/cci_gini_learned_ablation.csv` | same script (`--from-csv` to skip refits) |
| Tables 15--16 (ignorance semantics) | `python experiments/reliability/ds_ignorance_semantics.py --study noop` | `results/ignorance_noop.csv` | `make_revision_assets.py` |
| Runtime table | `make runtime` (on an idle machine) | `results/runtime_scaling/` | `wallclock_scaling.py` |
| Appendix E (covariate shift) | `python experiments/reliability/ds_credal_shift.py`; `python experiments/benchmark2/credal_stress.py` | `results/credal_shift.csv`, `results/real_shift.csv` | `make_results_assets.shift_assets` |
| Appendix F (refitted partitions) | `python experiments/benchmark/suarez_lutsko_ablation.py full` | `results/suarez_lutsko_ablation.csv` | numbers quoted in text |
| Appendix B (datasets) | -- | KEEL headers | `make_dataset_list.py` |

External baselines (RRL, RL-Net, SamRuLe, FUCS, NeuRules) run from their public code
at the revisions recorded in `experiments/benchmark2/harness.py` (`SOURCE_REVISIONS`,
`PUBLIC_REPOS`); set the environment variables named there to their checkouts. FUCS
needs a Java runtime.

## 4. Protocols and seeds

* Nominal KEEL attributes are integer-coded (the benchmark's encoding) for every
  method; set `FERL_NOMINAL=onehot` to one-hot encode them (used only for the
  encoding-sensitivity appendix).

* Tabular folds: `StratifiedKFold(5, shuffle=True, random_state=33)`. In each outer
  training fold, 25% is held out with `train_test_split(..., random_state=fold)`.
  FERL's native outputs ignore this calibration split; conformal baselines use it.
  The harness stores the exact indices for every dataset, fold and model.
* Tabular near-OOD: each class held out in turn; retained classes split 70/30 with
  seeds 0--2; training capped at 2,000 samples; global NumPy RNG seeded per fit.
* FERL-deep, FERL-compact and every experiment script are deterministic.
  FERL-medium draws its bootstrap indices from NumPy's global generator. The stored
  benchmark runs did not seed it, so refitting FERL-medium reproduces its
  reported numbers only up to small run-to-run variation.
