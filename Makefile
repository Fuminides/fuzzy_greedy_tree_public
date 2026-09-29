# Reproduce the FERL paper. See REPRODUCE.md.
# Run from the repository root. Override PY / KEEL_DIR on the command line, e.g.
#   make tables PY=.venv/bin/python KEEL_DIR=../keel_datasets
PY ?= python
KEEL_DIR ?= ../keel_datasets
export KEEL_DIR
ASSETS = experiments/paper_assets
REL = experiments/reliability

.PHONY: env data test tables tabular tuned encoding ablations diagnostics ood cbm runtime experiments

env:                 ## pinned CPU environment in .venv
	python3.12 -m venv .venv
	.venv/bin/pip install -r requirements-lock.txt
	.venv/bin/pip install -e ".[viz]"

data:                ## download and verify the 30 KEEL benchmarks
	scripts/get_keel_data.sh $(KEEL_DIR)

test:                ## the paper's variant table matches the code
	$(PY) -m pytest -q tests/test_variant_matrix.py

tables:              ## every table, figure and macro in paper/ from stored results (minutes)
	$(PY) -c "import sys; sys.path.insert(0, '$(ASSETS)'); import make_results_assets as A; A.tabular_assets(); A.significance_assets(); A.cbm_assets(); A.shift_assets()"
	$(PY) $(ASSETS)/make_dataset_list.py
	$(PY) $(ASSETS)/make_revision_assets.py
	$(PY) $(ASSETS)/make_depth_figure.py
	$(PY) $(ASSETS)/make_bounded_support_ablation.py
	$(PY) $(ASSETS)/make_cci_gini_learned_ablation.py --from-csv
	$(PY) $(ASSETS)/make_cbm_open_world_assets.py
	$(PY) $(ASSETS)/make_traced_example.py > /dev/null
	$(PY) experiments/benchmark2/wallclock_scaling.py   # resumes: rewrites the table from the stored timings

# ---- experiments (refit models; CPU hours) --------------------------------
tabular:             ## main benchmark: fit, then score (results/benchmark2_*.csv)
	$(PY) experiments/benchmark2/harness.py
	$(PY) experiments/benchmark2/score.py

tuned:               ## FERL-deep with inner-CV band width, merged into the benchmark CSV
	$(PY) experiments/benchmark2/harness.py --models FERL-deep-tuned --out-dir results/bench_tuned
	$(PY) experiments/benchmark2/score.py --bench-dir results/bench_tuned --replace-models FERL-deep-tuned

encoding:            ## one-hot sensitivity on the three datasets with multi-level nominal attributes
	FERL_NOMINAL=onehot $(PY) experiments/benchmark2/harness.py --datasets german australian crx \
	  --models CART C45 FIGS RuleFit LogReg FURIA FuzzyUCS-DS NeuRules SampledRuleList NCC ICDT \
	  CredalC45 EDL RF GBDT MLP FERL FERL-credal FERL-medium FERL-deep --out-dir results/bench_oh
	$(PY) experiments/benchmark2/score.py --bench-dir results/bench_oh \
	  --per-fold-csv results/benchmark2_onehot_per_fold.csv --summary-csv results/benchmark2_onehot_summary.csv

ablations:           ## read-out, band width, split criterion, bounded support
	$(PY) $(REL)/ds_decision_rule_ablation.py
	$(PY) $(REL)/ds_width_ablation.py
	$(PY) $(ASSETS)/make_cci_gini_learned_ablation.py
	$(PY) $(REL)/ds_bounded_support_ablation.py

diagnostics:         ## stability constants, ignorance semantics, depth sweep, covariate shift
	$(PY) $(REL)/ds_stability_constants.py
	$(PY) $(REL)/ds_ignorance_semantics.py --study noop
	$(PY) $(REL)/ds_ignorance_semantics.py --study depth
	$(PY) $(REL)/ds_credal_shift.py
	$(PY) experiments/benchmark2/credal_stress.py

ood:                 ## tabular near-OOD for the three variants (+ EDL and detectors)
	$(PY) $(REL)/ds_ood_residual_variants.py ferl-compact
	$(PY) $(REL)/ds_ood_residual_variants.py ferl-medium
	$(PY) $(REL)/ds_ood_residual_variants.py ferl-deep

cbm:                 ## concept-error propagation (needs the CUB concept files, see REPRODUCE.md)
	$(PY) experiments/cub_cbm/concept_error_robustness.py
	$(PY) experiments/cub_cbm/concept_error_robustness.py --subset 20 --depth 12 --detector-seeds 0 1 2

runtime:             ## wall-clock timings (run on an otherwise idle machine)
	$(PY) experiments/benchmark2/wallclock_scaling.py --no-resume

experiments: tabular tuned encoding ablations diagnostics ood cbm runtime
