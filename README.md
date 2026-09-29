# FERL: Fast Evidential Rule Learning

Code for the paper **"Evidential Rule Learning for Interpretable Classification with
Abstention"** (Javier Fumanal-Idocin and Javier Andreu-Perez).

FERL is a fuzzy rule-tree learner that uses its fired rules as belief masses for
Dempster-Shafer evidence. In one forward pass, without an auxiliary model or
calibration data, it returns:

- a point label;
- belief and plausibility for every class;
- a set-valued prediction that becomes an abstention when the evidence does not
  separate the classes;
- a near-out-of-distribution score that names the atypical attributes.

> This repository previously hosted the code of *A Fast Interpretable Fuzzy Tree
> Learner* (arXiv:2512.11616). That code is kept under the tag
> [`fgrt-arxiv-2512.11616`](https://github.com/Fuminides/fuzzy_greedy_tree_public/tree/fgrt-arxiv-2512.11616).

## Installation

Python 3.12 or later.

```bash
git clone https://github.com/Fuminides/fuzzy_greedy_tree_public.git
cd fuzzy_greedy_tree_public
pip install -e ".[viz]"
```

For the pinned environment used in the paper, run `make env` (see
[REPRODUCE.md](REPRODUCE.md)).

## Quick start

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from ferl.pipeline import make

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y)

model = make("ferl-deep").fit(X_train, y_train)

y_pred = model.predict(X_test)                       # point label
betp, bel, pl, ignorance = model.predict_ds(X_test, leaves_only=True)
sets = model.predict_set(X_test)                     # boolean (n_samples, n_classes)
abstain = sets.sum(axis=1) > 1                       # more than one class kept
```

The three variants of the paper:

| Paper | Code |
|---|---|
| FERL-compact | `make("ferl-compact")` |
| FERL-medium | `make("ferl-medium")` |
| FERL-deep | `make("ferl-deep")` (a `ferl.core.learned_tree.LearnedFuzzyTree`) |

## Repository structure

```
ferl/                 the library
  core/               FuzzyCART (compact, medium) and LearnedFuzzyTree (deep)
  fuzzification/      fuzzy partitions
  uncertainty/        conformal prediction and recalibration
  pipeline/           make() and the named configurations
ferl_fast/            optional Cython kernels
experiments/
  benchmark2/         main tabular benchmark and baselines
  reliability/        ablations, diagnostics and near-OOD experiments
  cub_cbm/            concept-bottleneck experiments (CUB, AwA2)
  paper_assets/       builders of the tables and figures
results/              result files read by the table builders
paper/                generated tables, macros and figures
tests/
```

## Reproducing the paper

```bash
make env                                # pinned environment in .venv
make data KEEL_DIR=../keel_datasets     # the 30 KEEL datasets
make test PY=.venv/bin/python
make tables PY=.venv/bin/python         # every table and figure from the stored results
```

[REPRODUCE.md](REPRODUCE.md) lists the command and the result file behind each table
and figure, and the protocols and seeds.

## Citation

```bibtex
@article{fumanal2026ferl,
  title={Evidential Rule Learning for Interpretable Classification with Abstention},
  author={Fumanal-Idocin, Javier and Andreu-Perez, Javier},
  year={2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
