"""Self-contained accuracy harness for FuzzyCART (no seaborn/scikit_posthocs needed).

Runs stratified 5-fold CV on a fixed set of Keel datasets and reports mean accuracy.
Used to baseline and measure algorithm changes. Deterministic (fixed seeds).
"""
import os
import sys
import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from ex_fuzzy import utils, fuzzy_sets as fs
from ferl.core.tree_learning import FuzzyCART

warnings.filterwarnings("ignore")

# Override with the KEEL_DIR env var when running on another machine.
KEEL_DIR = os.environ.get("KEEL_DIR", "../keel_datasets")
DATASETS = ["iris", "wine", "glass", "ecoli", "vehicle", "bupa", "balance", "appendicitis", "wisconsin", "pima"]
SEED = 33


def load_keel(name, nominal="ordinal"):
    """Load a KEEL dataset as (X, y).

    ``nominal='ordinal'`` integer-codes attributes declared nominal in the KEEL
    header (``{...}``) or holding non-numeric values, as the original benchmark
    did. ``nominal='onehot'`` expands every declared-nominal predictor with three
    or more levels into indicator columns (binary ones stay a single 0/1 column),
    so threshold splits never impose an order on unordered categories."""
    path = f"{KEEL_DIR}/{name}/{name}.dat"
    rows, in_data, declared = [], False, []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.lower().startswith("@data"):
                in_data = True
                continue
            if s.lower().startswith("@attribute"):
                declared.append("{" in s)
                continue
            if s.startswith("@"):
                continue
            if in_data:
                rows.append(s)
    data = [r.split(",") for r in rows]
    arr = np.array(data, dtype=object)
    X_raw = arr[:, :-1]
    y_raw = np.array([v.strip() for v in arr[:, -1]])
    # numeric features only; encode any non-numeric column ordinally
    X = np.zeros(X_raw.shape, dtype=float)
    for j in range(X_raw.shape[1]):
        col = np.array([v.strip() for v in X_raw[:, j]])
        try:
            X[:, j] = col.astype(float)
        except ValueError:
            X[:, j] = LabelEncoder().fit_transform(col).astype(float)
    y = LabelEncoder().fit_transform(y_raw)
    if nominal == "onehot":
        is_nominal = declared[:X_raw.shape[1]]
        blocks = []
        for j in range(X_raw.shape[1]):
            levels = np.unique(X[:, j])
            if is_nominal[j] and len(levels) >= 3:
                blocks.append((X[:, j][:, None] == levels[None, :]).astype(float))
            elif is_nominal[j]:
                blocks.append((X[:, j] == levels[-1]).astype(float)[:, None])
            else:
                blocks.append(X[:, [j]])
        X = np.hstack(blocks)
    elif nominal != "ordinal":
        raise ValueError(f"nominal must be 'ordinal' or 'onehot', got {nominal!r}")
    return X, y


def run(max_rules=20, target_metric="cci", min_improvement=0.0, prediction_mode="soft",
        consistent_cci=True, partition_fn=None, partition_tag="quantile"):
    print(f"  config: max_rules={max_rules} target_metric={target_metric} "
          f"min_improvement={min_improvement} prediction_mode={prediction_mode} "
          f"consistent_cci={consistent_cci} partitions={partition_tag}")
    overall = []
    for name in DATASETS:
        try:
            X, y = load_keel(name)
        except Exception as e:
            print(f"    {name:14s} SKIP ({e})")
            continue
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        accs = []
        for tr, te in skf.split(X, y):
            if partition_fn is None:
                parts = utils.construct_partitions(X[tr], fs.FUZZY_SETS.t1)
            else:
                parts = partition_fn(X[tr], y[tr])
            clf = FuzzyCART(parts, max_rules=max_rules, target_metric=target_metric,
                            min_improvement=min_improvement)
            clf.prediction_mode = prediction_mode
            clf.consistent_cci = consistent_cci
            clf.fit(X[tr], y[tr])
            accs.append(accuracy_score(y[te], clf.predict(X[te])))
        m = float(np.mean(accs))
        overall.append(m)
        print(f"    {name:14s} acc={m:.4f}")
    print(f"  >>> MEAN ACROSS DATASETS: {np.mean(overall):.4f}\n")
    return float(np.mean(overall))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        for mode in ["hard_gate", "soft_gate", "soft"]:
            print(f"=== mode={mode} ===")
            run(prediction_mode=mode)
    elif len(sys.argv) > 1 and sys.argv[1] == "ae":
        from functools import partial
        from ferl.fuzzification.fuzzification_ae import learn_partitions
        print("=== quantile baseline ===")
        run()
        for lam in [0.5, 1.0, 2.0]:
            print(f"=== autoencoder partitions (lam={lam}, anchored) ===")
            run(partition_fn=partial(learn_partitions, n_partitions=3, lam=lam, epochs=300),
                partition_tag=f"ae_lam{lam}_anchored")
    elif len(sys.argv) > 1 and sys.argv[1] == "mdlp":
        from functools import partial
        from ferl.fuzzification.fuzzification_mdlp import learn_partitions_mdlp
        print("=== quantile baseline ===")
        run()
        for ov in [0.3, 0.5, 0.8]:
            print(f"=== MDLP partitions (overlap={ov}) ===")
            run(partition_fn=partial(learn_partitions_mdlp, overlap_frac=ov),
                partition_tag=f"mdlp_ov{ov}")
    elif len(sys.argv) > 1 and sys.argv[1] == "consistent":
        print("=== soft inference + LEGACY cci scoring ===")
        run(prediction_mode="soft", consistent_cci=False)
        print("=== soft inference + CONSISTENT cci scoring ===")
        run(prediction_mode="soft", consistent_cci=True)
    else:
        tag = sys.argv[1] if len(sys.argv) > 1 else "run"
        print(f"=== {tag} ===")
        run()
