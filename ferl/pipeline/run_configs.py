"""
Config-driven benchmark over the full KEEL suite.

Runs several FERL configurations (ferl_pipeline.CONFIGS) side-by-side with the
sklearn baselines on the same stratified-CV splits. Calibration is a UNIFORM
eval axis: every method gets CV-isotonic calibration (fit out-of-fold on the
training data, so no training data is lost and accuracy stays comparable). We
report accuracy, #rules, and ECE both raw and calibrated.

Robust for long offline runs: per-dataset try/except, rare-class filtering, and
incremental CSV output (results/config_benchmark.csv) so partial progress is
saved.

Usage:
    python run_configs.py            # quick smoke: 5 datasets, 3 folds
    python run_configs.py full       # full 44-dataset suite, 5 folds
"""
import os
import sys
import csv
import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from ferl.pipeline.ferl_pipeline import make, CONFIGS
import ferl.eval._eval_accuracy as e
from ferl.uncertainty.recalibrate import ece

warnings.filterwarnings("ignore")

KEEL_DIR = e.KEEL_DIR
ACTIVE_CONFIGS = ["ferl-original", "ferl-compact", "ferl-enhanced"]   # credal stub excluded
BASELINES = ["CART", "C4.5", "RandomForest"]
ALL_DATASETS = [
    'appendicitis', 'australian', 'banana', 'bupa', 'chess', 'coil2000',
    'contraceptive', 'crx', 'ecoli', 'flare', 'german', 'glass', 'haberman',
    'hayes-roth', 'heart', 'hepatitis', 'housevotes', 'iris', 'led7digit',
    'magic', 'mammographic', 'monk-2', 'newthyroid', 'page-blocks', 'penbased',
    'phoneme', 'pima', 'ring', 'saheart', 'satimage', 'segment', 'sonar',
    'spambase', 'spectfheart', 'thyroid', 'titanic', 'twonorm', 'vehicle',
    'wdbc', 'wine', 'winequality-red', 'winequality-white', 'wisconsin', 'zoo',
]
SMOKE_DATASETS = ["iris", "wine", "ecoli", "pima", "vehicle"]
# Curated 30 for the DS credal / epistemic-aleatoric study: continuous-feature,
# low-imbalance (<~7), sized for the 4-way split (train/cal-head/cal-conformal/
# test), spanning 2->11 classes and 178->19k samples. See ds_* experiments.
SELECTED_30 = [
    "magic", "penbased", "ring", "twonorm", "satimage", "optdigits", "texture",
    "phoneme", "banana", "spambase", "segment", "contraceptive", "german", "vowel",
    "vehicle", "mammographic", "pima", "australian", "wisconsin", "crx", "balance",
    "wdbc", "saheart", "bupa", "ionosphere", "ecoli", "spectfheart", "heart", "glass", "wine",
]
EPS = 1e-8


def make_model(name, seed):
    if name in CONFIGS:
        return make(name, random_state=seed)
    if name == "CART":
        return DecisionTreeClassifier(criterion="gini", random_state=seed,
                                      min_samples_split=5, min_samples_leaf=2)
    if name == "C4.5":
        return DecisionTreeClassifier(criterion="entropy", random_state=seed,
                                      min_samples_split=5, min_samples_leaf=2)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=100, random_state=seed)
    raise KeyError(name)


def n_rules_of(model, name):
    if name in CONFIGS:
        return float(model.n_rules())
    if name == "RandomForest":
        return float(np.mean([t.get_n_leaves() for t in model.estimators_]))
    return float(model.get_n_leaves())


MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0")) or None  # cap dataset size (MDLP is O(N^2))


# Encoding of nominal KEEL attributes for every experiment that loads data here:
# 'ordinal' (the benchmark's integer codes; used for all reported results) or
# 'onehot' (the encoding-sensitivity analysis); override with FERL_NOMINAL.
NOMINAL = os.environ.get("FERL_NOMINAL", "ordinal")


def load_filtered(name, min_per_class=10, nominal=None):
    X, y = e.load_keel(name, nominal=nominal or NOMINAL)
    keep = [c for c in np.unique(y) if np.sum(y == c) >= min_per_class]
    mask = np.isin(y, keep)
    X, y = X[mask], y[mask]
    _, y = np.unique(y, return_inverse=True)
    if MAX_SAMPLES is not None and len(y) > MAX_SAMPLES:
        rs = np.random.RandomState(0)
        idx = rs.choice(len(y), MAX_SAMPLES, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def cv_isotonic_calibrated(name, Xtr, ytr, P_te_raw, seed, n_classes, inner_cv=3):
    """Out-of-fold isotonic calibration: fit OvR isotonic on OOF train probs,
    apply to the (full-data) test probs. No training data lost for accuracy."""
    oof = np.zeros((len(ytr), n_classes))
    skf = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    for tri, vai in skf.split(Xtr, ytr):
        m = make_model(name, seed)
        m.fit(Xtr[tri], ytr[tri])
        oof[vai] = m.predict_proba(Xtr[vai])
    out = np.zeros_like(P_te_raw)
    for c in range(n_classes):
        try:
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            ir.fit(oof[:, c], (ytr == c).astype(float))
            out[:, c] = ir.transform(P_te_raw[:, c])
        except Exception:
            out[:, c] = P_te_raw[:, c]
    out = np.clip(out, EPS, None)
    return out / out.sum(axis=1, keepdims=True)


def main(full=False):
    datasets = ALL_DATASETS if full else SMOKE_DATASETS
    n_folds = 5 if full else 3
    methods = ACTIVE_CONFIGS + BASELINES
    os.makedirs("results", exist_ok=True)
    csv_path = "results/config_benchmark.csv"
    print(f"Config benchmark | {len(datasets)} datasets | {n_folds} folds")
    print(f"methods: {methods}\nwriting per-dataset rows to {csv_path}\n")

    rows = []
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "method", "acc", "n_rules", "ece_raw", "ece_cal"])
        for ds in datasets:
            try:
                X, y = load_filtered(ds)
                if len(np.unique(y)) < 2 or len(y) < 30:
                    print(f"  {ds}: SKIP (too few samples/classes)"); continue
            except Exception as ex:
                print(f"  {ds}: SKIP load error ({ex})"); continue
            acc = {m: [] for m in methods}
            nr = {m: [] for m in methods}
            er = {m: [] for m in methods}
            ec = {m: [] for m in methods}
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=33)
            for seed, (tr, te) in enumerate(skf.split(X, y)):
                Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
                C = len(np.unique(ytr))
                for m in methods:
                    try:
                        model = make_model(m, seed); model.fit(Xtr, ytr)
                        P = model.predict_proba(Xte)
                        acc[m].append((P.argmax(1) == yte).mean())
                        nr[m].append(n_rules_of(model, m))
                        er[m].append(ece(P, yte))
                        Pc = cv_isotonic_calibrated(m, Xtr, ytr, P, seed, C)
                        ec[m].append(ece(Pc, yte))
                    except Exception as ex:
                        print(f"    {ds}/{m}/fold{seed}: ERROR {ex}")
            for m in methods:
                if not acc[m]:
                    continue
                r = [ds, m, np.mean(acc[m]), np.mean(nr[m]), np.mean(er[m]), np.mean(ec[m])]
                rows.append(r); w.writerow(r); fh.flush()
            print(f"  {ds}: done")

    # summary across datasets
    print("\n=== MEAN ACROSS DATASETS ===")
    print("method".ljust(16) + "acc".ljust(9) + "#rules".ljust(9) + "ECE_raw".ljust(9) + "ECE_cal")
    for m in methods:
        mr = [r for r in rows if r[1] == m]
        if not mr:
            continue
        cols = np.array([[r[2], r[3], r[4], r[5]] for r in mr], dtype=float).mean(0)
        print(m.ljust(16) + f"{cols[0]:.3f}".ljust(9) + f"{cols[1]:.1f}".ljust(9)
              + f"{cols[2]:.3f}".ljust(9) + f"{cols[3]:.3f}")
    print(f"\nFull results: {csv_path}")


if __name__ == "__main__":
    main(full=(len(sys.argv) > 1 and sys.argv[1] == "full"))
