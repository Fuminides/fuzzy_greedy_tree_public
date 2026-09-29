"""
Pressure-test the conditional-coverage claim for firing-stratified conformal.

Claim: FERL's rule-firing strength is a valid difficulty variable, so Mondrian
(group-conditional) conformal stratified by firing achieves better *conditional*
coverage than marginal CP (LAC/APS) at comparable set size.

Protocol: for each dataset, repeat over N_SEEDS random 50/25/25 train/cal/test
splits. Report, averaged over seeds:
  - marginal coverage          (all methods should hit ~1-alpha)
  - avg set size               (efficiency; smaller better)
  - worst firing-bin coverage  (conditional coverage; closer to 1-alpha better)
  - conditional violation      = max(0, (1-alpha) - worst-bin coverage)  [lower better]

The headline test is LAC (marginal, tight) vs Mondrian (firing-conditional).
"""
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from ferl.core.tree_learning import FuzzyCART
from ferl.fuzzification.fuzzification_mdlp import learn_partitions_mdlp
from ferl.uncertainty.conformal import (ConformalFERL, coverage, avg_set_size, worst_slab_coverage)
import ferl.eval._eval_accuracy as e

warnings.filterwarnings("ignore")

ALPHA = 0.1
N_SEEDS = 10
N_FIRING_BINS = 5
DATASETS = ["wine", "wisconsin", "pima", "vehicle", "balance",
            "australian", "bupa", "banana"]
SCORES = ["lac", "aps", "firing", "mondrian"]


def load_filtered(name, min_per_class=15, cap=2500):
    X, y = e.load_keel(name)
    keep = [c for c in np.unique(y) if np.sum(y == c) >= min_per_class]
    mask = np.isin(y, keep)
    X, y = X[mask], y[mask]
    _, y = np.unique(y, return_inverse=True)
    if len(y) > cap:  # bound MDLP cost
        rs = np.random.RandomState(0)
        idx = rs.choice(len(y), cap, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def run_one(X, y, seed):
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=seed, stratify=ytmp)
    parts = learn_partitions_mdlp(Xtr, ytr)
    clf = FuzzyCART(parts, max_rules=20)
    clf.fit(Xtr, ytr)
    firing_te = clf.firing_strength(Xte)
    out = {}
    for sc in SCORES:
        cp = ConformalFERL(clf, score=sc).calibrate(Xcal, ycal, alpha=ALPHA)
        S = cp.predict_set(Xte)
        cov = coverage(S, yte, clf.classes_)
        size = avg_set_size(S)
        wsc = worst_slab_coverage(S, yte, clf.classes_, firing_te, delta=0.1)
        out[sc] = (cov, size, wsc, max(0.0, (1 - ALPHA) - wsc))
    return out


def main():
    print(f"Conditional-coverage pressure-test | alpha={ALPHA} (target {1-ALPHA:.0%}) | "
          f"{N_SEEDS} seeds | {N_FIRING_BINS} firing bins")
    print("cells = marg-cov / size / WSC(firing,delta=0.1) / WSC-violation\n")
    header = "dataset".ljust(12) + "".join(s.ljust(26) for s in SCORES)
    print(header)
    agg = {s: {"cov": [], "size": [], "worst": [], "viol": []} for s in SCORES}
    for name in DATASETS:
        try:
            X, y = load_filtered(name)
        except Exception as ex:
            print(f"{name.ljust(12)} SKIP ({ex})")
            continue
        per = {s: {"cov": [], "size": [], "worst": [], "viol": []} for s in SCORES}
        for seed in range(N_SEEDS):
            res = run_one(X, y, seed)
            for s in SCORES:
                cov, size, worst, viol = res[s]
                per[s]["cov"].append(cov); per[s]["size"].append(size)
                per[s]["worst"].append(worst); per[s]["viol"].append(viol)
        row = name.ljust(12)
        for s in SCORES:
            for k in per[s]:
                agg[s][k].append(np.nanmean(per[s][k]))
            row += (f"{np.mean(per[s]['cov']):.2f}/{np.mean(per[s]['size']):.2f}/"
                    f"{np.nanmean(per[s]['worst']):.2f}/{np.nanmean(per[s]['viol']):.2f}").ljust(26)
        print(row)
    print("-" * len(header))
    mrow = "MEAN".ljust(12)
    for s in SCORES:
        mrow += (f"{np.mean(agg[s]['cov']):.2f}/{np.mean(agg[s]['size']):.2f}/"
                 f"{np.mean(agg[s]['worst']):.2f}/{np.mean(agg[s]['viol']):.2f}").ljust(26)
    print(mrow)

    # Headline paired comparison: LAC vs Mondrian on conditional violation & size.
    print("\nLAC vs Mondrian (mean over datasets):")
    print(f"  conditional violation:  LAC {np.mean(agg['lac']['viol']):.3f}  ->  "
          f"Mondrian {np.mean(agg['mondrian']['viol']):.3f}  (lower=better)")
    print(f"  avg set size:           LAC {np.mean(agg['lac']['size']):.2f}  ->  "
          f"Mondrian {np.mean(agg['mondrian']['size']):.2f}")
    wins = sum(np.array(agg['mondrian']['viol']) <= np.array(agg['lac']['viol']))
    print(f"  Mondrian <= LAC conditional violation on {wins}/{len(agg['lac']['viol'])} datasets")


if __name__ == "__main__":
    main()
