"""
Missingness sweep: does FERL degrade gracefully under missing features?

FERL handles missing features natively: unobserved features get uniform
membership (via observed_mask), which lowers firing Phi(x), so firing-aware
conformal sets widen automatically where information is missing. We compare:

  - FERL-native  : observed_mask marks missing features (fuzzy marginalization)
  - FERL-impute  : missing features filled with train mean, mask all-observed

Conformal calibrated on CLEAN data; evaluated as a fraction rho of test features
is masked. Metrics vs rho: point accuracy, marginal coverage, avg set size.
Claim: native handling keeps coverage/accuracy up and widens sets sensibly,
while imputation loses coverage faster.
"""
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ferl.core.tree_learning import FuzzyCART
from ferl.fuzzification.fuzzification_mdlp import learn_partitions_mdlp
from ferl.uncertainty.conformal import ConformalFERL, coverage, avg_set_size
from ferl.uncertainty.conformal_conditional import load_filtered, DATASETS

warnings.filterwarnings("ignore")

ALPHA = 0.1
N_SEEDS = 5
RHOS = [0.0, 0.25, 0.5, 0.75]
SCORE = "mondrian"   # firing-aware conformal


def mask_features(X, rho, rng):
    """Return (X_imputed_placeholder, observed_mask) with rho fraction masked per row."""
    n, d = X.shape
    mask = np.ones((n, d), dtype=bool)
    k = int(round(rho * d))
    if k > 0:
        for i in range(n):
            cols = rng.choice(d, k, replace=False)
            mask[i, cols] = False
    return mask


def run_one(X, y, seed):
    rng = np.random.RandomState(seed)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=seed, stratify=ytmp)
    parts = learn_partitions_mdlp(Xtr, ytr)
    clf = FuzzyCART(parts, max_rules=20)
    clf.fit(Xtr, ytr)
    train_mean = Xtr.mean(axis=0)

    cp = ConformalFERL(clf, score=SCORE).calibrate(Xcal, ycal, alpha=ALPHA)

    res = {}
    for rho in RHOS:
        mask = mask_features(Xte, rho, rng)
        # native: pass observed_mask (unobserved -> uniform membership)
        acc_n = accuracy_score(yte, clf.predict(Xte, observed_mask=mask))
        S_n = cp.predict_set_masked(Xte, mask)
        cov_n, size_n = coverage(S_n, yte, clf.classes_), avg_set_size(S_n)
        # impute: fill masked with train mean, mask all-observed
        Xi = Xte.copy()
        Xi[~mask] = np.broadcast_to(train_mean, Xte.shape)[~mask]
        acc_i = accuracy_score(yte, clf.predict(Xi))
        S_i = cp.predict_set(Xi)
        cov_i, size_i = coverage(S_i, yte, clf.classes_), avg_set_size(S_i)
        res[rho] = (acc_n, cov_n, size_n, acc_i, cov_i, size_i)
    return res


def main():
    print(f"Missingness sweep | conformal score={SCORE} | target cov {1-ALPHA:.0%} | {N_SEEDS} seeds")
    print("native = observed_mask (fuzzy) ; impute = train-mean fill\n")
    for name in DATASETS:
        try:
            X, y = load_filtered(name)
        except Exception as ex:
            print(f"{name}: SKIP ({ex})"); continue
        acc = {r: {"n": [], "i": []} for r in RHOS}
        cov = {r: {"n": [], "i": []} for r in RHOS}
        size = {r: {"n": [], "i": []} for r in RHOS}
        for seed in range(N_SEEDS):
            res = run_one(X, y, seed)
            for r in RHOS:
                an, cn, sn, ai, ci, si = res[r]
                acc[r]["n"].append(an); acc[r]["i"].append(ai)
                cov[r]["n"].append(cn); cov[r]["i"].append(ci)
                size[r]["n"].append(sn); size[r]["i"].append(si)
        print(f"{name}")
        print("  rho   acc(native/impute)   cov(native/impute)   size(native/impute)")
        for r in RHOS:
            print(f"  {r:.2f}   {np.mean(acc[r]['n']):.3f} / {np.mean(acc[r]['i']):.3f}"
                  f"        {np.mean(cov[r]['n']):.3f} / {np.mean(cov[r]['i']):.3f}"
                  f"        {np.mean(size[r]['n']):.2f} / {np.mean(size[r]['i']):.2f}")
        print()


if __name__ == "__main__":
    main()
