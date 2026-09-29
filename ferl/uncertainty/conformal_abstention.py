"""
Interpretable abstention via selective prediction: does rule-firing make a good
confidence score for deciding when FERL should say "I don't know"?

Risk-coverage analysis: rank test points by a confidence score, abstain on the
least confident, and measure error among the accepted. AURC = area under the
risk-coverage curve (lower = better). Compares three FERL confidence signals:
  - maxproba   : max class probability (standard)
  - firing     : total rule-firing Phi(x) (fuzzy coverage)
  - combined   : maxproba * normalized firing
"""
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from ferl.core.tree_learning import FuzzyCART
from ferl.fuzzification.fuzzification_mdlp import learn_partitions_mdlp
from ferl.uncertainty.conformal import selective_risk_coverage
import ferl.eval._eval_accuracy as e
from ferl.uncertainty.conformal_conditional import load_filtered, DATASETS

warnings.filterwarnings("ignore")
N_SEEDS = 10
CONfIGS = ["maxproba", "firing", "combined"]


def run_one(X, y, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    parts = learn_partitions_mdlp(Xtr, ytr)
    clf = FuzzyCART(parts, max_rules=20)
    clf.fit(Xtr, ytr)
    P = clf.predict_proba(Xte)
    phi = clf.firing_strength(Xte)
    phin = phi / (np.quantile(phi, 0.95) + 1e-8)
    conf = {
        "maxproba": P.max(axis=1),
        "firing": phi,
        "combined": P.max(axis=1) * np.clip(phin, 0, 1),
    }
    return {c: selective_risk_coverage(P, conf[c], yte, clf.classes_)[0] for c in CONfIGS}


def main():
    print(f"Selective prediction AURC (lower=better) | {N_SEEDS} seeds\n")
    header = "dataset".ljust(12) + "".join(c.ljust(12) for c in CONfIGS)
    print(header)
    agg = {c: [] for c in CONfIGS}
    for name in DATASETS:
        try:
            X, y = load_filtered(name)
        except Exception as ex:
            print(f"{name.ljust(12)} SKIP ({ex})")
            continue
        per = {c: [] for c in CONfIGS}
        for seed in range(N_SEEDS):
            res = run_one(X, y, seed)
            for c in CONfIGS:
                per[c].append(res[c])
        row = name.ljust(12)
        for c in CONfIGS:
            m = float(np.mean(per[c]))
            agg[c].append(m)
            row += f"{m:.3f}".ljust(12)
        print(row)
    print("-" * len(header))
    print("MEAN".ljust(12) + "".join(f"{np.mean(agg[c]):.3f}".ljust(12) for c in CONfIGS))
    best = min(CONfIGS, key=lambda c: np.mean(agg[c]))
    print(f"\nBest confidence score: {best}")


if __name__ == "__main__":
    main()
