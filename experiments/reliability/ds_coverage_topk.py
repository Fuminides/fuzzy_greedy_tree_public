"""
Anti-smoothing lever: route each node's non-top-class consequent mass to Theta
instead of to wrong-class singletons. Equivalent to discounting firing by the
top-k probability mass and using the renormalized top-k consequent.

Predicted to kill the stubborn multiclass below_Bel (wrong classes get Bel=0)
while also raising Pl (more m_theta). Compared against raw and the evidential
discount; also combined with it. Tracks pignistic accuracy (zeroing classes can
move the argmax). No ensemble.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from ferl.pipeline.ferl_pipeline import make
from ferl.pipeline.run_configs import load_filtered, SELECTED_30
from ferl.uncertainty.recalibrate import finetune_reliability
from ds_coverage_errors import breakdown
from ds_coverage_levers import ds_combine

warnings.filterwarnings("ignore")
N_FOLDS = 3
METHODS = ["raw", "evidential", "topk1", "topk2", "topk1_ev"]


def topk(cons, k):
    """(renormalized top-k consequent, per-node top-k mass r)."""
    K, C = cons.shape
    if k >= C:
        return cons, np.ones(K)
    idx = np.argsort(-cons, axis=1)[:, :k]
    mask = np.zeros_like(cons, bool)
    np.put_along_axis(mask, idx, True, axis=1)
    kept = np.where(mask, cons, 0.0)
    r = kept.sum(1)
    return kept / np.clip(r[:, None], 1e-12, None), r


def acc_of(bel, pl, y, C):
    mth = (pl - bel)[:, 0]
    betp = bel + mth[:, None] / C
    return (betp.argmax(1) == y).mean()


def main():
    agg = {g: {m: np.zeros(3, int) for m in METHODS} for g in ["binary", "multi", "all"]}
    accs = {g: {m: [] for m in METHODS} for g in ["binary", "multi", "all"]}
    for ds in SELECTED_30:
        try:
            X, y = load_filtered(ds)
        except Exception:
            continue
        C = len(np.unique(y))
        grp = "binary" if C == 2 else "multi"
        for seed, (tr, te) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=33).split(X, y)):
            Xtr_f, ytr_f, Xte, yte = X[tr], y[tr], X[te], y[te]
            try:
                Xtr, Xcal, ytr, ycal = train_test_split(
                    Xtr_f, ytr_f, test_size=0.33, random_state=seed, stratify=ytr_f)
            except ValueError:
                continue
            f = make("ferl-compact", random_state=0).fit(Xtr, ytr).tree_
            M_cal, cons, names = f.node_activation_matrix(Xcal)
            supp = np.array([f.node_dict_access[n]['coverage'] for n in names]) * f._n_train
            depth = np.array([f.node_dict_access[n]['depth'] for n in names])
            r_ev = finetune_reliability(M_cal, ycal, supp, depth, cons, C, loss="evidential")
            c1, r1 = topk(cons, 1)
            c2, r2 = topk(cons, 2)
            M = f.node_activation_matrix(Xte)[0]
            variants = {
                "raw": ds_combine(M, cons),
                "evidential": ds_combine(M * r_ev[None, :], cons),
                "topk1": ds_combine(M * r1[None, :], c1),
                "topk2": ds_combine(M * r2[None, :], c2),
                "topk1_ev": ds_combine(M * r_ev[None, :] * r1[None, :], c1),
            }
            for m, (bel, pl) in variants.items():
                _, cov, ab, be, _, _ = breakdown(bel, pl, yte)
                for g in (grp, "all"):
                    agg[g][m] += [cov, ab, be]
                    accs[g][m].append(acc_of(bel, pl, yte, C))
        print(f"  done {ds}", flush=True)

    rows = []
    print("\n=== covered / above_Pl / below_Bel / acc ===")
    for g in ["all", "binary", "multi"]:
        print(f"\n[{g}]")
        for m in METHODS:
            cov, ab, be = agg[g][m]; tot = cov + ab + be
            if tot == 0:
                continue
            ac = float(np.mean(accs[g][m]))
            print(f"  {m:11s}: covered={cov/tot:.1%}  above_Pl={ab/tot:.1%}  below_Bel={be/tot:.1%}  acc={ac:.3f}")
            rows.append([g, m, tot, cov, ab, be, ac])
    pd.DataFrame(rows, columns=["group", "method", "bins", "covered", "above_Pl", "below_Bel", "acc"]).to_csv(
        "results/ds_coverage_topk.csv", index=False)


if __name__ == "__main__":
    main()
