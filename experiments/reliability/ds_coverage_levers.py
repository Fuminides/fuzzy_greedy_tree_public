"""
Does each lever fix the coverage-error type it's predicted to?

  evidential : learned reliability discount -> widens (more m_theta). Predicted to
               shrink BOTH error types, especially above_Pl (raises plausibility).
  smooth     : Laplace-smoothed consequents (no discount) -> pulls belief off the
               wrong-class singletons. Predicted to shrink below_Bel, esp. multiclass.
  both       : discount + smoothing.

Reuses the bin breakdown from ds_coverage_errors; reports covered/above_Pl/below_Bel
split by binary vs multiclass. No ensemble.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from ferl.pipeline.ferl_pipeline import make
from ferl.pipeline.run_configs import load_filtered, SELECTED_30
from ferl.uncertainty.recalibrate import finetune_reliability
from ds_coverage_errors import breakdown

warnings.filterwarnings("ignore")
N_FOLDS = 3
LAPLACE = 1.0
METHODS = ["raw", "evidential", "smooth", "both"]


def ds_combine(M, cons):
    one_minus = 1.0 - M
    term = M[:, :, None] * cons[None, :, :] + one_minus[:, :, None]
    Qc = term.prod(1)
    Qt = one_minus.prod(1)
    m_c = np.clip(Qc - Qt[:, None], 0.0, None)
    tot = np.where(m_c.sum(1) + Qt <= 0, 1.0, m_c.sum(1) + Qt)
    m_c = m_c / tot[:, None]; m_th = Qt / tot
    return m_c, m_c + m_th[:, None]            # bel, pl


def smooth_cons(cons, support, alpha=LAPLACE):
    C = cons.shape[1]
    counts = cons * support[:, None]
    return (alpha + counts) / (C * alpha + support[:, None])


def main():
    # agg[group][method] = [covered, above, below]; acc[group][method] = [correct, total]
    agg = {g: {m: np.zeros(3, int) for m in METHODS} for g in ["binary", "multi", "all"]}
    acc = {g: {m: np.zeros(2, float) for m in METHODS} for g in ["binary", "multi", "all"]}
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
            cons_s = smooth_cons(cons, supp)
            M_te, _, _ = f.node_activation_matrix(Xte)
            variants = {
                "raw": ds_combine(M_te, cons),
                "evidential": ds_combine(M_te * r_ev[None, :], cons),
                "smooth": ds_combine(M_te, cons_s),
                "both": ds_combine(M_te * r_ev[None, :], cons_s),
            }
            for m, (bel, pl) in variants.items():
                _, cov, ab, be, _, _ = breakdown(bel, pl, yte)
                # Point prediction is bel.argmax (pl = bel + const Theta mass).
                ncorr = float((bel.argmax(1) == yte).sum())
                for g in (grp, "all"):
                    agg[g][m] += [cov, ab, be]
                    acc[g][m] += [ncorr, len(yte)]
        print(f"  done {ds}", flush=True)

    rows = []
    print("\n=== covered / above_Pl / below_Bel  (counts and % of bins) ===")
    for g in ["all", "binary", "multi"]:
        print(f"\n[{g}]")
        for m in METHODS:
            cov, ab, be = agg[g][m]; tot = cov + ab + be
            if tot == 0:
                continue
            accuracy = acc[g][m][0] / acc[g][m][1] if acc[g][m][1] else float("nan")
            print(f"  {m:11s}: acc={accuracy:.1%}  covered={cov/tot:.1%}  above_Pl={ab/tot:.1%}  below_Bel={be/tot:.1%}  (n={tot})")
            rows.append([g, m, tot, accuracy, cov, ab, be])
    pd.DataFrame(rows, columns=["group", "method", "bins", "accuracy", "covered", "above_Pl", "below_Bel"]).to_csv(
        "results/ds_coverage_levers.csv", index=False)


if __name__ == "__main__":
    main()
