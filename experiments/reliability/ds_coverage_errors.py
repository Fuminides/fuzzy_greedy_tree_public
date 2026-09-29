"""
Error analysis of the DS credal interval calibration. For each one-vs-rest bin
(same pooling as ds_imprecise.imprecise_cal), the empirical frequency should fall
in [mean Bel, mean Pl]. We classify every bin as:
  covered    : Bel <= emp <= Pl
  above_Pl   : emp > Pl   -> truth MORE likely than plausibility (interval too LOW)
  below_Bel  : emp < Bel  -> truth LESS likely than belief      (interval too HIGH)
and report the split of the *errors*, plus the mean violation gap of each kind.
No ensemble needed -- just the focal FERL's Bel/Pl on held-out folds.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from ferl.pipeline.ferl_pipeline import make
from ferl.pipeline.run_configs import load_filtered, SELECTED_30

warnings.filterwarnings("ignore")
N_FOLDS = 3
NBINS = 10
EPS = 1e-6


def breakdown(bel, pl, y, nbins=NBINS):
    C = bel.shape[1]
    low = np.concatenate([bel[:, c] for c in range(C)])
    up = np.concatenate([pl[:, c] for c in range(C)])
    tg = np.concatenate([(y == c).astype(float) for c in range(C)])
    mid = 0.5 * (low + up)
    edges = np.quantile(mid, np.linspace(0, 1, nbins + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    tot = cov = above = below = 0
    a_gap, b_gap = [], []
    for b in range(nbins):
        m = (mid > edges[b]) & (mid <= edges[b + 1])
        if m.sum() < 5:
            continue
        emp, lo, hi = tg[m].mean(), low[m].mean(), up[m].mean()
        tot += 1
        if emp > hi + EPS:
            above += 1; a_gap.append(emp - hi)
        elif emp < lo - EPS:
            below += 1; b_gap.append(lo - emp)
        else:
            cov += 1
    return tot, cov, above, below, a_gap, b_gap


def main():
    rows = []
    T = dict(tot=0, cov=0, above=0, below=0)
    AG, BG = [], []
    for ds in SELECTED_30:
        try:
            X, y = load_filtered(ds)
        except Exception:
            continue
        t = c = a = b = 0
        ncorr = nsamp = 0
        for seed, (tr, te) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=33).split(X, y)):
            f = make("ferl-compact", random_state=0).fit(X[tr], y[tr]).tree_
            betp, bel, pl, _ = f.predict_ds(X[te])
            tt, cc, aa, bb, ag, bg = breakdown(bel, pl, y[te])
            t += tt; c += cc; a += aa; b += bb; AG += ag; BG += bg
            ncorr += int((betp.argmax(1) == y[te]).sum()); nsamp += len(y[te])
        if t == 0:
            continue
        rows.append([ds, ncorr / nsamp if nsamp else float("nan"), t, c, a, b])
        T["tot"] += t; T["cov"] += c; T["above"] += a; T["below"] += b
        print(f"  {ds}: bins={t} covered={c} above_Pl={a} below_Bel={b}", flush=True)

    pd.DataFrame(rows, columns=["dataset", "accuracy", "bins", "covered", "above_Pl", "below_Bel"]).to_csv(
        "results/ds_coverage_errors.csv", index=False)
    t = T["tot"]; err = T["above"] + T["below"]
    print("\n=== TOTAL across datasets ===")
    print(f"  bins                : {t}")
    print(f"  covered             : {T['cov']} ({T['cov']/t:.1%})")
    print(f"  ERRORS              : {err} ({err/t:.1%})")
    if err:
        print(f"    above Pl (truth > plausibility, interval too LOW)  : {T['above']} ({T['above']/err:.1%} of errors)")
        print(f"    below Bel (truth < belief, interval too HIGH)      : {T['below']} ({T['below']/err:.1%} of errors)")
    print(f"  mean gap above Pl   : {np.mean(AG) if AG else 0:.3f}")
    print(f"  mean gap below Bel  : {np.mean(BG) if BG else 0:.3f}")


if __name__ == "__main__":
    main()
