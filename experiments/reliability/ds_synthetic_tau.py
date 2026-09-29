"""
Top-p (nucleus) routing with tau calibration, across number of classes.

Per node, keep top classes until cumulative consequent mass >= tau; route the
tail to Theta. tau is calibrated on a held-out split using ONLY labels: the most
aggressive routing (smallest tau / max coverage) whose cal accuracy stays within
`TOL` of the best. Does ONE tau adapt across C where fixed k could not?

Synthetic C-Gaussians on a circle (overlap fixed across C), known posterior.
Compares per C: Dempster, top-p@fixed tau=0.8, top-p@calibrated tau*, and k=1.
Metrics vs true posterior: covered / acc / DKL(true||betp).
"""
import warnings
import numpy as np
import pandas as pd
from ferl.pipeline.ferl_pipeline import make
from ds_coverage_levers import ds_combine
from ds_coverage_topk import topk

warnings.filterwarnings("ignore")
CS = [3, 4, 6, 8, 10]
ADJ, SIGMA, SEEDS = 2.5, 1.0, 3
TAU_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
FIXED_TAU = 0.8
TOL = 0.01


def means_for(C):
    R = ADJ / (2 * np.sin(np.pi / C))
    a = 2 * np.pi * np.arange(C) / C
    return R * np.stack([np.cos(a), np.sin(a)], axis=1)


def gen(n, M, rng):
    y = rng.randint(0, len(M), n)
    return M[y] + rng.randn(n, 2) * SIGMA, y


def post(X, M):
    d2 = ((X[:, None, :] - M[None, :, :]) ** 2).sum(2)
    lp = -d2 / (2 * SIGMA ** 2); lp -= lp.max(1, keepdims=True)
    p = np.exp(lp); return p / p.sum(1, keepdims=True)


def dkl(Pt, Pp):
    Pp = np.clip(Pp, 1e-12, 1.0)
    return float((Pt * (np.log(np.clip(Pt, 1e-12, 1.0)) - np.log(Pp))).sum(1).mean())


def topp(cons, tau):
    """Nucleus routing: keep top classes until cumulative >= tau (incl. crosser)."""
    order = np.argsort(-cons, axis=1)
    sc = np.take_along_axis(cons, order, axis=1)
    before = np.cumsum(sc, axis=1) - sc          # cumulative mass strictly before each
    keep = np.where(before < tau, sc, 0.0)
    kept = np.zeros_like(cons)
    np.put_along_axis(kept, order, keep, axis=1)
    r = kept.sum(1)
    return kept / np.clip(r[:, None], 1e-12, None), r


def evalp(M, cons, Pt, y, C):
    bel, pl = ds_combine(M, cons)
    betp = bel + (pl - bel)[:, 0][:, None] / C
    above = (Pt > pl + 1e-9).mean(); below = (Pt < bel - 1e-9).mean()
    return 1 - above - below, (betp.argmax(1) == y).mean(), dkl(Pt, betp)


def main():
    rows = []
    for C in CS:
        M = means_for(C)
        out = {m: {"cov": [], "acc": [], "dkl": [], "tau": []}
               for m in ["dempster", "fixed", "calib", "k1"]}
        for seed in range(SEEDS):
            rng = np.random.RandomState(seed)
            Xtr, ytr = gen(7000, M, rng)
            Xca, yca = gen(3500, M, rng)
            Xte, yte = gen(7000, M, rng)
            Pt = post(Xte, M)
            f = make("ferl-compact", random_state=0).fit(Xtr, ytr).tree_
            _, cons, _ = f.node_activation_matrix(Xtr[:1])
            M_ca = f.node_activation_matrix(Xca)[0]
            M_te = f.node_activation_matrix(Xte)[0]

            # calibrate tau on cal using labels only (betp argmax = bel argmax)
            base_acc = (ds_combine(M_ca, cons)[0].argmax(1) == yca).mean()  # tau=1 ~ keep all
            tau_star = 1.0
            for tau in TAU_GRID:                       # ascending -> first that preserves acc
                ch, rh = topp(cons, tau)
                acc = (ds_combine(M_ca * rh[None, :], ch)[0].argmax(1) == yca).mean()
                if acc >= base_acc - TOL:
                    tau_star = tau; break

            for name, tau, kk in [("calib", tau_star, None), ("fixed", FIXED_TAU, None),
                                  ("dempster", 1.0, None), ("k1", None, 1)]:
                if kk == 1:
                    ck, rk = topk(cons, 1)
                else:
                    ck, rk = topp(cons, tau)
                cov, ac, kl = evalp(M_te * rk[None, :], ck, Pt, yte, C)
                out[name]["cov"].append(cov); out[name]["acc"].append(ac)
                out[name]["dkl"].append(kl); out[name]["tau"].append(tau if kk is None else 0)

        print(f"\n=== C={C} ===")
        print(f"{'method':10s} {'tau':>5s} {'covered':>8s} {'acc':>6s} {'DKL':>7s}")
        for m in ["dempster", "fixed", "calib", "k1"]:
            o = {x: float(np.mean(out[m][x])) for x in out[m]}
            lbl = {"fixed": f"top-p .8", "calib": "top-p cal", "dempster": "dempster", "k1": "top-k=1"}[m]
            print(f"{lbl:10s} {o['tau']:5.2f} {o['cov']:8.1%} {o['acc']:6.3f} {o['dkl']:7.3f}")
            rows.append([C, m, o['tau'], o['cov'], o['acc'], o['dkl']])
    pd.DataFrame(rows, columns=["C", "method", "tau", "covered", "acc", "dkl"]).to_csv(
        "results/ds_synthetic_tau.csv", index=False)


if __name__ == "__main__":
    main()
