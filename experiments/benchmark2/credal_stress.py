"""Credal stress battery for the bounded learned fuzzy tree -- the OOD / shift metrics that
do NOT fit the proba-only Stage-B harness (they need generated OOD / shifted test data).
Complements score.py (which now carries FERL-deep's in-distribution native-set u65/u80 via
predict_set). Run from repo root:

  PYTHONPATH=.:experiments/benchmark2 python experiments/benchmark2/credal_stress.py

Writes results/credal_ood.csv, results/credal_shift_synth.csv, results/real_shift.csv and prints
a summary. Each test compares the density-aware DS-conf credal set against global split-conformal.
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from ferl.pipeline.run_configs import load_filtered
from ferl.pipeline.ferl_pipeline import make
from ferl.core.learned_tree import LearnedFuzzyTree

ALPHA = 0.10; TARGET = 1 - ALPHA; MAX_N = 4000
DEFAULT_DS = ["magic", "satimage", "penbased", "texture", "optdigits", "phoneme", "ring"]


def _load(ds):
    X, y = load_filtered(ds); X = np.asarray(X, float)
    if len(X) > MAX_N:
        X, _, y, _ = train_test_split(X, y, train_size=MAX_N, random_state=0, stratify=y)
    return X, y


def _qhi(s):
    n = len(s); k = min(int(np.ceil((n + 1) * TARGET)), n); return np.sort(s)[k - 1] if n else np.inf


def _qlo(s):
    n = len(s); k = max(1, int(np.floor((n + 1) * ALPHA))); return np.sort(s)[k - 1] if n else -np.inf


def _calibrate(m, Xcal, yi_cal):
    """Return (q_ds, q_gl): DS-conf plausibility threshold and global LAC threshold."""
    Pcal = m.predict_proba(Xcal); plc = m.predict_ds(Xcal, rule="dempster", leaves_only=True)[2]
    return _qlo(plc[np.arange(len(yi_cal)), yi_cal]), _qhi(1.0 - Pcal[np.arange(len(yi_cal)), yi_cal])


def _sets(m, X, q_ds, q_gl):
    ple = m.predict_ds(X, rule="dempster", leaves_only=True)[2]
    return ple >= q_ds - 1e-12, (1.0 - m.predict_proba(X)) <= q_gl + 1e-12


def ood_detection(datasets=DEFAULT_DS, seeds=2, depth=10):
    """Geometric-OOD detection: bounded firing/ignorance + DS-conf set size vs ferl-compact firing."""
    rows = []
    for ds in datasets:
        X, y = _load(ds); lo, hi = X.min(0), X.max(0); rng = hi - lo + 1e-9
        for seed in range(seeds):
            Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
            Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=seed, stratify=ytmp)
            m = LearnedFuzzyTree(max_depth=depth, random_state=0).fit(Xtr, ytr)   # bounded default
            cls = list(m.classes_); yi_cal = np.array([cls.index(v) for v in ycal])
            q_ds, q_gl = _calibrate(m, Xcal, yi_cal)
            Xood = np.random.default_rng(seed).uniform(lo - rng, hi + rng, size=Xte.shape)
            lab = np.r_[np.zeros(len(Xte)), np.ones(len(Xood))]
            ig_id = m.predict_ds(Xte, leaves_only=True)[3]; ig_ood = m.predict_ds(Xood, leaves_only=True)[3]
            dsc_id, _ = _sets(m, Xte, q_ds, q_gl); dsc_ood, _ = _sets(m, Xood, q_ds, q_gl)
            base = make("ferl-compact", random_state=0).fit(Xtr, ytr)
            rows.append(dict(ds=ds, seed=seed,
                auroc_ignorance=roc_auc_score(lab, np.r_[ig_id, ig_ood]),
                auroc_dsconf_setsize=roc_auc_score(lab, np.r_[dsc_id.sum(1), dsc_ood.sum(1)]),
                auroc_base_firing=roc_auc_score(lab, np.r_[-base.tree_.firing_strength(Xte),
                                                           -base.tree_.firing_strength(Xood)]),
                ds_size_id=dsc_id.sum(1).mean(), ds_size_ood=dsc_ood.sum(1).mean()))
    return pd.DataFrame(rows)


def shift_synthetic(datasets=DEFAULT_DS, seeds=3, depth=10, ks=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0)):
    """Graded per-feature Gaussian noise -> coverage/size vs shift magnitude (best case)."""
    rows = []
    for ds in datasets:
        X, y = _load(ds)
        for seed in range(seeds):
            Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
            Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=seed, stratify=ytmp)
            m = LearnedFuzzyTree(max_depth=depth, random_state=0).fit(Xtr, ytr)
            cls = list(m.classes_); sig = Xtr.std(0) + 1e-9
            yi_cal = np.array([cls.index(v) for v in ycal]); yi_te = np.array([cls.index(v) for v in yte])
            q_ds, q_gl = _calibrate(m, Xcal, yi_cal); rs = np.random.default_rng(seed)
            for k in ks:
                Xs = Xte + k * sig * rs.standard_normal(Xte.shape)
                ds_set, gl_set = _sets(m, Xs, q_ds, q_gl); ix = np.arange(len(yi_te))
                rows.append(dict(ds=ds, seed=seed, k=k, acc=(m.predict_proba(Xs).argmax(1) == yi_te).mean(),
                    ds_cov=ds_set[ix, yi_te].mean(), ds_size=ds_set.sum(1).mean(),
                    gl_cov=gl_set[ix, yi_te].mean(), gl_size=gl_set.sum(1).mean()))
    return pd.DataFrame(rows)


def _pick_split_feature(X, ycodes):
    d = X.shape[1]
    if d < 3:
        return 0
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9); Cm = np.nan_to_num(np.corrcoef(Xz, rowvar=False))
    yz = (ycodes - ycodes.mean()) / (ycodes.std() + 1e-9)
    cy = np.nan_to_num(np.array([abs(np.corrcoef(Xz[:, f], yz)[0, 1]) for f in range(d)]))
    score = (np.abs(Cm).sum(1) - 1.0) / (d - 1) - cy
    score[X.std(0) < 1e-9] = -np.inf
    return int(np.argmax(score))


def shift_real(datasets=DEFAULT_DS, seeds=3, depth=10):
    """Feature-domain split: real covariate shift (partial support overlap)."""
    rows = []
    for ds in datasets:
        X, y = _load(ds); s = _pick_split_feature(X, pd.factorize(y)[0].astype(float))
        med = np.median(X[:, s]); keep = [j for j in range(X.shape[1]) if j != s]
        src, tgt = X[:, s] < med, X[:, s] >= med
        Xsrc, ysrc = X[np.ix_(src, keep)], y[src]; Xtgt, ytgt = X[np.ix_(tgt, keep)], y[tgt]
        cset = set(np.unique(ysrc)); tm = np.array([v in cset for v in ytgt])
        Xtgt, ytgt = Xtgt[tm], ytgt[tm]
        if len(Xtgt) < 30 or len(np.unique(ysrc)) < 2:
            continue
        for seed in range(seeds):
            try:
                Xtr, Xtmp, ytr, ytmp = train_test_split(Xsrc, ysrc, test_size=0.5, random_state=seed, stratify=ysrc)
                Xcal, Xid, ycal, yid = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=seed, stratify=ytmp)
            except ValueError:
                continue
            m = LearnedFuzzyTree(max_depth=depth, random_state=0).fit(Xtr, ytr)
            cls = list(m.classes_); yi_cal = np.array([cls.index(v) for v in ycal])
            q_ds, q_gl = _calibrate(m, Xcal, yi_cal)
            for dom, Xe, ye in [("ID", Xid, yid), ("shift", Xtgt, ytgt)]:
                yi = np.array([cls.index(v) for v in ye]); ds_set, gl_set = _sets(m, Xe, q_ds, q_gl)
                ix = np.arange(len(yi))
                rows.append(dict(ds=ds, seed=seed, domain=dom,
                    ds_cov=ds_set[ix, yi].mean(), gl_cov=gl_set[ix, yi].mean(),
                    ds_size=ds_set.sum(1).mean(), gl_size=gl_set.sum(1).mean()))
    return pd.DataFrame(rows)


def main():
    print("credal stress battery (bounded learned tree, DS-conf vs global conformal)\n")
    ood = ood_detection(); ood.to_csv("results/credal_ood.csv", index=False)
    g = ood.mean(numeric_only=True)
    print(f"[OOD] geometric-OOD AUROC: ignorance {g.auroc_ignorance:.3f}, DS-conf set {g.auroc_dsconf_setsize:.3f}, "
          f"ferl-compact firing {g.auroc_base_firing:.3f}; DS set {g.ds_size_id:.2f}->{g.ds_size_ood:.2f} ID->OOD")
    syn = shift_synthetic(); syn.to_csv("results/credal_shift_synth.csv", index=False)
    s1 = syn[syn.k == 1.0].mean(numeric_only=True)
    print(f"[SHIFT-synth k=1] coverage DS-conf {s1.ds_cov:.3f} vs global {s1.gl_cov:.3f} (best case)")
    real = shift_real(); real.to_csv("results/real_shift.csv", index=False)
    r = real.groupby("domain").mean(numeric_only=True)
    print(f"[SHIFT-real] shifted coverage DS-conf {r.loc['shift','ds_cov']:.3f} vs global {r.loc['shift','gl_cov']:.3f} "
          f"(realistic; wins on {(real[real.domain=='shift'].groupby('ds').ds_cov.mean() > real[real.domain=='shift'].groupby('ds').gl_cov.mean()).sum()} datasets)")


if __name__ == "__main__":
    main()
