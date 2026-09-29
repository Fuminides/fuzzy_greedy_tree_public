"""
Near-OOD (semantic novelty) across FERL variants: does the residual-feature
signal hold up on deeper trees, and how does it compare to dedicated detectors?

Protocol: leave-one-class-out on the multiclass SELECTED_30. Hold one class out
entirely as OOD; train FERL on the rest; at test, separate held-in test points
(ID) from the held-out class (OOD) by AUROC. The held-out class OVERLAPS the
retained feature manifold, so this is the hard near-OOD case where FERL's native
firing/ignorance is blind (rules still fire).

Signals:
  ignorance : native DS ignorance (predict_ds) -- the bounded native readout
  residual  : per-node membership-weighted diagonal Gaussian over the features
              NOT on each node's path (the "model the ignored directions" idea).
              This is what recovers near-OOD from the tree's own structure.
  maha/knn/isoforest : dedicated post-hoc detectors fit on ID train features only
              (a SEPARATE model FERL does not need) -- the fair peer.
  edl_vacuity/edl_entropy : evidential deep learning (Sensoy et al. 2018) fit on
              the same ID train split; vacuity K/S and predictive entropy.

Metrics per signal: AUROC, reject@95 (TNR at 95% ID acceptance, an ROC operating
point computed on the evaluation set), FPR95 (ID false-positive rate when 95% of
OOD inputs are flagged) and AUPR-Out (average precision, OOD = positive).
For ferl-deep, ``ignorance`` is the leaves-only Dempster read-out (FERL-deep).
Global NumPy RNG is seeded per fit so FuzzyCART's learned-mode bootstrap repeats.

Reports mean +/- std over seeds. ferl-compact/performance are FuzzyCART(Fast) with
node_dict_access; ferl-deep is LearnedFuzzyTree (nested-dict nodes, walked
directly). Run from the repo root with the datasci interpreter.
"""
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from ferl.pipeline.ferl_pipeline import make
from ferl.pipeline.run_configs import load_filtered, SELECTED_30
from ds_ood_real import detector_scores
sys.path.insert(0, "experiments/benchmark2")
from edl import EDL

warnings.filterwarnings("ignore")
CONFIGS = ["ferl-compact", "ferl-medium", "ferl-deep"]
COLS = ["ignorance", "residual", "residual_leaf", "residual_chi2", "residual_chi2log",
        "maha", "knn", "isoforest", "softmax", "edl_vacuity", "edl_entropy"]
REJ_COLS = [c + "_rej" for c in COLS]          # novelty-rejection @95% ID acceptance
FPR_COLS = [c + "_fpr95" for c in COLS]        # ID FPR at 95% OOD TPR
AUPR_COLS = [c + "_aupr" for c in COLS]        # AUPR-Out (OOD = positive class)
SEEDS = 3
MAX_TR = 2000                                  # cap ID train for tractable learned fits


def fit_residual_cart(tree, Xtr, D):
    """Per-node residual stats for FuzzyCART(Fast): read _cached_path (path
    features) and existing_membership (training weights) off node_dict_access.
    Returns (stats, leaves) where leaves is the set of leaf-node names."""
    stats, leaves = {}, set()
    for name, node in tree.node_dict_access.items():
        if name == "root":
            continue
        if len(node.get("children", {})) == 0:
            leaves.add(name)
        path_feats = {f for f, _ in node.get("_cached_path", [])}
        free = np.array([f for f in range(D) if f not in path_feats])
        w = node.get("existing_membership")
        if w is None or len(free) == 0 or w.sum() < 1e-6:
            continue
        mu = (w[:, None] * Xtr[:, free]).sum(0) / w.sum()
        var = (w[:, None] * (Xtr[:, free] - mu) ** 2).sum(0) / w.sum() + 1e-6
        stats[name] = (free, mu, var)
    return stats, leaves


def fit_residual_learned(tree, Xtr, D):
    """Per-node residual stats for LearnedFuzzyTree: walk root_ tracking the set
    of ancestor split features and the node's path membership over Xtr.
    Returns (stats, leaves) where leaves is the set of leaf-node names."""
    stats, leaves = {}, set()

    def walk(node, name, pathfeats, m):
        if name != "r":
            if node["leaf"]:
                leaves.add(name)
            if len(pathfeats) < D and m.sum() > 1e-6:
                free = np.array([f for f in range(D) if f not in pathfeats])
                if len(free):
                    mu = (m[:, None] * Xtr[:, free]).sum(0) / m.sum()
                    var = (m[:, None] * (Xtr[:, free] - mu) ** 2).sum(0) / m.sum() + 1e-6
                    stats[name] = (free, mu, var)
        if not node["leaf"]:
            left, right = tree._split(node, Xtr)
            walk(node["L"], name + "_0", pathfeats | {node["f"]}, m * left)
            walk(node["R"], name + "_1", pathfeats | {node["f"]}, m * right)

    walk(tree.root_, "r", set(), np.ones(len(Xtr)))
    return stats, leaves


def reject_at_95(id_scores, ood_scores):
    """Novelty-rejection rate at the operating point that accepts 95% of ID.
    All signals here are oriented so higher = more OOD (AUROC uses them raw as
    the positive score), so tau is the 95th ID percentile and we reject any
    novel-class sample scoring above it. Mirrors the CBM open-world 'Reject'
    column (threshold fixed to accept 95% of retained-class validation)."""
    tau = np.quantile(id_scores, 0.95)
    return float(np.mean(ood_scores > tau))


def fpr_at_95(id_scores, ood_scores):
    """FPR95: fraction of ID inputs flagged when the threshold flags 95% of OOD."""
    tau = np.quantile(ood_scores, 0.05)
    return float(np.mean(id_scores >= tau))


def edl_scores(Xtr, ytr, Xeval, seed):
    """EDL vacuity (K/S) and predictive entropy; higher = more OOD."""
    alpha = EDL(random_state=seed).fit(Xtr, ytr)._alpha(Xeval)
    S = alpha.sum(1)
    P = alpha / S[:, None]
    return {"edl_vacuity": alpha.shape[1] / S,
            "edl_entropy": -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)}


def residual_score(tree, X, stats, transform=None):
    """transform: standardise each node's score to a common scale before averaging
    -- under the node's Gaussian, |free|*z2 ~ chi2_{|free|}, so 'cdf' maps it to
    Uniform(0,1) regardless of |free| (Kriegel et al. 2011 unification); 'logsf'
    uses -log survival instead, same standardisation without the CDF's saturation
    at 1 for strongly atypical points."""
    res = tree.node_activation_matrix(X)          # 3-tuple (FuzzyCART) / 4-tuple (learned)
    M, names = res[0], res[2]
    num = np.zeros(len(X)); den = np.zeros(len(X))
    for k, name in enumerate(names):
        if name not in stats:
            continue
        free, mu, var = stats[name]
        z2 = (((X[:, free] - mu) ** 2) / var).mean(1)
        if transform == "cdf":
            z2 = chi2.cdf(z2 * len(free), df=len(free))
        elif transform == "logsf":
            xdf = z2 * len(free)
            t = -chi2.logsf(xdf, df=len(free))
            z2 = np.where(np.isfinite(t), t, xdf / 2.0)   # sf underflow: -logsf ~ x/2
        num += M[:, k] * z2; den += M[:, k]
    return num / np.clip(den, 1e-9, None)


def one_seed(cfg, seed):
    """Mean-over-held-classes AUROC per (dataset, signal) for one seed."""
    is_learned = cfg == "ferl-deep"
    fit_res = fit_residual_learned if is_learned else fit_residual_cart
    rows = []
    for ds in SELECTED_30:
        try:
            X, y = load_filtered(ds)
        except Exception:
            continue
        classes = np.unique(y)
        if len(classes) < 3 or len(y) < 60:
            continue
        per = {c: [] for c in COLS}
        per_rej = {c: [] for c in COLS}
        per_fpr = {c: [] for c in COLS}
        per_aupr = {c: [] for c in COLS}
        for held in classes:
            id_mask = y != held
            Xid, yid = X[id_mask], y[id_mask]
            _, yid = np.unique(yid, return_inverse=True)
            Xood = X[y == held]
            if len(Xood) < 5 or len(np.unique(yid)) < 2:
                continue
            try:
                Xtr, Xte, ytr, _ = train_test_split(Xid, yid, test_size=0.3,
                                                    random_state=seed, stratify=yid)
            except ValueError:
                continue
            if len(Xtr) > MAX_TR:
                idx = np.random.RandomState(seed).choice(len(Xtr), MAX_TR, replace=False)
                Xtr, ytr = Xtr[idx], ytr[idx]
            D = Xtr.shape[1]
            np.random.seed(seed)
            t = make(cfg, random_state=seed).fit(Xtr, ytr)
            t = getattr(t, "tree_", t)
            stats, leaves = fit_res(t, Xtr, D)
            leaf_stats = {n: v for n, v in stats.items() if n in leaves}
            Xeval = np.vstack([Xte, Xood])
            is_ood = np.r_[np.zeros(len(Xte)), np.ones(len(Xood))]
            ign = (t.predict_ds(Xeval, leaves_only=True) if is_learned
                   else t.predict_ds(Xeval))[-1]
            sig = {"ignorance": np.asarray(ign).ravel(),
                   "residual": residual_score(t, Xeval, stats),
                   "residual_leaf": residual_score(t, Xeval, leaf_stats),
                   "residual_chi2": residual_score(t, Xeval, stats, transform="cdf"),
                   "residual_chi2log": residual_score(t, Xeval, stats, transform="logsf"),
                   "softmax": -(lambda P: (P * np.log(np.clip(P, 1e-12, 1))).sum(1))(
                       np.asarray(t.predict_proba(Xeval))),
                   **detector_scores(Xtr, ytr, Xeval),
                   **edl_scores(Xtr, ytr, Xeval, seed)}
            for c in COLS:
                if np.std(sig[c]) > 1e-9:
                    idv, oodv = sig[c][is_ood == 0], sig[c][is_ood == 1]
                    per[c].append(roc_auc_score(is_ood, sig[c]))
                    per_rej[c].append(reject_at_95(idv, oodv))
                    per_fpr[c].append(fpr_at_95(idv, oodv))
                    per_aupr[c].append(average_precision_score(is_ood, sig[c]))
        if per["residual"]:
            rows.append([ds, seed]
                        + [np.mean(per[c]) if per[c] else np.nan for c in COLS]
                        + [np.mean(per_rej[c]) if per_rej[c] else np.nan for c in COLS]
                        + [np.mean(per_fpr[c]) if per_fpr[c] else np.nan for c in COLS]
                        + [np.mean(per_aupr[c]) if per_aupr[c] else np.nan for c in COLS])
    return pd.DataFrame(rows, columns=["dataset", "seed"] + COLS + REJ_COLS
                        + FPR_COLS + AUPR_COLS)


def main():
    cfgs = sys.argv[1:] or CONFIGS
    summary = []
    for cfg in cfgs:
        parts = []
        for seed in range(SEEDS):
            df = one_seed(cfg, seed)
            parts.append(df)
            print(f"  [{cfg}] seed {seed}: "
                  + " ".join(f"{c[:4]}={df[c].mean():.3f}" for c in COLS), flush=True)
        full = pd.concat(parts, ignore_index=True)
        full.insert(0, "config", cfg)
        full.to_csv(f"results/ds_ood_residual_{cfg}.csv", index=False)
        # per-seed dataset-mean, then mean +/- std over seeds
        allc = COLS + REJ_COLS + FPR_COLS + AUPR_COLS
        seed_means = full.groupby("seed")[allc].mean()
        print(f"\n=== {cfg}: near-OOD AUROC / reject@95, mean +/- std over {SEEDS} "
              f"seeds ({full['dataset'].nunique()} datasets) ===")
        for c in COLS:
            print(f"  {c:10s} auroc={seed_means[c].mean():.3f}+/-{seed_means[c].std():.3f}"
                  f"  rej={seed_means[c + '_rej'].mean():.3f}+/-{seed_means[c + '_rej'].std():.3f}")
        row = {"config": cfg}
        row.update({c: seed_means[c].mean() for c in allc})
        row.update({f"{c}_std": seed_means[c].std() for c in allc})
        summary.append(row)
    pd.DataFrame(summary).to_csv("results/ds_ood_residual_variants.csv", index=False)
    print("\nwrote results/ds_ood_residual_variants.csv", flush=True)


if __name__ == "__main__":
    main()
