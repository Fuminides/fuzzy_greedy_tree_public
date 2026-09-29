"""
Experiment 2: real-data OOD confirmation. Leave-one-class-out (the hard,
overlapping OOD case) on the multiclass SELECTED_30; AUROC for ID vs the held-out
class. Adds the firing-Phi signal that won on synthetic, vs softmax baselines.

  entropy / one_minus_maxp : softmax (what conformal-on-softmax sees)
  ignorance / set_size     : credal (top-p routed)
  neg_firing = 1/(1+Phi)   : raw total rule firing -- softmax discards this
  maha / knn / isoforest   : dedicated post-hoc OOD detectors, fit on ID train
                             features only (a SEPARATE model FERL does not need)

Synthetic showed firing >> softmax for geometric OOD; leave-class-out is harder
(OOD class overlaps the feature space) -- honest test of whether it still helps.
The dedicated detectors are the fair peer: FERL's firing signal comes free from
the classifier, theirs requires a purpose-built density/novelty model.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from ferl.pipeline.ferl_pipeline import make
from ferl.pipeline.run_configs import load_filtered, SELECTED_30
from ds_coverage_levers import ds_combine
from ds_synthetic_tau import topp

warnings.filterwarnings("ignore")
SIGNALS = ["entropy", "one_minus_maxp", "ignorance", "set_size", "neg_firing",
           "conflict_disp", "conflict_ds", "residual"]
DETECTORS = ["maha", "knn", "isoforest"]
COLUMNS = SIGNALS + DETECTORS
TAU_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]


def entropy(P):
    return -(P * np.log(np.clip(P, 1e-12, 1.0))).sum(1)


def fit_residual(tree, Xtr, D):
    """Per-node membership-weighted mean/var over the features NOT on the node's
    path -- the directions the tree ignored. Blind-spot near-OOD detector."""
    stats = {}
    for name, node in tree.node_dict_access.items():
        if name == "root":
            continue
        path_feats = {f for f, _ in node.get("_cached_path", [])}
        free = np.array([f for f in range(D) if f not in path_feats])
        w = node.get("existing_membership")
        if w is None or len(free) == 0:
            continue
        sw = w.sum()
        if sw < 1e-6:
            continue
        mu = (w[:, None] * Xtr[:, free]).sum(0) / sw
        var = (w[:, None] * (Xtr[:, free] - mu) ** 2).sum(0) / sw + 1e-6
        stats[name] = (free, mu, var)
    return stats


def residual_score(tree, X, stats):
    """Membership-weighted diagonal Mahalanobis over ignored features, averaged
    over the nodes a sample activates. Higher = more atypical off-path."""
    M, _, names = tree.node_activation_matrix(X)
    num = np.zeros(len(X)); den = np.zeros(len(X))
    for k, name in enumerate(names):
        if name not in stats:
            continue
        free, mu, var = stats[name]
        z2 = (((X[:, free] - mu) ** 2) / var).mean(1)
        num += M[:, k] * z2; den += M[:, k]
    return num / np.clip(den, 1e-9, None)


def detector_scores(Xtr, ytr, Xeval):
    """Dedicated post-hoc OOD detectors, fit on ID train features only.
    Returns scores where higher = more OOD (matching the signal convention)."""
    sc = StandardScaler().fit(Xtr)
    Ztr, Zev = sc.transform(Xtr), sc.transform(Xeval)
    # class-conditional Mahalanobis (tied covariance), min distance over classes
    cov = np.cov(Ztr.T) + 1e-6 * np.eye(Ztr.shape[1])
    prec = np.linalg.pinv(cov)
    dists = [np.einsum("ij,jk,ik->i", Zev - Ztr[ytr == c].mean(0), prec,
                       Zev - Ztr[ytr == c].mean(0)) for c in np.unique(ytr)]
    maha = np.min(dists, axis=0)
    # kNN distance to the k-th nearest ID point
    k = min(20, len(Ztr) - 1)
    knn = NearestNeighbors(n_neighbors=k).fit(Ztr).kneighbors(Zev)[0][:, -1]
    # Isolation Forest anomaly score (negate so higher = more anomalous)
    iso = -IsolationForest(random_state=0).fit(Ztr).score_samples(Zev)
    return {"maha": maha, "knn": knn, "isoforest": iso}


def vote_dispersion(M, cons):
    """Firing-weighted variance of the per-rule consequents = how much the rules
    that fire DISAGREE (open-world conflict proxy; bounded, doesn't saturate)."""
    w = M / np.clip(M.sum(1, keepdims=True), 1e-12, None)     # (N,K)
    v = w @ cons                                              # (N,C) weighted mean vote
    return np.clip(w @ (cons ** 2).sum(1) - (v ** 2).sum(1), 0, None)


def dempster_conflict(M, cons):
    """True Dempster conflict mass K = 1 - (unnormalised focal mass)."""
    one_minus = 1.0 - M
    term = M[:, :, None] * cons[None, :, :] + one_minus[:, :, None]   # (N,K,C)
    Qc = term.prod(1)
    Qt = one_minus.prod(1)
    total = np.clip(Qc - Qt[:, None], 0, None).sum(1) + Qt
    return np.clip(1.0 - total, 0, 1)


def main():
    rows = []
    for ds in SELECTED_30:
        try:
            X, y = load_filtered(ds)
        except Exception:
            continue
        classes = np.unique(y)
        if len(classes) < 3 or len(y) < 60:
            continue
        per = {s: [] for s in COLUMNS}
        for held in classes:
            id_mask = y != held
            Xid, yid = X[id_mask], y[id_mask]
            _, yid = np.unique(yid, return_inverse=True)
            Xood = X[y == held]
            if len(Xood) < 5 or len(np.unique(yid)) < 2:
                continue
            try:
                Xtr, Xte, ytr, yte = train_test_split(Xid, yid, test_size=0.3,
                                                      random_state=0, stratify=yid)
            except ValueError:
                continue
            f = make("ferl-compact", random_state=0).fit(Xtr, ytr).tree_
            C = len(np.unique(ytr))
            res_stats = fit_residual(f, Xtr, Xtr.shape[1])
            _, cons, _ = f.node_activation_matrix(Xtr[:1])
            # tau* accuracy-preserving on a slice of train
            M_tr = f.node_activation_matrix(Xtr)[0]
            base = (ds_combine(M_tr, cons)[0].argmax(1) == ytr).mean()
            tau = 1.0
            for t in TAU_GRID:
                ch, rh = topp(cons, t)
                if (ds_combine(M_tr * rh[None, :], ch)[0].argmax(1) == ytr).mean() >= base - 0.01:
                    tau = t; break
            ch, rh = topp(cons, tau)

            Xeval = np.vstack([Xte, Xood])
            is_ood = np.r_[np.zeros(len(Xte)), np.ones(len(Xood))]
            P = f.predict_proba(Xeval)
            phi = f.firing_strength(Xeval)
            M_ev = f.node_activation_matrix(Xeval)[0]
            bel, pl = ds_combine(M_ev * rh[None, :], ch)
            sig = {
                "entropy": entropy(P),
                "one_minus_maxp": 1.0 - P.max(1),
                "ignorance": (pl - bel)[:, 0],
                "set_size": (pl >= bel.max(1, keepdims=True) - 1e-12).sum(1).astype(float),
                "neg_firing": 1.0 / (1.0 + phi),
                "conflict_disp": vote_dispersion(M_ev, cons),
                "conflict_ds": dempster_conflict(M_ev, cons),
                "residual": residual_score(f, Xeval, res_stats),
                **detector_scores(Xtr, ytr, Xeval),
            }
            for s in COLUMNS:
                if np.std(sig[s]) > 1e-9:
                    per[s].append(roc_auc_score(is_ood, sig[s]))
        if per["neg_firing"]:
            rows.append([ds] + [np.mean(per[s]) if per[s] else np.nan for s in COLUMNS])
            print(f"  {ds}: " + " ".join(f"{s[:4]}={np.mean(per[s]):.2f}" for s in COLUMNS if per[s]), flush=True)

    df = pd.DataFrame(rows, columns=["dataset"] + COLUMNS)
    df.to_csv("results/ds_ood_real.csv", index=False)
    print(f"\n=== MEAN leave-class-out OOD AUROC ({len(df)} multiclass datasets) ===")
    print(df[COLUMNS].mean().round(4).to_string())
    best_soft = df[["entropy", "one_minus_maxp"]].max(1)
    best_det = df[DETECTORS].max(1)
    print(f"\nneg_firing beats best softmax on {(df['neg_firing'] > best_soft).sum()}/{len(df)} datasets")
    print(f"neg_firing beats best dedicated detector on {(df['neg_firing'] > best_det).sum()}/{len(df)} datasets")
    best_native = df[["ignorance", "set_size", "neg_firing", "conflict_disp", "conflict_ds"]].max(1)
    print(f"residual beats best other native signal on {(df['residual'] > best_native).sum()}/{len(df)} datasets")
    print(f"residual beats best dedicated detector on {(df['residual'] > best_det).sum()}/{len(df)} datasets")


if __name__ == "__main__":
    main()
