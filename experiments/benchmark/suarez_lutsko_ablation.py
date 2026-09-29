"""
Suarez-Lutsko ablation: fixed trapezoidal partitions vs. globally refit
membership breakpoints, with the tree topology held fixed.

Answers the reviewer question "why keep the fuzzy sets fixed instead of
optimizing the thresholds (Suarez & Lutsko 1999, globally optimal fuzzy trees)?"
empirically. We grow a fixed-partition FERL (FuzzyCART, CCI criterion), then
freeze which (feature, fuzzy set) sits at each node and optimize *only* the
trapezoidal breakpoints by gradient descent on training cross-entropy (a
differentiable re-implementation of the product-path predictor). We report the
change in accuracy and calibration, and the partition drift = how far the
breakpoints moved (the interpretability cost of refitting).

Run from the repo root:
    python experiments/benchmark/suarez_lutsko_ablation.py
    python experiments/benchmark/suarez_lutsko_ablation.py full
Outputs results/suarez_lutsko_ablation.csv
"""
import os
import sys
import csv
import copy
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
import ex_fuzzy.utils as utils
import ex_fuzzy.fuzzy_sets as fs
from sklearn.model_selection import StratifiedKFold, train_test_split

from ferl import FuzzyCART
from ferl.pipeline.run_configs import load_filtered, ALL_DATASETS
from ferl.uncertainty.recalibrate import ece

warnings.filterwarnings("ignore")

CAP = 3000          # subsample large datasets so the tree fit stays fast
MAX_RULES = 12
N_FOLDS = 3
STEPS = 600         # max gradient steps for the breakpoint refit
LR = 2e-3
CLIP = 1.0          # grad-norm clip: shoulder ramps give ~1/GMIN-scale gradients
VAL_FRAC = 0.25     # inner validation split for early stopping (fair refit)
EPS = 1e-8
GMIN = 1e-3         # floor on trapezoid segment widths (keeps sets valid & smooth)

# Continuous-feature, modestly sized datasets where trapezoidal partitions are
# meaningful. `full` switches to the whole KEEL suite.
DATASETS = ["iris", "wine", "wdbc", "wisconsin", "pima", "vehicle", "sonar",
            "ecoli", "glass", "heart", "banana", "phoneme", "segment",
            "newthyroid", "bupa"]


def _inv_softplus(y):
    # stable inverse of softplus for y > 0
    return np.log(np.expm1(np.clip(y, 1e-6, None)))


class RefitTree(torch.nn.Module):
    """Differentiable re-implementation of FuzzyCART's non-gated 'soft'
    all-nodes predictor over a FIXED topology.

    Faithful to ``_predict_proba_all_nodes`` in 'soft' mode: every non-root node
    votes with its (frozen) ``class_probabilities`` consequent, weighted by its
    path membership (product of trapezoids along the path), then the vote is
    normalized. The only learnable parameters are the trapezoidal breakpoints of
    every (feature, fuzzy set) used in the tree. A set is one object, so
    refitting it moves it in every node that uses it -- global membership
    optimization (Suarez-Lutsko) with structure and consequents held fixed.
    """

    def __init__(self, tree: FuzzyCART, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

        nodes = []                                               # (path pairs, cp vector)
        for nd in tree._extract_all_nodes():
            if nd["path_length"] == 0:                           # root excluded in 'soft'
                continue
            store = tree.node_dict_access.get(nd["name"])
            cp = store.get("class_probabilities") if store is not None else None
            if cp is None or len(cp) != n_classes:
                continue
            path = list(zip(nd["path_features"], nd["path_fuzzy_sets"]))
            nodes.append((path, np.asarray(cp, dtype=np.float64)))

        # Collect the unique (feature, set) pairs used and their initial [a,b,c,d].
        used = {}
        for path, _ in nodes:
            for f, s in path:
                used.setdefault((int(f), int(s)),
                                np.asarray(tree.fuzzy_partitions[f][s].membership_parameters,
                                           dtype=np.float64))
        self.key_to_idx = {k: i for i, k in enumerate(used)}
        self.features = [k[0] for k in used]                     # per param-set feature col
        self._init_bpts = np.stack([used[k] for k in used])      # (S,4) initial breakpoints

        # Parametrize each set by a free left edge `a` and 3 positive segment
        # widths via softplus, so a<=b<=c<=d is guaranteed throughout training.
        a0 = self._init_bpts[:, 0]
        gaps = np.diff(self._init_bpts, axis=1)                  # (S,3) b-a, c-b, d-c
        self.a = torch.nn.Parameter(torch.tensor(a0, dtype=torch.float64))
        self.raw_g = torch.nn.Parameter(
            torch.tensor(_inv_softplus(gaps - GMIN), dtype=torch.float64))

        # Nodes as (param-set indices along path, consequent vector).
        self.node_idxs = [[self.key_to_idx[(int(f), int(s))] for f, s in path]
                          for path, _ in nodes]
        self.register_buffer("CP", torch.tensor(np.stack([cp for _, cp in nodes]),
                                                dtype=torch.float64))

    def breakpoints(self):
        g = GMIN + torch.nn.functional.softplus(self.raw_g)     # (S,3) positive widths
        a = self.a
        b = a + g[:, 0]
        c = b + g[:, 1]
        d = c + g[:, 2]
        return a, b, c, d

    def forward(self, X):
        a, b, c, d = self.breakpoints()
        # Membership of every param-set on its own feature column: (S, n).
        xf = X[:, self.features].t()                            # (S, n)
        left = (xf - a[:, None]) / (b - a)[:, None]
        right = (d[:, None] - xf) / (d - c)[:, None]
        mu = torch.clamp(torch.minimum(torch.minimum(left, right),
                                       torch.ones_like(xf)), 0.0, 1.0)   # (S, n)

        n = X.shape[0]
        S = torch.zeros(n, self.n_classes, dtype=X.dtype)
        T = torch.zeros(n, dtype=X.dtype)
        for k, idxs in enumerate(self.node_idxs):
            w = mu[idxs].prod(dim=0) if idxs else torch.ones(n, dtype=X.dtype)
            S = S + w[:, None] * self.CP[k][None, :]
            T = T + w
        return S / (T[:, None] + EPS)


def _proba_np(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float64)).numpy()


def run_fold(Xtr, ytr, Xte, yte, n_classes):
    # Standardize on train stats: trapezoid breakpoints live in feature units, so
    # a common learning rate needs a common scale (accuracy is unaffected).
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    parts = utils.construct_partitions(Xtr, fs.FUZZY_SETS.t1, n_partitions=3,
                                       shape="trapezoid")
    tree = FuzzyCART(fuzzy_partitions=parts, max_rules=MAX_RULES, target_metric="cci")
    tree.fit(Xtr, ytr)
    if len(tree._get_leaves()) < 2:
        return None                                            # degenerate: nothing to refit
    # Use the non-gated 'soft' predictor so the real model matches the (fully
    # differentiable) surrogate; the gated default has a non-differentiable step.
    tree.prediction_mode = "soft"
    tree._cached_all_nodes = tree._extract_all_nodes()

    net = RefitTree(tree, n_classes)
    P_fixed_te = _proba_np(net, Xte)                           # breakpoints at init
    acc_fixed = (P_fixed_te.argmax(1) == yte).mean()
    acc_model = (tree.predict(Xte) == yte).mean()             # sanity: real predictor
    ece_fixed = ece(P_fixed_te, yte)
    tr_acc_fixed = (_proba_np(net, Xtr).argmax(1) == ytr).mean()

    # Global membership refit: optimize breakpoints on train CE with grad-norm
    # clipping and early stopping on an inner validation split (a fair refit,
    # not an unregularized overfit).
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=VAL_FRAC,
                                      random_state=0, stratify=ytr)
    Xa_t, ya_t = torch.tensor(Xa), torch.tensor(ya, dtype=torch.long)
    Xv_t, yv_t = torch.tensor(Xv), torch.tensor(yv, dtype=torch.long)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    # Select the refit by validation ACCURACY (loss as tiebreak): for an accuracy
    # comparison this is the fair criterion, and it guards against a low-CE but
    # accuracy-wrecking collapse on imbalanced data. Init is kept if nothing beats it.
    def val_score():
        with torch.no_grad():
            P = net(Xv_t)
            acc = (P.argmax(1) == yv_t).double().mean().item()
            vl = torch.nn.functional.nll_loss(torch.log(P + EPS), yv_t).item()
        return (acc, -vl)
    best = (val_score(), copy.deepcopy(net.state_dict()))
    for _ in range(STEPS):
        opt.zero_grad()
        loss = torch.nn.functional.nll_loss(torch.log(net(Xa_t) + EPS), ya_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        opt.step()
        sc = val_score()
        if sc > best[0]:
            best = (sc, copy.deepcopy(net.state_dict()))
    net.load_state_dict(best[1])

    P_refit_te = _proba_np(net, Xte)
    acc_refit = (P_refit_te.argmax(1) == yte).mean()
    ece_refit = ece(P_refit_te, yte)
    tr_acc_refit = (_proba_np(net, Xtr).argmax(1) == ytr).mean()

    # Partition drift: per-set breakpoint movement, normalized by feature range.
    with torch.no_grad():
        a, b, c, d = net.breakpoints()
        bpts_refit = torch.stack([a, b, c, d], dim=1).numpy()
    # Range floored at 1 std to avoid a blow-up on near-constant features
    # (features are standardized, so a normal range is a few std).
    rng = np.maximum(Xtr.max(0) - Xtr.min(0), 1.0)
    per_set = np.linalg.norm(bpts_refit - net._init_bpts, axis=1) / \
        (2.0 * rng[net.features])                              # /2 ~ per-breakpoint scale
    drift = float(per_set.mean())

    return dict(n_classes=n_classes, n_rules=len(tree._get_leaves()),
                acc_model=acc_model, acc_fixed=acc_fixed, acc_refit=acc_refit,
                d_acc=acc_refit - acc_fixed, ece_fixed=ece_fixed, ece_refit=ece_refit,
                drift=drift, train_acc_fixed=tr_acc_fixed, train_acc_refit=tr_acc_refit)


def main(full=False):
    datasets = ALL_DATASETS if full else DATASETS
    os.makedirs("results", exist_ok=True)
    csv_path = "results/suarez_lutsko_ablation.csv"
    cols = ["dataset", "fold", "n_classes", "n_rules", "acc_model", "acc_fixed",
            "acc_refit", "d_acc", "ece_fixed", "ece_refit", "drift",
            "train_acc_fixed", "train_acc_refit"]
    print(f"Suarez-Lutsko refit ablation | {len(datasets)} datasets | {N_FOLDS} folds")
    print(f"writing {csv_path}\n")

    rows = []
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for ds in datasets:
            try:
                X, y = load_filtered(ds)
                if len(y) > CAP:
                    rs = np.random.RandomState(0)
                    idx = rs.choice(len(y), CAP, replace=False)
                    X, y = X[idx], y[idx]
                X = X.astype(np.float64)
                n_classes = len(np.unique(y))
                skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
                accs = []
                for k, (tri, tei) in enumerate(skf.split(X, y)):
                    r = run_fold(X[tri], y[tri], X[tei], y[tei], n_classes)
                    if r is None:
                        continue
                    w.writerow([ds, k] + [r[c] for c in cols[2:]])
                    fh.flush()
                    rows.append({"dataset": ds, **r})
                    accs.append(r["d_acc"])
                if accs:
                    print(f"{ds:14s} n={len(y):5d} C={n_classes} | "
                          f"mean d_acc={np.mean(accs):+.4f}")
            except Exception as ex:
                print(f"{ds:14s} FAILED: {ex!r}")

    if rows:
        d_acc = np.array([r["d_acc"] for r in rows])
        drift = np.array([r["drift"] for r in rows])
        d_ece = np.array([r["ece_refit"] - r["ece_fixed"] for r in rows])
        print("\n=== SUMMARY (per fold, n={}) ===".format(len(rows)))
        print(f"accuracy change (refit - fixed): mean {d_acc.mean():+.4f}  "
              f"median {np.median(d_acc):+.4f}  "
              f"wins {int((d_acc > 0).sum())} / losses {int((d_acc < 0).sum())}")
        print(f"ECE change (refit - fixed):      mean {d_ece.mean():+.4f} "
              f"(negative = better calibrated)")
        print(f"partition drift (frac of range): mean {drift.mean():.3f}  "
              f"median {np.median(drift):.3f}  max {drift.max():.3f}")


if __name__ == "__main__":
    main(full=(len(sys.argv) > 1 and sys.argv[1] == "full"))
