"""CUB-CBM extras: intervention, reliability-discounted fusion, calibration.

E5  Test-time concept intervention (Koh et al. style): replace the k "most
    useful" predicted concepts with oracle-derived values (train-score
    percentiles, so intervened values stay on the model's input manifold) and
    trace accuracy vs k. Policies: random, model global importance, and
    per-sample path order (ferl-deep, decision_tree). For ferl-deep the
    native credal set size is traced too (does intervention collapse
    uncertainty?).

E6  Reliability-discounted evidential fusion: per-concept detector reliability
    rho_j estimated on the validation split, per-leaf reliability = product of
    rho over the leaf's path concepts, passed as a Dempster-Shafer discount.
    Compared against the plain native set under clean / corrupted-concepts /
    detector-swap conditions. ferl-deep only.

E7  Per-concept isotonic calibration of detector scores (fit on val) as an
    ablation: every method refit on calibrated vs raw scores.

E9  Adaptive support-guided intervention for ferl-deep: unresolved samples
    query the nearest repair to a zero-support route, or otherwise the active
    concept with greatest one-step expected uncertainty reduction. The route is
    recomputed after every verified answer and samples stop querying once their
    native credal set is a singleton.

E10 Matched selective intervention comparison: logistic regression receives a
    validation-only confidence threshold matching FERL's validation acceptance
    rate, then uses adaptive expected information gain or fixed coefficient
    importance. FERL additionally gets a utility-aware policy that combines
    detector reliability, validation route purity, and the validation-estimated
    net correction of verifying each route concept. Both heads query only
    rejected samples and stop on acceptance. Correctness transitions expose
    wrong singleton resolutions.

E11 Concept-budget frontier: rank concepts by mutual information using the
    training split only, then refit FERL-deep, CART, and logistic regression on
    exactly the same top-m concept subsets. This separates head quality from
    access to a larger concept vocabulary.

Run from the repo root, e.g.:
    python3 experiments/cub_cbm/run_cub_cbm_extras.py \
        --artifact-dir results/cub_cbm_artifacts/cub_koh112_20 --subset 20 \
        --detector-seeds 0 1 2 --experiments all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from experiments.cub_cbm.run_cub_cbm import (
    _append_csv,
    create_smoke_artifact,
    describe_ferl_rules,
    encoded_splits,
    make_model,
    read_bundle,
)
from experiments.cub_cbm.adaptive_intervention import (
    adaptive_lr_query_trajectory,
    adaptive_query_trajectory,
    adaptive_reliability_query_trajectory,
    confidence_acceptance,
    confidence_threshold_for_coverage,
    fit_feature_intervention_utility,
    fit_route_reliability,
    fixed_order_lr_query_trajectory,
    fixed_order_query_trajectory,
    trace_dominant_route,
)
from ferl.core.learned_tree import LearnedFuzzyTree, _ds_combine

E5_METHODS = ("decision_tree", "logistic_regression", "ferl-medium", "ferl-deep")


# --- shared helpers ----------------------------------------------------------

def make_model_x(method: str, seed: int, learned_depth: int):
    if method == "ferl-deep":
        return LearnedFuzzyTree(max_depth=learned_depth, random_state=seed)
    return make_model(method, seed)


def load_pair(artifact_dir: Path, subset: str, detector_seed: int):
    """Aligned (oracle, predicted) splits for one detector seed."""
    ob = read_bundle(artifact_dir, source="oracle", subset=subset)
    pb = read_bundle(artifact_dir, source="predicted", detector_seed=detector_seed, subset=subset)
    _, otr, ova, ote = encoded_splits(ob)
    _, ptr, pva, pte = encoded_splits(pb)
    for a, b in ((otr, ptr), (ova, pva), (ote, pte)):
        assert np.array_equal(a.y, b.y), "oracle/predicted splits are not sample-aligned"
    return (otr, ova, ote), (ptr, pva, pte)


def intervention_values(oracle_train_C: np.ndarray, pred_train_C: np.ndarray):
    """Koh-style intervention targets: per concept, the 95th percentile of the
    detector's train scores among oracle-positive samples (for c=1) and the 5th
    among oracle-negatives (for c=0), so intervened inputs stay on-manifold."""
    n_c = pred_train_C.shape[1]
    v1 = np.empty(n_c)
    v0 = np.empty(n_c)
    for j in range(n_c):
        pos = pred_train_C[oracle_train_C[:, j] >= 0.5, j]
        neg = pred_train_C[oracle_train_C[:, j] < 0.5, j]
        v1[j] = np.percentile(pos, 95) if len(pos) else np.percentile(pred_train_C[:, j], 95)
        v0[j] = np.percentile(neg, 5) if len(neg) else np.percentile(pred_train_C[:, j], 5)
    return v0, v1


def concept_reliability(oracle_val_C: np.ndarray, pred_val_C: np.ndarray) -> np.ndarray:
    """rho_j in [0,1]: 2*balanced_acc-1 of the thresholded detector score.
    A useless concept head (bacc 0.5) gets rho 0 -> full discount to ignorance."""
    n_c = pred_val_C.shape[1]
    rho = np.zeros(n_c)
    for j in range(n_c):
        c = (oracle_val_C[:, j] >= 0.5).astype(int)
        pred = (pred_val_C[:, j] >= 0.5).astype(int)
        if len(np.unique(c)) < 2:
            rho[j] = max(0.0, 2.0 * float((pred == c).mean()) - 1.0)
            continue
        bacc = balanced_accuracy_score(c, pred)
        rho[j] = max(0.0, 2.0 * bacc - 1.0)
    return rho


def learned_path_features(model: LearnedFuzzyTree, name: str) -> list[int]:
    """Split features along the path to node ``name`` ('r_0_1...')."""
    node, feats = model.root_, []
    for d in name.split("_")[1:]:
        feats.append(int(node["f"]))
        node = node["L"] if d == "0" else node["R"]
    return feats


def predict_set_learned(model: LearnedFuzzyTree, X: np.ndarray, rho_concepts=None,
                        rho_agg: str = "product", batch: int = 256):
    """Native leaves-only Dempster set, optionally reliability-discounted.

    ``rho_agg`` maps the path's per-concept reliabilities to one per-leaf
    discount: 'product' (independent-source, very cautious on deep trees),
    'min' (a rule is as reliable as its weakest symbol), 'mean' (graded).
    Batched over samples: _ds_combine materialises an (n, leaves, classes)
    tensor, which OOMs on full CUB (5794 x ~200 x 200) if done in one shot."""
    X = np.asarray(X, float)
    sets_parts, ign_parts = [], []
    rel = None
    for start in range(0, len(X), batch):
        M, cons, names, support = model.node_activation_matrix(X[start:start + batch])
        keep = np.flatnonzero(model.leaf_mask(names))
        M, cons, support = M[:, keep], cons[keep], support[keep]
        names = [names[i] for i in keep]
        if rho_concepts is not None and rel is None:
            agg = {"product": np.prod, "min": np.min, "mean": np.mean}[rho_agg]
            rel = np.array([
                float(agg([rho_concepts[f] for f in sorted(set(learned_path_features(model, n)))]))
                for n in names
            ])
        _, bel, pl, ign = _ds_combine(M, cons, names, model.C, rule="dempster",
                                      support=support, reliability=rel)
        sets_parts.append(pl >= bel.max(1, keepdims=True) - 1e-12)
        ign_parts.append(ign)
    return np.concatenate(sets_parts, axis=0), np.concatenate(ign_parts, axis=0)


def set_metrics(sets: np.ndarray, y: np.ndarray, classes: np.ndarray) -> dict:
    sizes = sets.sum(1)
    col = {c: i for i, c in enumerate(classes)}
    hit = np.array([sets[i, col[y[i]]] for i in range(len(y))], dtype=bool)
    singleton = sizes == 1
    acc_singleton = float(hit[singleton].mean()) if singleton.any() else np.nan
    return dict(
        coverage=float(hit.mean()),
        avg_set_size=float(sizes.mean()),
        abstention_rate=float((sizes > 1).mean()),
        accepted_accuracy=acc_singleton,
    )


def global_importance_order(model, method: str, concept_names: list[str], n_c: int) -> np.ndarray:
    if method == "decision_tree":
        scores = model.feature_importances_
    elif method in ("logistic_regression", "logistic_l1"):
        scores = np.abs(model.coef_).mean(0)
    elif method == "ferl-deep":
        scores = np.zeros(n_c)

        def walk(node):
            if node["leaf"]:
                return
            scores[int(node["f"])] += float(node["support"])
            walk(node["L"])
            walk(node["R"])

        walk(model.root_)
    else:  # FuzzyCART pipeline
        scores = np.zeros(n_c)
        _, importance = describe_ferl_rules(model, concept_names, max_rules=10**9)
        for row in importance:
            scores[int(row["concept_index"])] = float(row["coverage_weight"]) + 1e-9
    return np.argsort(-scores, kind="stable")


def per_sample_path_order(model, method: str, X: np.ndarray, global_order: np.ndarray) -> np.ndarray:
    """(n, n_concepts) per-sample concept ordering: own path first (root->leaf),
    then the model's global importance for the rest."""
    n, n_c = X.shape
    orders = np.empty((n, n_c), dtype=int)
    if method == "decision_tree":
        node_feature = model.tree_.feature
        indicator = model.decision_path(X)
        for i in range(n):
            path_nodes = indicator.indices[indicator.indptr[i]:indicator.indptr[i + 1]]
            feats = [node_feature[t] for t in path_nodes if node_feature[t] >= 0]
            orders[i] = _pad_order(feats, global_order, n_c)
        return orders
    # ferl-deep: follow the stronger branch and stop rather than inventing a
    # branch if bounded support has made both memberships zero.
    for i in range(n):
        trace = trace_dominant_route(model, X[i])
        feats = [step.feature for step in trace.steps]
        if trace.failure_feature is not None:
            feats.append(trace.failure_feature)
        orders[i] = _pad_order(feats, global_order, n_c)
    return orders


def _pad_order(feats: list[int], global_order: np.ndarray, n_c: int) -> np.ndarray:
    seen, out = set(), []
    for f in list(feats) + list(global_order):
        if f not in seen:
            seen.add(f)
            out.append(f)
    return np.asarray(out[:n_c], dtype=int)


def apply_intervention(X_pred: np.ndarray, X_target: np.ndarray, orders: np.ndarray, k: int) -> np.ndarray:
    """Replace each sample's first-k concepts (per its ordering) with targets."""
    if k <= 0:
        return X_pred
    X = X_pred.copy()
    k = min(k, X.shape[1])
    rows = np.repeat(np.arange(len(X)), k)
    cols = orders[:, :k].ravel()
    X[rows, cols] = X_target[rows, cols]
    return X


# --- E5: test-time concept intervention --------------------------------------

def _used_concepts(model, method: str, n_c: int) -> int:
    """How many concepts the fitted model actually reads (for the matched-budget
    comparison); n_c when the model is dense in the concepts."""
    if method == "ferl-deep":
        return len(_tree_features(model))
    if method == "decision_tree":
        return int((model.feature_importances_ > 0).sum())
    if method in ("logistic_regression", "logistic_l1"):
        return int((np.abs(model.coef_).max(0) > 1e-8).sum())
    return n_c


def run_e5(pair, *, subset, detector_seed, concept_names, seed, ks, learned_depth, output_dir):
    (otr, ova, ote), (ptr, pva, pte) = pair
    n, n_c = pte.C.shape
    classes = np.arange(len(np.unique(np.concatenate([ptr.y, pte.y]))))
    # Every model here consumes a per-concept isotonic-calibrated detector
    # (fit on val against the oracle); raw scores are reserved for E7, whose
    # whole point is the raw-vs-calibrated contrast. Targets are percentiles of
    # the *calibrated* train scores, so intervened inputs stay on the manifold
    # the models were actually fit on.
    cals = _fit_concept_calibrators(ova.C, pva.C)
    Ctr, Cte = (_apply_calibrators(cals, A) for A in (ptr.C, pte.C))
    v0, v1 = intervention_values(otr.C, Ctr)
    X_target = np.where(ote.C >= 0.5, v1[None, :], v0[None, :])
    rng = np.random.default_rng(seed)
    random_orders = np.argsort(rng.random((n, n_c)), axis=1)

    fitted = []
    for method in E5_METHODS:
        print(f"[cbm:e5] fit subset={subset} det={detector_seed} method={method}", flush=True)
        model = make_model_x(method, seed, learned_depth)
        model.fit(Ctr, ptr.y)
        fitted.append((method, model))
    # concept-matched sparse head: an L1 LR restricted to as many concepts as
    # the FERL tree actually uses, so the LR comparison is not won on breadth.
    budget = len(_tree_features(dict(fitted)["ferl-deep"]))
    print(f"[cbm:e5] fit subset={subset} det={detector_seed} method=logistic_l1", flush=True)
    fitted.append(("logistic_l1", _fit_sparse_lr(Ctr, ptr.y, budget, seed)))

    rows = []
    for method, model in fitted:
        g_order = global_importance_order(model, method, concept_names, n_c)
        policies = {
            "random": random_orders,
            "importance": np.broadcast_to(g_order, (n, n_c)),
        }
        if method in ("decision_tree", "ferl-deep"):
            policies["path"] = per_sample_path_order(model, method, Cte, g_order)
        for policy, orders in policies.items():
            for k in ks:
                Xk = apply_intervention(Cte, X_target, orders, k)
                pred = model.predict(Xk)
                row = dict(subset=subset, detector_seed=detector_seed, method=method,
                           concepts="calibrated", policy=policy, k=int(k),
                           n_concepts=int(_used_concepts(model, method, n_c)),
                           accuracy=accuracy_score(pte.y, pred),
                           macro_f1=f1_score(pte.y, pred, average="macro", zero_division=0))
                if method == "ferl-deep":
                    sets, ign = predict_set_learned(model, Xk)
                    row.update(set_metrics(sets, pte.y, classes))
                    row["mean_ignorance"] = float(ign.mean())
                rows.append(row)
            print(f"[cbm:e5] done method={method} policy={policy}", flush=True)
    df = pd.DataFrame(rows)
    _append_csv(output_dir / "e5_intervention.csv", df)
    plot_e5(df, output_dir / f"e5_intervention_{subset}_seed{detector_seed}.png")
    return df


def plot_e5(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    styles = {"random": ":", "importance": "-", "path": "--"}
    for (method, policy), g in df.groupby(["method", "policy"]):
        g = g.sort_values("k")
        axes[0].plot(g["k"], g["accuracy"], styles[policy], marker="o", ms=3,
                     label=f"{method}:{policy}")
    axes[0].set_xlabel("# concepts intervened")
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=6)
    lf = df[(df.method == "ferl-deep") & df.avg_set_size.notna()] if "avg_set_size" in df else pd.DataFrame()
    for policy, g in lf.groupby("policy"):
        g = g.sort_values("k")
        axes[1].plot(g["k"], g["avg_set_size"], styles[policy], marker="o", ms=3, label=policy)
    axes[1].set_xlabel("# concepts intervened")
    axes[1].set_ylabel("native credal set size (ferl-deep)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- E9: adaptive support-guided intervention -------------------------------

def _e9_metrics(model, X, y, classes, query_counts, initial_abstain):
    prediction = model.predict(X)
    sets, ignorance = predict_set_learned(model, X)
    sizes = sets.sum(1)
    singleton = sizes == 1
    initial_count = int(initial_abstain.sum())
    recovered = initial_abstain & singleton
    correct_recovery = recovered & (prediction == y)
    row = dict(
        accuracy=accuracy_score(y, prediction),
        macro_f1=f1_score(y, prediction, average="macro", zero_division=0),
        resolution_rate=float(singleton.mean()),
        recovery_rate=(float(recovered.sum() / initial_count) if initial_count else 1.0),
        correct_recovery_rate=(
            float(correct_recovery.sum() / initial_count) if initial_count else 1.0
        ),
        mean_queries=float(np.mean(query_counts)),
        mean_queries_initial_abstained=(
            float(np.mean(query_counts[initial_abstain])) if initial_count else 0.0
        ),
        mean_ignorance=float(ignorance.mean()),
        **set_metrics(sets, y, classes),
    )
    return row


def run_e9(pair, *, subset, detector_seed, concept_names, seed, ks,
           learned_depth, output_dir, calibrate_concepts=False):
    """Compare adaptive FERL queries with the fixed E5 query orderings."""
    (otr, ova, ote), (ptr, pva, pte) = pair
    n, n_concepts = pte.C.shape
    classes = np.arange(len(np.unique(np.concatenate([ptr.y, pte.y]))))

    concept_variant = "calibrated" if calibrate_concepts else "raw"
    if calibrate_concepts:
        calibrators = _fit_concept_calibrators(ova.C, pva.C)
        model_train, model_test = (
            _apply_calibrators(calibrators, values) for values in (ptr.C, pte.C)
        )
    else:
        model_train, model_test = ptr.C, pte.C
    # Verified values are train-score percentiles on the same scale as the
    # selected raw/calibrated model inputs, not literal binary edits.
    value_absent, value_present = intervention_values(otr.C, model_train)
    verified = np.where(ote.C >= 0.5, value_present[None, :], value_absent[None, :])

    print(
        f"[cbm:e9] fit subset={subset} det={detector_seed} concepts={concept_variant} "
        "method=ferl-deep",
        flush=True,
    )
    model = make_model_x("ferl-deep", seed, learned_depth)
    model.fit(model_train, ptr.y)
    global_order = global_importance_order(
        model, "ferl-deep", concept_names, n_concepts,
    )
    path_orders = per_sample_path_order(
        model, "ferl-deep", model_test, global_order,
    )
    rng = np.random.default_rng(seed)
    random_orders = np.argsort(rng.random((n, n_concepts)), axis=1)
    fixed_orders = {
        "random": random_orders,
        "importance": np.broadcast_to(global_order, (n, n_concepts)),
        "path": path_orders,
    }

    initial_sets, _ = predict_set_learned(model, model_test)
    initial_abstain = initial_sets.sum(1) > 1
    budgets = sorted(set(int(k) for k in ks if int(k) >= 0))
    if not budgets:
        budgets = [0]
    max_budget = min(max(budgets), n_concepts)
    trajectory = adaptive_query_trajectory(
        model,
        model_test,
        verified,
        value_absent,
        value_present,
        max_queries=max_budget,
        global_order=global_order,
    )
    fixed_trajectories = {
        policy: fixed_order_query_trajectory(
            model,
            model_test,
            verified,
            orders,
            max_queries=max_budget,
        )
        for policy, orders in fixed_orders.items()
    }

    rows = []
    for budget in budgets:
        budget = min(budget, n_concepts)
        snapshot = trajectory[budget]
        adaptive_row = dict(
            subset=subset,
            detector_seed=detector_seed,
            method="ferl-deep",
            policy="adaptive_support",
            concepts=concept_variant,
            k=budget,
            initial_abstention_rate=float(initial_abstain.mean()),
            support_backtrack_queries=snapshot.reason_counts.get("support_backtrack", 0),
            expected_uncertainty_queries=snapshot.reason_counts.get("expected_uncertainty", 0),
            global_fallback_queries=snapshot.reason_counts.get("global_fallback", 0),
            **_e9_metrics(
                model, snapshot.X, pte.y, classes,
                snapshot.query_counts, initial_abstain,
            ),
        )
        rows.append(adaptive_row)

        for policy, fixed_trajectory in fixed_trajectories.items():
            fixed_snapshot = fixed_trajectory[budget]
            rows.append(dict(
                subset=subset,
                detector_seed=detector_seed,
                method="ferl-deep",
                policy=policy,
                concepts=concept_variant,
                k=budget,
                initial_abstention_rate=float(initial_abstain.mean()),
                support_backtrack_queries=0,
                expected_uncertainty_queries=0,
                global_fallback_queries=0,
                **_e9_metrics(
                    model, fixed_snapshot.X, pte.y, classes,
                    fixed_snapshot.query_counts, initial_abstain,
                ),
            ))
        print(f"[cbm:e9] evaluated query budget={budget}", flush=True)

    df = pd.DataFrame(rows)
    output_csv = output_dir / "e9_adaptive_intervention.csv"
    # Migrate the first development-run artifact, which predated the explicit
    # raw/calibrated column and used calibrated inputs.
    if output_csv.exists():
        previous = pd.read_csv(output_csv)
        if "concepts" not in previous:
            previous["concepts"] = "calibrated"
            previous.to_csv(output_csv, index=False)
    if output_csv.exists():
        previous = pd.read_csv(output_csv)
        keys = ["subset", "detector_seed", "concepts", "policy", "k"]
        combined = pd.concat([previous, df], ignore_index=True)
        # CLI subset labels arrive as strings, whereas pandas may infer a
        # numeric label when reloading the existing CSV. Normalize before the
        # upsert so reruns replace rather than duplicate the same experiment.
        combined["subset"] = combined["subset"].astype(str)
        combined = combined.drop_duplicates(subset=keys, keep="last")
        combined.to_csv(output_csv, index=False)
    else:
        df.to_csv(output_csv, index=False)
    plot_e9(
        df,
        output_dir / f"e9_adaptive_intervention_{subset}_{concept_variant}_seed{detector_seed}.pdf",
    )
    return df


def plot_e9(df: pd.DataFrame, path: Path) -> None:
    colors = {
        "adaptive_support": "#D55E00",
        "path": "#0072B2",
        "importance": "#009E73",
        "random": "#777777",
    }
    labels = {
        "adaptive_support": "adaptive support",
        "path": "fixed path",
        "importance": "global",
        "random": "random",
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for policy, group in df.groupby("policy"):
        group = group.sort_values("mean_queries_initial_abstained")
        axes[0].plot(
            group["mean_queries_initial_abstained"], group["recovery_rate"],
            marker="o", ms=3, lw=1.5, color=colors[policy], label=labels[policy],
        )
        group = group.sort_values("mean_queries")
        axes[1].plot(
            group["mean_queries"], group["accuracy"], marker="o", ms=3,
            lw=1.5, color=colors[policy], label=labels[policy],
        )
    axes[0].set_xlabel("mean queries on initially abstained samples")
    axes[0].set_ylabel("fraction resolved")
    axes[1].set_xlabel("mean queries per test sample")
    axes[1].set_ylabel("accuracy")
    for axis in axes:
        axis.grid(True, ls=":", alpha=0.4)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# --- E10: matched FERL / LR adaptive intervention ---------------------------

def _selective_intervention_metrics(
    prediction: np.ndarray,
    y: np.ndarray,
    accepted: np.ndarray,
    query_counts: np.ndarray,
    initial_rejected: np.ndarray,
    initial_prediction: np.ndarray,
) -> dict:
    initial_count = int(initial_rejected.sum())
    recovered = initial_rejected & accepted
    correct_recovery = recovered & (prediction == y)
    wrong_recovery = recovered & (prediction != y)
    initially_wrong = initial_rejected & (initial_prediction != y)
    initially_correct = initial_rejected & (initial_prediction == y)
    wrong_to_correct = initially_wrong & (prediction == y)
    correct_to_wrong = initially_correct & (prediction != y)
    total_queries = int(np.asarray(query_counts, dtype=int).sum())
    return dict(
        accuracy=accuracy_score(y, prediction),
        macro_f1=f1_score(y, prediction, average="macro", zero_division=0),
        resolution_rate=float(accepted.mean()),
        recovery_rate=(float(recovered.sum() / initial_count) if initial_count else 1.0),
        correct_recovery_rate=(
            float(correct_recovery.sum() / initial_count) if initial_count else 1.0
        ),
        wrong_singleton_rate=(
            float(wrong_recovery.sum() / initial_count) if initial_count else 0.0
        ),
        wrong_acceptance_rate=(
            float(wrong_recovery.sum() / initial_count) if initial_count else 0.0
        ),
        wrong_to_correct_rate=(
            float(wrong_to_correct.sum() / initial_count) if initial_count else 0.0
        ),
        correct_to_wrong_rate=(
            float(correct_to_wrong.sum() / initial_count) if initial_count else 0.0
        ),
        net_correction_rate=(
            float((wrong_to_correct.sum() - correct_to_wrong.sum()) / initial_count)
            if initial_count else 0.0
        ),
        wrong_to_correct_given_initially_wrong=(
            float(wrong_to_correct.sum() / initially_wrong.sum())
            if initially_wrong.any() else 0.0
        ),
        correct_to_wrong_given_initially_correct=(
            float(correct_to_wrong.sum() / initially_correct.sum())
            if initially_correct.any() else 0.0
        ),
        correct_recoveries_per_query=(
            float(correct_recovery.sum() / total_queries) if total_queries else np.nan
        ),
        net_corrections_per_query=(
            float((wrong_to_correct.sum() - correct_to_wrong.sum()) / total_queries)
            if total_queries else np.nan
        ),
        accepted_accuracy=(
            float((prediction[accepted] == y[accepted]).mean())
            if accepted.any() else np.nan
        ),
        mean_queries=float(np.mean(query_counts)),
        mean_queries_initial_abstained=(
            float(np.mean(query_counts[initial_rejected])) if initial_count else 0.0
        ),
    )


def _accepted_accuracy(prediction: np.ndarray, y: np.ndarray,
                       accepted: np.ndarray) -> float:
    if not accepted.any():
        return np.nan
    return float((prediction[accepted] == y[accepted]).mean())


def _learned_tree_shape(model: LearnedFuzzyTree) -> dict[str, int]:
    """Structural metadata needed to distinguish nominal and realised depth."""
    def visit(node: dict, depth: int) -> tuple[int, int, int]:
        if node["leaf"]:
            return 1, 1, depth
        left = visit(node["L"], depth + 1)
        right = visit(node["R"], depth + 1)
        return (
            1 + left[0] + right[0],
            left[1] + right[1],
            max(left[2], right[2]),
        )

    nodes, leaves, depth = visit(model.root_, 0)
    return dict(
        ferl_max_depth=int(model.max_depth),
        ferl_actual_depth=depth,
        ferl_nodes=nodes,
        ferl_leaves=leaves,
    )


def run_e10(pair, *, subset, detector_seed, concept_names, seed, ks,
            learned_depth, output_dir, calibrate_concepts=False):
    """Compare FERL support queries with a validation-matched adaptive LR."""
    (otr, ova, ote), (ptr, pva, pte) = pair
    n, n_concepts = pte.C.shape
    concept_variant = "calibrated" if calibrate_concepts else "raw"
    if calibrate_concepts:
        calibrators = _fit_concept_calibrators(ova.C, pva.C)
        model_train, model_val, model_test = (
            _apply_calibrators(calibrators, values)
            for values in (ptr.C, pva.C, pte.C)
        )
    else:
        model_train, model_val, model_test = ptr.C, pva.C, pte.C

    value_absent, value_present = intervention_values(otr.C, model_train)
    verified = np.where(ote.C >= 0.5, value_present[None, :], value_absent[None, :])
    budgets = sorted(set(int(k) for k in ks if int(k) >= 0)) or [0]
    max_budget = min(max(budgets), n_concepts)

    print(
        f"[cbm:e10] fit subset={subset} det={detector_seed} "
        f"concepts={concept_variant} methods=ferl,lr",
        flush=True,
    )
    ferl = make_model_x("ferl-deep", seed, learned_depth).fit(model_train, ptr.y)
    lr = make_model_x("logistic_regression", seed, learned_depth).fit(model_train, ptr.y)
    tree_shape = _learned_tree_shape(ferl)
    print(
        f"[cbm:e10] FERL structure depth={tree_shape['ferl_actual_depth']} "
        f"nodes={tree_shape['ferl_nodes']} leaves={tree_shape['ferl_leaves']}",
        flush=True,
    )

    # FERL defines the target coverage without labels. LR's threshold uses only
    # validation confidences and is then frozen before any test intervention.
    ferl_val_sets, _ = predict_set_learned(ferl, model_val)
    ferl_val_accepted = ferl_val_sets.sum(1) == 1
    target_val_acceptance = float(ferl_val_accepted.mean())
    lr_val_prediction, lr_val_confidence, _ = confidence_acceptance(
        lr, model_val, float("-inf"),
    )
    lr_threshold = confidence_threshold_for_coverage(
        lr_val_confidence, target_val_acceptance,
    )
    lr_val_accepted = lr_val_confidence >= lr_threshold
    ferl_val_prediction = ferl.predict(model_val)

    ferl_initial_sets, ferl_initial_ignorance = predict_set_learned(
        ferl, model_test,
    )
    ferl_initial_rejected = ferl_initial_sets.sum(1) > 1
    ferl_initial_prediction = ferl.predict(model_test)
    lr_initial_prediction, _, lr_initial_accepted = confidence_acceptance(
        lr, model_test, lr_threshold,
    )
    lr_initial_rejected = ~lr_initial_accepted

    ferl_order = global_importance_order(
        ferl, "ferl-deep", concept_names, n_concepts,
    )
    reliability = concept_reliability(ova.C, model_val)
    route_reliability = fit_route_reliability(ferl, model_val, pva.y)
    verified_val = np.where(
        ova.C >= 0.5, value_present[None, :], value_absent[None, :],
    )
    feature_utility = fit_feature_intervention_utility(
        ferl, model_val, pva.y, verified_val,
    )
    ferl_trajectory = adaptive_query_trajectory(
        ferl,
        model_test,
        verified,
        value_absent,
        value_present,
        max_queries=max_budget,
        global_order=ferl_order,
    )
    ferl_reliability_trajectory = adaptive_reliability_query_trajectory(
        ferl,
        model_test,
        verified,
        value_absent,
        value_present,
        reliability,
        route_reliability=route_reliability,
        feature_utility=feature_utility,
        max_queries=max_budget,
        global_order=ferl_order,
    )
    lr_adaptive_trajectory = adaptive_lr_query_trajectory(
        lr,
        model_test,
        verified,
        value_absent,
        value_present,
        confidence_threshold=lr_threshold,
        max_queries=max_budget,
    )
    lr_order = global_importance_order(
        lr, "logistic_regression", concept_names, n_concepts,
    )
    lr_orders = np.broadcast_to(lr_order, (n, n_concepts))
    lr_fixed_trajectory = fixed_order_lr_query_trajectory(
        lr,
        model_test,
        verified,
        lr_orders,
        confidence_threshold=lr_threshold,
        max_queries=max_budget,
    )

    validation = {
        "ferl-deep": dict(
            threshold=np.nan,
            acceptance=float(ferl_val_accepted.mean()),
            accepted_accuracy=_accepted_accuracy(
                ferl_val_prediction, pva.y, ferl_val_accepted,
            ),
        ),
        "logistic_regression": dict(
            threshold=lr_threshold,
            acceptance=float(lr_val_accepted.mean()),
            accepted_accuracy=_accepted_accuracy(
                lr_val_prediction, pva.y, lr_val_accepted,
            ),
        ),
    }

    rows = []
    for budget in budgets:
        budget = min(budget, n_concepts)
        for policy, trajectory in (
            ("adaptive_support", ferl_trajectory),
            ("adaptive_reliability_purity", ferl_reliability_trajectory),
        ):
            ferl_snapshot = trajectory[budget]
            if budget == 0 or not ferl_initial_rejected.any():
                ferl_accepted = ~ferl_initial_rejected
                ferl_prediction = ferl_initial_prediction
                mean_ferl_ignorance = float(ferl_initial_ignorance.mean())
            else:
                changed = np.flatnonzero(ferl_initial_rejected)
                changed_sets, changed_ignorance = predict_set_learned(
                    ferl, ferl_snapshot.X[changed],
                )
                ferl_accepted = ~ferl_initial_rejected.copy()
                ferl_accepted[changed] = changed_sets.sum(1) == 1
                ferl_prediction = ferl_initial_prediction.copy()
                ferl_prediction[changed] = ferl.predict(ferl_snapshot.X[changed])
                unchanged_ignorance = ferl_initial_ignorance[~ferl_initial_rejected]
                mean_ferl_ignorance = float(
                    (unchanged_ignorance.sum() + changed_ignorance.sum()) / n
                )
            rows.append(dict(
                subset=subset,
                detector_seed=detector_seed,
                concepts=concept_variant,
                method="ferl-deep",
                policy=policy,
                k=budget,
                **tree_shape,
                confidence_threshold=np.nan,
                target_validation_acceptance_rate=target_val_acceptance,
                validation_acceptance_rate=validation["ferl-deep"]["acceptance"],
                validation_accepted_accuracy=validation["ferl-deep"]["accepted_accuracy"],
                initial_abstention_rate=float(ferl_initial_rejected.mean()),
                mean_concept_reliability=float(reliability.mean()),
                validation_route_accuracy=route_reliability.global_accuracy,
                validation_routes=len(route_reliability.leaf_accuracy),
                mean_validation_feature_net_correction=float(
                    feature_utility.net_correction.mean()
                ),
                expected_information_gain_queries=0,
                fixed_importance_queries=0,
                support_backtrack_queries=ferl_snapshot.reason_counts.get(
                    "support_backtrack", 0,
                ),
                expected_uncertainty_queries=ferl_snapshot.reason_counts.get(
                    "expected_uncertainty", 0,
                ),
                global_fallback_queries=ferl_snapshot.reason_counts.get(
                    "global_fallback", 0,
                ),
                reliability_purity_support_queries=ferl_snapshot.reason_counts.get(
                    "reliability_purity_support", 0,
                ),
                reliability_purity_uncertainty_queries=ferl_snapshot.reason_counts.get(
                    "reliability_purity_uncertainty", 0,
                ),
                reliability_purity_fallback_queries=ferl_snapshot.reason_counts.get(
                    "reliability_purity_fallback", 0,
                ),
                mean_ignorance=mean_ferl_ignorance,
                **_selective_intervention_metrics(
                    ferl_prediction,
                    pte.y,
                    ferl_accepted,
                    ferl_snapshot.query_counts,
                    ferl_initial_rejected,
                    ferl_initial_prediction,
                ),
            ))

        for policy, trajectory in (
            ("adaptive_eig", lr_adaptive_trajectory),
            ("fixed_importance", lr_fixed_trajectory),
        ):
            snapshot = trajectory[budget]
            prediction, _, accepted = confidence_acceptance(
                lr, snapshot.X, lr_threshold,
            )
            rows.append(dict(
                subset=subset,
                detector_seed=detector_seed,
                concepts=concept_variant,
                method="logistic_regression",
                policy=policy,
                k=budget,
                **tree_shape,
                confidence_threshold=validation["logistic_regression"]["threshold"],
                target_validation_acceptance_rate=target_val_acceptance,
                validation_acceptance_rate=validation["logistic_regression"]["acceptance"],
                validation_accepted_accuracy=validation["logistic_regression"]["accepted_accuracy"],
                initial_abstention_rate=float(lr_initial_rejected.mean()),
                mean_concept_reliability=float(reliability.mean()),
                validation_route_accuracy=np.nan,
                validation_routes=np.nan,
                mean_validation_feature_net_correction=np.nan,
                expected_information_gain_queries=snapshot.reason_counts.get(
                    "expected_information_gain", 0,
                ),
                fixed_importance_queries=(
                    int(snapshot.query_counts.sum())
                    if policy == "fixed_importance" else 0
                ),
                support_backtrack_queries=0,
                expected_uncertainty_queries=0,
                global_fallback_queries=0,
                reliability_purity_support_queries=0,
                reliability_purity_uncertainty_queries=0,
                reliability_purity_fallback_queries=0,
                mean_ignorance=np.nan,
                **_selective_intervention_metrics(
                    prediction,
                    pte.y,
                    accepted,
                    snapshot.query_counts,
                    lr_initial_rejected,
                    lr_initial_prediction,
                ),
            ))
        print(f"[cbm:e10] evaluated query budget={budget}", flush=True)

    df = pd.DataFrame(rows)
    output_csv = output_dir / "e10_matched_adaptive_lr.csv"
    if output_csv.exists():
        previous = pd.read_csv(output_csv)
        if "ferl_max_depth" not in previous:
            # All E10 rows predating structural metadata were the documented
            # CUB-20/default-depth run.
            previous["ferl_max_depth"] = 12
            previous["ferl_actual_depth"] = np.nan
            previous["ferl_nodes"] = np.nan
            previous["ferl_leaves"] = np.nan
        combined = pd.concat([previous, df], ignore_index=True)
        combined["subset"] = combined["subset"].astype(str)
        combined = combined.drop_duplicates(
            subset=[
                "subset", "detector_seed", "concepts", "method", "policy",
                "ferl_max_depth", "k",
            ],
            keep="last",
        )
        combined.to_csv(output_csv, index=False)
    else:
        df.to_csv(output_csv, index=False)
    plot_e10(
        df,
        output_dir / (
            f"e10_matched_adaptive_lr_{subset}_{concept_variant}_"
            f"depth{learned_depth}_seed{detector_seed}.pdf"
        ),
    )
    return df


def plot_e10(df: pd.DataFrame, path: Path) -> None:
    styles = {
        ("ferl-deep", "adaptive_support"): ("FERL adaptive support", "#D55E00", "-"),
        ("ferl-deep", "adaptive_reliability_purity"): (
            "FERL utility-aware", "#CC79A7", "-.",
        ),
        ("logistic_regression", "adaptive_eig"): ("LR adaptive EIG", "#0072B2", "-"),
        ("logistic_regression", "fixed_importance"): ("LR fixed importance", "#009E73", "--"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2))
    for key, group in df.groupby(["method", "policy"]):
        label, color, linestyle = styles[key]
        by_recovery = group.sort_values("mean_queries_initial_abstained")
        axes[0].plot(
            by_recovery["mean_queries_initial_abstained"],
            by_recovery["net_correction_rate"],
            marker="o", ms=3, lw=1.5, ls=linestyle, color=color, label=label,
        )
        axes[1].plot(
            by_recovery["mean_queries_initial_abstained"],
            by_recovery["wrong_singleton_rate"],
            marker="o", ms=3, lw=1.5, ls=linestyle, color=color, label=label,
        )
        by_accuracy = group.sort_values("mean_queries")
        axes[2].plot(
            by_accuracy["mean_queries"], by_accuracy["accuracy"],
            marker="o", ms=3, lw=1.5, ls=linestyle, color=color, label=label,
        )
    axes[0].set_xlabel("mean queries on initially rejected samples")
    axes[0].set_ylabel("net correction rate")
    axes[1].set_xlabel("mean queries on initially rejected samples")
    axes[1].set_ylabel("wrong accepted-decision rate")
    axes[2].set_xlabel("mean queries per test sample")
    axes[2].set_ylabel("accuracy")
    for axis in axes:
        axis.grid(True, ls=":", alpha=0.4)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_e10_aggregate(df: pd.DataFrame, path: Path) -> None:
    """Plot the across-detector-seed E10 mean with one-standard-deviation bars."""
    styles = {
        ("ferl-deep", "adaptive_support"): ("FERL adaptive support", "#D55E00", "-"),
        ("ferl-deep", "adaptive_reliability_purity"): (
            "FERL utility-aware", "#CC79A7", "-.",
        ),
        ("logistic_regression", "adaptive_eig"): ("LR adaptive EIG", "#0072B2", "-"),
        ("logistic_regression", "fixed_importance"): ("LR fixed importance", "#009E73", "--"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2))
    for key, group in df.groupby(["method", "policy"]):
        label, color, linestyle = styles[key]
        summary = group.groupby("k").agg(
            queries=("mean_queries_initial_abstained", "mean"),
            net=("net_correction_rate", "mean"),
            net_sd=("net_correction_rate", "std"),
            wrong=("wrong_singleton_rate", "mean"),
            wrong_sd=("wrong_singleton_rate", "std"),
            queries_all=("mean_queries", "mean"),
            accuracy=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
        ).reset_index()
        axes[0].errorbar(
            summary["queries"], summary["net"], yerr=summary["net_sd"],
            marker="o", ms=3, lw=1.5, ls=linestyle, capsize=2,
            color=color, label=label,
        )
        axes[1].errorbar(
            summary["queries"], summary["wrong"], yerr=summary["wrong_sd"],
            marker="o", ms=3, lw=1.5, ls=linestyle, capsize=2,
            color=color, label=label,
        )
        axes[2].errorbar(
            summary["queries_all"], summary["accuracy"],
            yerr=summary["accuracy_sd"], marker="o", ms=3, lw=1.5,
            ls=linestyle, capsize=2, color=color, label=label,
        )
    axes[0].set_xlabel("mean queries on initially rejected samples")
    axes[0].set_ylabel("net correction rate")
    axes[1].set_xlabel("mean queries on initially rejected samples")
    axes[1].set_ylabel("wrong accepted-decision rate")
    axes[2].set_xlabel("mean queries per test sample")
    axes[2].set_ylabel("accuracy")
    for axis in axes:
        axis.grid(True, ls=":", alpha=0.4)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# --- E11: shared concept-budget frontier ------------------------------------

def _upsert_csv(path: Path, rows: pd.DataFrame, key: list[str]) -> None:
    """Append experiment rows while replacing an already-run configuration."""
    if path.exists():
        previous = pd.read_csv(path)
        combined = pd.concat([previous, rows], ignore_index=True)
        if "subset" in combined:
            combined["subset"] = combined["subset"].astype(str)
        combined = combined.drop_duplicates(subset=key, keep="last")
    else:
        combined = rows.copy()
        if "subset" in combined:
            combined["subset"] = combined["subset"].astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)


def _head_used_features(model, method: str) -> set[int]:
    if method == "ferl-deep":
        return _tree_features(model)
    if method == "decision_tree":
        return set(int(f) for f in model.tree_.feature if f >= 0)
    coefficient = np.asarray(model.coef_, dtype=float)
    return set(np.flatnonzero(np.abs(coefficient).max(0) > 1e-8).tolist())


def run_e11(pair, *, subset, detector_seed, concept_names, seed,
            concept_budgets, learned_depth, output_dir,
            calibrate_concepts=False):
    """Refit all heads on identical train-ranked concept subsets.

    Concept ranking uses only ``(X_train, y_train)``. Validation labels are not
    used to select a vocabulary size or reorder features; when calibration is
    requested, validation concept annotations are used solely for the same
    pre-specified isotonic score transform as E7/E10.
    """
    (otr, ova, _), (ptr, pva, pte) = pair
    concept_variant = "calibrated" if calibrate_concepts else "raw"
    if calibrate_concepts:
        calibrators = _fit_concept_calibrators(ova.C, pva.C)
        model_train, model_val, model_test = (
            _apply_calibrators(calibrators, values)
            for values in (ptr.C, pva.C, pte.C)
        )
    else:
        model_train, model_val, model_test = ptr.C, pva.C, pte.C

    n_concepts = model_train.shape[1]
    budgets = sorted(set(
        min(n_concepts, max(1, int(budget))) for budget in concept_budgets
    ))
    information = mutual_info_classif(
        model_train,
        ptr.y,
        discrete_features=False,
        random_state=seed,
    )
    ranking = np.argsort(-information, kind="stable")
    methods = ("ferl-deep", "decision_tree", "logistic_regression")
    rows: list[dict] = []
    for budget in budgets:
        # Preserve the artifact's original column order inside the selected
        # set. Greedy trees break exact gain ties by feature position; feeding
        # the MI ranking as a column permutation would otherwise make the
        # full-budget endpoint differ for a reason unrelated to the budget.
        selected = np.sort(ranking[:budget])
        X_train = model_train[:, selected]
        X_val = model_val[:, selected]
        X_test = model_test[:, selected]
        for method in methods:
            print(
                f"[cbm:e11] fit subset={subset} det={detector_seed} "
                f"concepts={concept_variant} budget={budget} method={method}",
                flush=True,
            )
            model = make_model_x(method, seed, learned_depth).fit(X_train, ptr.y)
            prediction = model.predict(X_test)
            validation_prediction = model.predict(X_val)
            used_local = _head_used_features(model, method)
            used_original = sorted(int(selected[index]) for index in used_local)
            row = dict(
                subset=subset,
                detector_seed=detector_seed,
                concepts=concept_variant,
                method=method,
                ferl_max_depth=learned_depth,
                concept_budget=budget,
                input_concepts=budget,
                used_concepts=len(used_original),
                selected_concept_indices=json.dumps(selected.tolist()),
                used_concept_indices=json.dumps(used_original),
                selected_concept_names=json.dumps(
                    [concept_names[index] for index in selected],
                ),
                used_concept_names=json.dumps(
                    [concept_names[index] for index in used_original],
                ),
                mean_selected_mutual_information=float(information[selected].mean()),
                validation_accuracy=accuracy_score(pva.y, validation_prediction),
                accuracy=accuracy_score(pte.y, prediction),
                macro_f1=f1_score(
                    pte.y, prediction, average="macro", zero_division=0,
                ),
                actual_depth=np.nan,
                nodes=np.nan,
                leaves=np.nan,
                resolution_rate=np.nan,
                accepted_accuracy=np.nan,
                mean_ignorance=np.nan,
            )
            if method == "ferl-deep":
                shape = _learned_tree_shape(model)
                row.update(
                    actual_depth=shape["ferl_actual_depth"],
                    nodes=shape["ferl_nodes"],
                    leaves=shape["ferl_leaves"],
                )
                sets, ignorance = predict_set_learned(model, X_test)
                accepted = sets.sum(1) == 1
                row.update(
                    resolution_rate=float(accepted.mean()),
                    accepted_accuracy=_accepted_accuracy(
                        prediction, pte.y, accepted,
                    ),
                    mean_ignorance=float(ignorance.mean()),
                )
            elif method == "decision_tree":
                row.update(
                    actual_depth=model.get_depth(),
                    nodes=model.tree_.node_count,
                    leaves=model.get_n_leaves(),
                )
            rows.append(row)

    df = pd.DataFrame(rows)
    _upsert_csv(
        output_dir / "e11_concept_budget_frontier.csv",
        df,
        key=[
            "subset", "detector_seed", "concepts", "method",
            "ferl_max_depth", "concept_budget",
        ],
    )
    plot_e11(
        df,
        output_dir / (
            f"e11_concept_budget_frontier_{subset}_{concept_variant}_"
            f"depth{learned_depth}_seed{detector_seed}.pdf"
        ),
    )
    return df


def plot_e11(df: pd.DataFrame, path: Path) -> None:
    styles = {
        "ferl-deep": ("FERL-deep", "#D55E00", "o"),
        "decision_tree": ("CART", "#009E73", "s"),
        "logistic_regression": ("Logistic regression", "#0072B2", "^"),
    }
    fig, axis = plt.subplots(figsize=(4.4, 3.2))
    grouped = df.groupby(["method", "concept_budget"])["accuracy"].agg(
        ["mean", "std"],
    ).reset_index()
    for method, group in grouped.groupby("method"):
        label, color, marker = styles[method]
        group = group.sort_values("concept_budget")
        error = group["std"].fillna(0.0)
        axis.errorbar(
            group["concept_budget"], 100.0 * group["mean"],
            yerr=100.0 * error, marker=marker, ms=4, lw=1.5,
            capsize=2, color=color, label=label,
        )
    axis.set_xlabel("available concepts (train-ranked)")
    axis.set_ylabel("test accuracy (%)")
    axis.grid(True, ls=":", alpha=0.4)
    axis.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# --- E6: reliability-discounted evidential fusion -----------------------------

def run_e6(pair, *, artifact_dir, subset, detector_seed, detector_seeds, seed,
           corrupt_frac, learned_depth, output_dir):
    (otr, ova, ote), (ptr, pva, pte) = pair
    classes = np.arange(len(np.unique(np.concatenate([ptr.y, pte.y]))))
    print(f"[cbm:e6] fit subset={subset} det={detector_seed}", flush=True)
    model = LearnedFuzzyTree(max_depth=learned_depth, random_state=seed).fit(ptr.C, ptr.y)
    rng = np.random.default_rng(seed)

    def record(condition, level, X_test, y_test, rho):
        variants = [("plain", None, "product")] + [
            (f"disc_{agg}", rho, agg) for agg in ("product", "mean", "min")
        ]
        pred = model.predict(X_test)
        for variant, r, agg in variants:
            sets, ign = predict_set_learned(model, X_test, r, rho_agg=agg)
            rows.append(dict(subset=subset, detector_seed=detector_seed,
                             condition=condition, level=level, variant=variant,
                             accuracy=accuracy_score(y_test, pred),
                             mean_ignorance=float(ign.mean()),
                             **set_metrics(sets, y_test, classes)))

    rows: list[dict] = []
    record("clean", 0.0, pte.C, pte.y, concept_reliability(ova.C, pva.C))

    # targeted corruption: destroy a fraction of concept heads (column shuffle),
    # same corruption applied to val so reliability sees the deployment state
    bad = rng.choice(pte.C.shape[1], max(1, int(round(corrupt_frac * pte.C.shape[1]))), replace=False)
    Xte, Xva = pte.C.copy(), pva.C.copy()
    for j in bad:
        Xte[:, j] = rng.permutation(Xte[:, j])
        Xva[:, j] = rng.permutation(Xva[:, j])
    record("corrupt_concepts", corrupt_frac, Xte, pte.y, concept_reliability(ova.C, Xva))

    for test_seed in detector_seeds:
        if test_seed == detector_seed:
            continue
        _, (str_, sva, ste) = load_pair(artifact_dir, subset, test_seed)
        record("detector_swap", float(test_seed), ste.C, ste.y,
               concept_reliability(ova.C, sva.C))

    df = pd.DataFrame(rows)
    _append_csv(output_dir / "e6_reliability.csv", df)
    return df


# --- E8: risk-controlled selective prediction ----------------------------------

def _cp_upper(k: int, n: int, delta: float) -> float:
    """Clopper-Pearson upper bound on a binomial proportion."""
    from scipy.stats import beta as beta_dist
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(1.0 - delta, k + 1, n - k))


def sgr_threshold(conf_val: np.ndarray, err_val: np.ndarray, beta: float,
                  delta: float = 0.05, grid: int = 64):
    """Risk-controlled acceptance threshold (SGR/Learn-then-Test flavour):
    Bonferroni-corrected Clopper-Pearson bound over a grid of coverage levels;
    returns the tau (accept if conf >= tau) with the largest certified
    coverage whose accepted-set error rate is <= beta w.p. 1-delta, or None."""
    n = len(conf_val)
    order = np.argsort(-conf_val, kind="stable")
    cum_err = np.cumsum(err_val[order].astype(int))
    sizes = np.unique(np.clip(np.round(np.linspace(1, n, grid)).astype(int), 1, n))
    d = delta / len(sizes)
    tau, best_m = None, 0
    for m in sizes:
        if _cp_upper(int(cum_err[m - 1]), int(m), d) <= beta and m > best_m:
            best_m, tau = m, float(conf_val[order[m - 1]])
    return tau


def _score_suite(model, method: str, X: np.ndarray, rho=None, ferl_extras=None):
    """(point predictions, {score_name: confidence}) — higher = more confident."""
    P = model.predict_proba(X)
    pred = model.classes_[P.argmax(1)]
    top2 = -np.partition(-P, 1, axis=1)[:, :2]
    scores = {"max_proba": P.max(1), "margin": top2[:, 0] - top2[:, 1]}
    if method == "ferl-deep":
        sets, ign = predict_set_learned(model, X)
        scores["ignorance"] = -ign
        scores["neg_set_size"] = -sets.sum(1).astype(float)
        if rho is not None:
            _, ign_d = predict_set_learned(model, X, rho, rho_agg="mean")
            scores["ignorance_disc"] = -ign_d
        if ferl_extras is not None:
            T, meta = ferl_extras
            PT = P ** (1.0 / T)
            PT /= PT.sum(1, keepdims=True)
            scores["max_proba_T"] = PT.max(1)
            if meta is not None:
                col = list(meta.classes_).index(0)
                scores["meta"] = meta.predict_proba(_proba_feats(P))[:, col]
    return pred, scores


def _proba_feats(P: np.ndarray) -> np.ndarray:
    top2 = -np.partition(-P, 1, axis=1)[:, :2]
    ent = -(P * np.log(np.clip(P, 1e-12, None))).sum(1)
    return np.column_stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent])


def _tree_features(model: LearnedFuzzyTree) -> set[int]:
    feats: set[int] = set()

    def walk(n):
        if n["leaf"]:
            return
        feats.add(int(n["f"]))
        walk(n["L"])
        walk(n["R"])

    walk(model.root_)
    return feats


def _fit_ferl_confidence(Xtr, ytr, seed, learned_depth, n_folds=3):
    """Cross-fitted (train-only, val never touched -> certificates stay valid)
    confidence extras for ferl-deep: a temperature minimising out-of-fold
    NLL, and a logistic failure-predictor over probability-shape features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    if np.bincount(ytr).min() < n_folds:
        return None
    folds = []
    feats, errs = [], []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(Xtr, ytr):
        m = LearnedFuzzyTree(max_depth=learned_depth, random_state=seed).fit(Xtr[tr], ytr[tr])
        P = m.predict_proba(Xtr[te])
        cls = {c: i for i, c in enumerate(m.classes_)}
        true_idx = np.array([cls.get(y, -1) for y in ytr[te]])
        folds.append((P, true_idx))
        feats.append(_proba_feats(P))
        errs.append((m.classes_[P.argmax(1)] != ytr[te]).astype(int))
    # temperature: minimise OOF NLL over a grid
    best_T, best_nll = 1.0, np.inf
    for T in np.geomspace(0.25, 8.0, 25):
        nll, n = 0.0, 0
        for P, ti in folds:
            PT = P ** (1.0 / T)
            PT /= PT.sum(1, keepdims=True)
            ok = ti >= 0
            nll += -np.log(np.clip(PT[np.flatnonzero(ok), ti[ok]], 1e-12, None)).sum()
            n += int(ok.sum())
        if n and nll / n < best_nll:
            best_nll, best_T = nll / n, float(T)
    err_all = np.concatenate(errs)
    # meta needs both classes; on tiny/near-perfect OOF folds it can be all-correct
    meta = None
    if len(np.unique(err_all)) == 2:
        meta = LogisticRegression(max_iter=1000).fit(np.vstack(feats), err_all)
    return best_T, meta


def _fit_sparse_lr(Xtr, ytr, budget: int, seed: int):
    """L1 logistic head whose number of used concepts is matched to the FERL
    tree's. Bisection on C for the densest model with 0 < used <= budget; the
    L1 path can jump past small budgets, in which case the sparsest
    non-trivial model is returned (with its actual concept count logged)."""
    from sklearn.linear_model import LogisticRegression

    def fit(C):
        m = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=3000,
                               random_state=seed).fit(Xtr, ytr)
        return m, int((np.abs(m.coef_).max(0) > 1e-8).sum())

    lo, hi = 0.02, 3.0
    best = None                      # densest with 0 < used <= budget
    fallback = (*fit(hi), hi)        # sparsest seen with used > 0
    for _ in range(5):
        mid = float(np.sqrt(lo * hi))
        m, used = fit(mid)
        if 0 < used <= budget:
            if best is None or used > best[1]:
                best = (m, used, mid)
            lo = mid
        elif used == 0:
            lo = mid
        else:
            if used < fallback[1]:
                fallback = (m, used, mid)
            hi = mid
    m, used, C = best if best is not None else fallback
    print(f"[cbm:e8] sparse LR: C={C:.4f} concepts_used={used} (budget {budget})", flush=True)
    return m


def _fit_concept_calibrators(oracle_val_C: np.ndarray, pred_val_C: np.ndarray):
    cals = []
    for j in range(pred_val_C.shape[1]):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(pred_val_C[:, j], (oracle_val_C[:, j] >= 0.5).astype(float))
        cals.append(iso)
    return cals


def _apply_calibrators(cals, X: np.ndarray) -> np.ndarray:
    X = X.copy()
    for j, iso in enumerate(cals):
        X[:, j] = iso.predict(X[:, j])
    return X


def run_e8(pair, *, artifact_dir, subset, detector_seed, detector_seeds, seed,
           corrupt_frac, betas, learned_depth, output_dir, delta=0.05,
           calibrate_concepts=False):
    from sklearn.metrics import roc_auc_score

    (otr, ova, ote), (ptr, pva, pte) = pair
    prefix = "e8cal" if calibrate_concepts else "e8"
    Xtr, Xva0, Xte0 = ptr.C, pva.C, pte.C
    if calibrate_concepts:
        cals = _fit_concept_calibrators(ova.C, pva.C)
        Xtr, Xva0, Xte0 = (_apply_calibrators(cals, A) for A in (Xtr, Xva0, Xte0))
    rng = np.random.default_rng(seed)
    models = {}
    for method in ("ferl-deep", "decision_tree", "logistic_regression"):
        print(f"[cbm:{prefix}] fit subset={subset} det={detector_seed} method={method}", flush=True)
        models[method] = make_model_x(method, seed, learned_depth)
        models[method].fit(Xtr, ptr.y)
    budget = len(_tree_features(models["ferl-deep"]))
    models["logistic_l1"] = _fit_sparse_lr(Xtr, ptr.y, budget, seed)
    print(f"[cbm:{prefix}] cross-fitting ferl confidence extras", flush=True)
    ferl_extras = _fit_ferl_confidence(Xtr, ptr.y, seed, learned_depth)

    # conditions: (name, level, X_val, X_test, y_test, rho_val_source)
    conditions = [("clean", 0.0, Xva0, Xte0, pte.y, Xva0)]
    bad = rng.choice(Xte0.shape[1], max(1, int(round(corrupt_frac * Xte0.shape[1]))), replace=False)
    Xte_c, Xva_c = Xte0.copy(), Xva0.copy()
    for j in bad:
        Xte_c[:, j] = rng.permutation(Xte_c[:, j])
        Xva_c[:, j] = rng.permutation(Xva_c[:, j])
    conditions.append(("corrupt_concepts", corrupt_frac, Xva_c, Xte_c, pte.y, Xva_c))
    for test_seed in detector_seeds:
        if test_seed == detector_seed:
            continue
        _, (_str, sva, ste) = load_pair(artifact_dir, subset, test_seed)
        sva_C, ste_C = sva.C, ste.C
        if calibrate_concepts:
            cals_s = _fit_concept_calibrators(ova.C, sva.C)
            sva_C, ste_C = _apply_calibrators(cals_s, sva_C), _apply_calibrators(cals_s, ste_C)
        conditions.append(("detector_swap", float(test_seed), sva_C, ste_C, ste.y, sva_C))

    rows = []
    frontier_rows: list[dict] = []
    for cond, level, Xva, Xte, yte, rho_src in conditions:
        rho = concept_reliability(ova.C, rho_src)
        for method, model in models.items():
            extras = ferl_extras if method == "ferl-deep" else None
            pred_va, sc_va = _score_suite(model, method, Xva, rho, extras)
            pred_te, sc_te = _score_suite(model, method, Xte, rho, extras)
            err_va = (pred_va != pva.y).astype(int)
            err_te = (pred_te != yte).astype(int)
            # stale calibration: clean val scores (exchangeability broken off-clean)
            pred_va0, sc_va0 = _score_suite(model, method, Xva0,
                                            concept_reliability(ova.C, Xva0), extras)
            err_va0 = (pred_va0 != pva.y).astype(int)
            for score_name, conf_te in sc_te.items():
                auroc = np.nan
                if 0 < err_te.mean() < 1:
                    auroc = float(roc_auc_score(err_te, -conf_te))
                order = np.argsort(-np.asarray(conf_te, float), kind="stable")
                cum_risk = np.cumsum(err_te[order]) / np.arange(1, len(order) + 1)
                aurc = float(cum_risk.mean())
                for beta in betas:
                    # best-in-hindsight threshold on THIS test set (no statistics),
                    # and the perfect-score ceiling given only the base error:
                    # accept all corrects plus errors up to a beta fraction.
                    ok = np.flatnonzero(cum_risk <= beta)
                    cov_oracle = float((ok[-1] + 1) / len(cum_risk)) if len(ok) else 0.0
                    cov_ceiling = float(min(1.0, (1.0 - err_te.mean()) / (1.0 - beta)))
                    for calib, (cv, ev) in (("iid", (sc_va[score_name], err_va)),
                                            ("stale", (sc_va0[score_name], err_va0))):
                        tau = sgr_threshold(np.asarray(cv, float), ev, beta, delta)
                        if tau is None:
                            cov, risk = 0.0, np.nan
                        else:
                            acc = conf_te >= tau
                            cov = float(acc.mean())
                            risk = float(err_te[acc].mean()) if acc.any() else np.nan
                        rows.append(dict(
                            subset=subset, detector_seed=detector_seed, condition=cond,
                            level=level, method=method, score=score_name, calib=calib,
                            beta=beta, auroc_failure=auroc, aurc=float(aurc),
                            tau=tau, coverage=cov, selective_risk=risk,
                            coverage_oracle=cov_oracle, coverage_ceiling=cov_ceiling,
                            bound_holds=bool((risk != risk) or (risk <= beta)),
                        ))
                # dual formulation: fix a coverage budget, certify the risk
                # (iid calibration only)
                cv = np.asarray(sc_va[score_name], float)
                order_v = np.argsort(-cv, kind="stable")
                cum_v = np.cumsum(err_va[order_v])
                cov_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                d_g = delta / len(cov_grid)
                for c in cov_grid:
                    m = max(1, int(round(c * len(cv))))
                    tau_c = float(cv[order_v[m - 1]])
                    acc = conf_te >= tau_c
                    frontier_rows.append(dict(
                        subset=subset, detector_seed=detector_seed, condition=cond,
                        level=level, method=method, score=score_name,
                        target_coverage=c,
                        certified_risk=_cp_upper(int(cum_v[m - 1]), m, d_g),
                        coverage=float(acc.mean()),
                        realized_risk=float(err_te[acc].mean()) if acc.any() else np.nan,
                    ))
        print(f"[cbm:{prefix}] done condition={cond} level={level}", flush=True)
    df = pd.DataFrame(rows)
    _append_csv(output_dir / f"{prefix}_selective_guarantee.csv", df)
    _append_csv(output_dir / f"{prefix}_frontier.csv", pd.DataFrame(frontier_rows))
    return df


# --- E7: per-concept score calibration ----------------------------------------

def run_e7(pair, *, subset, detector_seed, seed, learned_depth, output_dir):
    (otr, ova, ote), (ptr, pva, pte) = pair
    classes = np.arange(len(np.unique(np.concatenate([ptr.y, pte.y]))))
    Xtr_c, Xte_c = ptr.C.copy(), pte.C.copy()
    for j in range(ptr.C.shape[1]):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(pva.C[:, j], (ova.C[:, j] >= 0.5).astype(float))
        Xtr_c[:, j] = iso.predict(ptr.C[:, j])
        Xte_c[:, j] = iso.predict(pte.C[:, j])

    rows = []
    for variant, (Xtr, Xte) in (("raw", (ptr.C, pte.C)), ("calibrated", (Xtr_c, Xte_c))):
        ferl_model = None
        for method in E5_METHODS:
            print(f"[cbm:e7] fit subset={subset} det={detector_seed} {variant} {method}", flush=True)
            model = make_model_x(method, seed, learned_depth)
            model.fit(Xtr, ptr.y)
            pred = model.predict(Xte)
            row = dict(subset=subset, detector_seed=detector_seed, variant=variant,
                       method=method, accuracy=accuracy_score(pte.y, pred),
                       macro_f1=f1_score(pte.y, pred, average="macro", zero_division=0))
            if method == "ferl-deep":
                ferl_model = model
                sets, ign = predict_set_learned(model, Xte)
                row.update(set_metrics(sets, pte.y, classes))
                row["mean_ignorance"] = float(ign.mean())
            rows.append(row)
        # sparse LR at the FERL tree's concept budget (same head as E8)
        budget = len(_tree_features(ferl_model))
        slr = _fit_sparse_lr(Xtr, ptr.y, budget, seed)
        pred = slr.predict(Xte)
        rows.append(dict(subset=subset, detector_seed=detector_seed, variant=variant,
                         method="logistic_l1", accuracy=accuracy_score(pte.y, pred),
                         macro_f1=f1_score(pte.y, pred, average="macro", zero_division=0)))
    df = pd.DataFrame(rows)
    _append_csv(output_dir / "e7_calibration.csv", df)
    return df


# --- CLI -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", type=Path)
    p.add_argument("--subset", help="Subset label; defaults to artifact dir name")
    p.add_argument("--detector-seeds", nargs="+", type=int, default=[0])
    p.add_argument(
        "--experiments", nargs="+", default=["all"],
        choices=["e5", "e6", "e7", "e8", "e9", "e10", "e11", "all"],
    )
    p.add_argument("--betas", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    p.add_argument("--calibrate-concepts", action="store_true",
                   help="e8/e9/e10: apply per-concept isotonic calibration (fit on val)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ks", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16, 32, 64, 112])
    p.add_argument(
        "--concept-budgets", nargs="+", type=int,
        default=[8, 16, 32, 64, 112],
        help="e11: shared top-m concept vocabularies ranked on training data",
    )
    p.add_argument("--corrupt-frac", type=float, default=0.25)
    p.add_argument("--learned-depth", type=int, default=12)
    p.add_argument("--output-dir", type=Path, default=Path("results/cub_cbm_perf"))
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.artifact_dir = create_smoke_artifact(Path("/tmp/ferl_cub_cbm_smoke"))
        args.detector_seeds = [0, 1]
        args.output_dir = Path("/tmp/ferl_cub_cbm_smoke_results")
    if args.artifact_dir is None:
        raise SystemExit("--artifact-dir is required unless --smoke is used")
    subset = args.subset or args.artifact_dir.name
    experiments = {"e5", "e6", "e7", "e8", "e9", "e10", "e11"} if "all" in args.experiments else set(args.experiments)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    concept_names = json.loads((args.artifact_dir / "concept_names.json").read_text())

    for det in args.detector_seeds:
        pair = load_pair(args.artifact_dir, subset, det)
        if "e5" in experiments:
            run_e5(pair, subset=subset, detector_seed=det, concept_names=concept_names,
                   seed=args.seed, ks=args.ks, learned_depth=args.learned_depth,
                   output_dir=args.output_dir)
        if "e6" in experiments:
            run_e6(pair, artifact_dir=args.artifact_dir, subset=subset, detector_seed=det,
                   detector_seeds=args.detector_seeds, seed=args.seed,
                   corrupt_frac=args.corrupt_frac, learned_depth=args.learned_depth,
                   output_dir=args.output_dir)
        if "e7" in experiments:
            run_e7(pair, subset=subset, detector_seed=det, seed=args.seed,
                   learned_depth=args.learned_depth, output_dir=args.output_dir)
        if "e8" in experiments:
            run_e8(pair, artifact_dir=args.artifact_dir, subset=subset, detector_seed=det,
                   detector_seeds=args.detector_seeds, seed=args.seed,
                   corrupt_frac=args.corrupt_frac, betas=args.betas,
                   learned_depth=args.learned_depth, output_dir=args.output_dir,
                   calibrate_concepts=args.calibrate_concepts)
        if "e9" in experiments:
            run_e9(pair, subset=subset, detector_seed=det, concept_names=concept_names,
                   seed=args.seed, ks=args.ks, learned_depth=args.learned_depth,
                   output_dir=args.output_dir,
                   calibrate_concepts=args.calibrate_concepts)
        if "e10" in experiments:
            run_e10(pair, subset=subset, detector_seed=det, concept_names=concept_names,
                    seed=args.seed, ks=args.ks, learned_depth=args.learned_depth,
                    output_dir=args.output_dir,
                    calibrate_concepts=args.calibrate_concepts)
        if "e11" in experiments:
            run_e11(
                pair,
                subset=subset,
                detector_seed=det,
                concept_names=concept_names,
                seed=args.seed,
                concept_budgets=args.concept_budgets,
                learned_depth=args.learned_depth,
                output_dir=args.output_dir,
                calibrate_concepts=args.calibrate_concepts,
            )
    if "e10" in experiments:
        e10 = pd.read_csv(args.output_dir / "e10_matched_adaptive_lr.csv")
        concept_variant = "calibrated" if args.calibrate_concepts else "raw"
        selected = e10[
            e10["subset"].astype(str).eq(str(subset))
            & e10["concepts"].eq(concept_variant)
            & e10["ferl_max_depth"].eq(args.learned_depth)
            & e10["detector_seed"].isin(args.detector_seeds)
            & e10["k"].isin(args.ks)
        ]
        plot_e10_aggregate(
            selected,
            args.output_dir / (
                f"e10_matched_adaptive_lr_{subset}_{concept_variant}_"
                f"depth{args.learned_depth}_aggregate.pdf"
            ),
        )
    if "e11" in experiments:
        e11 = pd.read_csv(args.output_dir / "e11_concept_budget_frontier.csv")
        concept_variant = "calibrated" if args.calibrate_concepts else "raw"
        selected = e11[
            e11["subset"].astype(str).eq(str(subset))
            & e11["concepts"].eq(concept_variant)
            & e11["ferl_max_depth"].eq(args.learned_depth)
            & e11["detector_seed"].isin(args.detector_seeds)
            & e11["concept_budget"].isin([
                min(len(concept_names), max(1, int(value)))
                for value in args.concept_budgets
            ])
        ]
        plot_e11(
            selected,
            args.output_dir / (
                f"e11_concept_budget_frontier_{subset}_{concept_variant}_"
                f"depth{args.learned_depth}_aggregate.pdf"
            ),
        )
    print(f"wrote CBM extras outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
