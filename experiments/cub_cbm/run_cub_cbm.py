"""Stage B runner for CUB concept-bottleneck FERL experiments.

Input artifacts:

    concept_names.json
    oracle_train.npz / oracle_val.npz / oracle_test.npz
    predicted_seed<k>_train.npz / predicted_seed<k>_val.npz / predicted_seed<k>_test.npz

Each npz contains C (concept matrix), y (labels), and optionally ids.
Run from the repository root, for example:

    python experiments/cub_cbm/run_cub_cbm.py \
        --artifact-dir /path/to/cub_koh112_5 \
        --concept-sources oracle predicted \
        --detector-seeds 0 1 2 3 4 \
        --experiments e1 e2 e3 e4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from ferl.pipeline import CONFIGS, make
from ferl.uncertainty.conformal import (
    ConformalFERL,
    abstention_rate,
    avg_set_size,
    coverage,
    selective_risk_coverage,
)


FERL_CONFIGS = ("ferl-medium", "ferl-deep")
BASELINE_METHODS = ("majority_lookup", "decision_tree", "logistic_regression")
ALL_METHODS = BASELINE_METHODS + FERL_CONFIGS


@dataclass(frozen=True)
class SplitData:
    C: np.ndarray
    y: np.ndarray
    ids: np.ndarray


@dataclass(frozen=True)
class ArtifactBundle:
    artifact_dir: Path
    subset: str
    source: str
    detector_seed: int | None
    concept_names: list[str]
    train: SplitData
    val: SplitData
    test: SplitData


class MajorityLookupHead:
    """Majority-label table over thresholded concept vectors."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityLookupHead":
        self.classes_ = np.unique(y)
        self._class_to_col = {label: idx for idx, label in enumerate(self.classes_)}
        table: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        global_counts: Counter[int] = Counter()
        for row, label in zip(_binary_keys(X), y):
            label = int(label)
            table[row][label] += 1
            global_counts[label] += 1
        if not global_counts:
            raise ValueError("Cannot fit majority lookup on an empty matrix")
        self._default_counts = global_counts
        self._table = dict(table)
        self._default_label = _majority_label(global_counts)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for key in _binary_keys(X):
            counts = self._table.get(key, self._default_counts)
            preds.append(_majority_label(counts))
        return np.asarray(preds, dtype=self.classes_.dtype)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros((len(X), len(self.classes_)), dtype=float)
        for row_idx, key in enumerate(_binary_keys(X)):
            counts = self._table.get(key, self._default_counts)
            total = sum(counts.values())
            for label, count in counts.items():
                out[row_idx, self._class_to_col[label]] = count / total
        return out

    def n_rules(self) -> int:
        return len(self._table)


def _binary_keys(X: np.ndarray) -> Iterable[tuple[int, ...]]:
    return (tuple((row >= 0.5).astype(int).tolist()) for row in np.asarray(X))


def _majority_label(counts: Counter[int]) -> int:
    max_count = max(counts.values())
    return min(label for label, count in counts.items() if count == max_count)


def read_bundle(
    artifact_dir: Path,
    *,
    source: str,
    detector_seed: int | None = None,
    subset: str | None = None,
) -> ArtifactBundle:
    concept_names = json.loads((artifact_dir / "concept_names.json").read_text(encoding="utf-8"))
    prefix = "oracle" if source == "oracle" else f"predicted_seed{detector_seed}"
    if source == "predicted" and detector_seed is None:
        raise ValueError("predicted source requires --detector-seeds")
    splits = {
        split: _read_split(artifact_dir / f"{prefix}_{split}.npz", len(concept_names))
        for split in ("train", "val", "test")
    }
    return ArtifactBundle(
        artifact_dir=artifact_dir,
        subset=subset or artifact_dir.name.replace("cub_koh112_", ""),
        source=source,
        detector_seed=detector_seed,
        concept_names=list(concept_names),
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
    )


def _read_split(path: Path, n_concepts: int) -> SplitData:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    C = np.asarray(data["C"], dtype=float)
    y = np.asarray(data["y"])
    ids = np.asarray(data["ids"]) if "ids" in data.files else np.arange(len(y))
    if C.ndim != 2 or C.shape[1] != n_concepts:
        raise ValueError(f"{path}: expected C shape (n, {n_concepts}), got {C.shape}")
    if C.shape[0] != len(y) or len(ids) != len(y):
        raise ValueError(f"{path}: C, y, and ids lengths differ")
    if not np.all(np.isfinite(C)):
        raise ValueError(f"{path}: C contains non-finite values")
    if C.min(initial=0.0) < -1e-8 or C.max(initial=0.0) > 1.0 + 1e-8:
        raise ValueError(f"{path}: C values must be in [0, 1]")
    return SplitData(C=np.clip(C, 0.0, 1.0), y=y, ids=ids)


def encoded_splits(bundle: ArtifactBundle) -> tuple[LabelEncoder, SplitData, SplitData, SplitData]:
    encoder = LabelEncoder()
    encoder.fit(np.concatenate([bundle.train.y, bundle.val.y, bundle.test.y]))

    def enc(split: SplitData) -> SplitData:
        return SplitData(C=split.C, y=encoder.transform(split.y), ids=split.ids)

    return encoder, enc(bundle.train), enc(bundle.val), enc(bundle.test)


# set from --learned-depth so E1/E2/E3 build the same deep learned tree the
# extras (E5-E8) use; without this ferl-deep defaulted to depth 12 here and
# depth 60 there, so E1 undersold it (0.815 vs 0.950 on AwA2 predicted).
LEARNED_DEPTH: int | None = None


def make_model(method: str, seed: int):
    if method == "ferl-deep" and LEARNED_DEPTH is not None:
        from ferl.core.learned_tree import LearnedFuzzyTree
        return LearnedFuzzyTree(max_depth=LEARNED_DEPTH, random_state=seed)
    if method == "majority_lookup":
        return MajorityLookupHead()
    if method == "decision_tree":
        return DecisionTreeClassifier(
            criterion="gini",
            random_state=seed,
            min_samples_split=5,
            min_samples_leaf=2,
        )
    if method == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if method in CONFIGS or method == "ferl-deep":
        return make(method, random_state=seed)
    raise KeyError(f"Unknown method: {method}")


def model_complexity(model, method: str) -> float:
    if hasattr(model, "n_rules"):
        return float(model.n_rules())
    if method == "decision_tree":
        return float(model.get_n_leaves())
    coef = getattr(model, "coef_", None)
    if coef is not None:
        return float(np.count_nonzero(np.abs(coef) > 1e-12))
    return float("nan")


def predict_proba_aligned(model, X: np.ndarray, classes: np.ndarray) -> np.ndarray:
    P = np.asarray(model.predict_proba(X), dtype=float)
    model_classes = np.asarray(getattr(model, "classes_", classes))
    if np.array_equal(model_classes, classes):
        return _normalize_proba(P)
    out = np.zeros((len(X), len(classes)), dtype=float)
    col = {label: idx for idx, label in enumerate(classes)}
    for src_idx, label in enumerate(model_classes):
        if label in col:
            out[:, col[label]] = P[:, src_idx]
    return _normalize_proba(out)


def _normalize_proba(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-12, None)
    denom = P.sum(axis=1, keepdims=True)
    return P / np.where(denom <= 0, 1.0, denom)


def run_e1(
    bundle: ArtifactBundle,
    *,
    methods: list[str],
    seeds: list[int],
    output_dir: Path,
) -> pd.DataFrame:
    _, train, _, test = encoded_splits(bundle)
    classes = np.arange(len(np.unique(np.concatenate([train.y, test.y]))))
    rows = []
    for seed in seeds:
        for method in methods:
            print(f"[cub_cbm:e1] fit source={bundle.source} detector={bundle.detector_seed} seed={seed} method={method}", flush=True)
            start = time.perf_counter()
            model = make_model(method, seed)
            model.fit(train.C, train.y)
            train_s = time.perf_counter() - start
            P = predict_proba_aligned(model, test.C, classes)
            pred = P.argmax(axis=1)
            rows.append(
                dict(
                    experiment="e1",
                    subset=bundle.subset,
                    source=bundle.source,
                    detector_seed=bundle.detector_seed,
                    method=method,
                    seed=seed,
                    accuracy=accuracy_score(test.y, pred),
                    macro_f1=f1_score(test.y, pred, average="macro", zero_division=0),
                    complexity=model_complexity(model, method),
                    train_s=train_s,
                )
            )
            print(f"[cub_cbm:e1] done method={method} seed={seed} train_s={train_s:.2f}", flush=True)
    df = pd.DataFrame(rows)
    _append_csv(output_dir / "e1_accuracy.csv", df)
    summary = summarize_e1(output_dir / "e1_accuracy.csv")
    summary.to_csv(output_dir / "e1_accuracy_summary.csv", index=False)
    write_friedman_table(output_dir / "e1_accuracy.csv", output_dir / "e1_friedman.csv")
    return df


def summarize_e1(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keys = ["subset", "source", "detector_seed", "method"]
    grouped = df.groupby(keys, dropna=False)
    return grouped.agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        complexity_mean=("complexity", "mean"),
        train_s_mean=("train_s", "mean"),
        n=("accuracy", "size"),
    ).reset_index()


def write_friedman_table(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv)
    rows = []
    for (subset, source), group in df.groupby(["subset", "source"], dropna=False):
        pivot = group.pivot_table(
            index=["detector_seed", "seed"],
            columns="method",
            values="accuracy",
            aggfunc="mean",
        ).dropna(axis=0)
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            continue
        try:
            stat, p_value = friedmanchisquare(*[pivot[col].values for col in pivot.columns])
        except ValueError:
            continue
        mean_ranks = pivot.rank(axis=1, ascending=False).mean(axis=0)
        for method in pivot.columns:
            rows.append(
                dict(
                    subset=subset,
                    source=source,
                    method=method,
                    mean_rank=float(mean_ranks[method]),
                    friedman_stat=float(stat),
                    p_value=float(p_value),
                    n_blocks=int(pivot.shape[0]),
                )
            )
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def run_e2(
    bundle: ArtifactBundle,
    *,
    ferl_config: str,
    methods: list[str],
    seed: int,
    output_dir: Path,
    frontier_depths: list[int],
) -> None:
    _, train, _, test = encoded_splits(bundle)
    model = make_model(ferl_config, seed)
    print(f"[cub_cbm:e2] fit rule dump source={bundle.source} detector={bundle.detector_seed} method={ferl_config}", flush=True)
    model.fit(train.C, train.y)
    rule_text, importance = describe_ferl_rules(model, bundle.concept_names, max_rules=25)
    path = output_dir / f"e2_rules_{bundle.subset}_{bundle.source}.txt"
    if bundle.detector_seed is not None:
        path = output_dir / f"e2_rules_{bundle.subset}_{bundle.source}_seed{bundle.detector_seed}.txt"
    path.write_text(rule_text, encoding="utf-8")

    classes = np.arange(len(np.unique(np.concatenate([train.y, test.y]))))
    rows = []
    e2_methods = []
    for method in [ferl_config, *methods]:
        if (method in CONFIGS or method == "ferl-deep") and method not in e2_methods:
            e2_methods.append(method)
    for method in e2_methods:
        print(f"[cub_cbm:e2] frontier fit source={bundle.source} detector={bundle.detector_seed} method={method}", flush=True)
        m = make_model(method, seed)
        m.fit(train.C, train.y)
        P = predict_proba_aligned(m, test.C, classes)
        rows.append(
            dict(
                subset=bundle.subset,
                source=bundle.source,
                detector_seed=bundle.detector_seed,
                method=method,
                budget="config",
                accuracy=accuracy_score(test.y, P.argmax(1)),
                macro_f1=f1_score(test.y, P.argmax(1), average="macro", zero_division=0),
                complexity=model_complexity(m, method),
            )
        )
    for depth in frontier_depths:
        print(f"[cub_cbm:e2] frontier fit decision_tree depth={depth}", flush=True)
        tree = DecisionTreeClassifier(max_depth=depth, random_state=seed, min_samples_leaf=2)
        tree.fit(train.C, train.y)
        P = predict_proba_aligned(tree, test.C, classes)
        rows.append(
            dict(
                subset=bundle.subset,
                source=bundle.source,
                detector_seed=bundle.detector_seed,
                method="decision_tree",
                budget=f"max_depth={depth}",
                accuracy=accuracy_score(test.y, P.argmax(1)),
                macro_f1=f1_score(test.y, P.argmax(1), average="macro", zero_division=0),
                complexity=model_complexity(tree, "decision_tree"),
            )
        )
    frontier = pd.DataFrame(rows)
    _append_csv(output_dir / "e2_frontier.csv", frontier)
    importance_path = output_dir / f"e2_importance_{bundle.subset}_{bundle.source}.csv"
    if bundle.detector_seed is not None:
        importance_path = output_dir / (
            f"e2_importance_{bundle.subset}_{bundle.source}_seed{bundle.detector_seed}.csv"
        )
    pd.DataFrame(importance).to_csv(importance_path, index=False)


def describe_ferl_rules(model, concept_names: list[str], max_rules: int = 25) -> tuple[str, list[dict]]:
    tree = getattr(model, "tree_", model)
    nodes = [node for node in getattr(tree, "_cached_all_nodes", []) if node.get("path_length", 0) > 0]
    if not nodes and hasattr(tree, "_extract_all_nodes"):
        nodes = [node for node in tree._extract_all_nodes() if node.get("path_length", 0) > 0]
    nodes = sorted(nodes, key=lambda n: (-float(n.get("coverage", 0.0)), n.get("path_length", 0)))
    lines = [
        f"FERL rule dump: {len(nodes)} non-root rules",
        f"Showing top {min(max_rules, len(nodes))} by training coverage",
        "",
    ]
    importance_counts: Counter[int] = Counter()
    importance_weight: Counter[int] = Counter()
    for rank, node in enumerate(nodes[:max_rules], start=1):
        conditions = []
        for feature, fuzzy_set in zip(node.get("path_features", []), node.get("path_fuzzy_sets", [])):
            feature = int(feature)
            fuzzy_set = int(fuzzy_set)
            concept = concept_names[feature] if feature < len(concept_names) else f"feature_{feature}"
            set_name = _fuzzy_set_name(tree, feature, fuzzy_set)
            conditions.append(f"{concept} is {set_name}")
            importance_counts[feature] += 1
            importance_weight[feature] += float(node.get("coverage", 0.0))
        antecedent = " AND ".join(conditions) if conditions else "TRUE"
        lines.append(
            f"{rank}. IF {antecedent} THEN class={node.get('prediction')} "
            f"(coverage={float(node.get('coverage', 0.0)):.4f}, depth={node.get('path_length', 0)})"
        )
    importance = [
        dict(
            concept_index=idx,
            concept_name=concept_names[idx] if idx < len(concept_names) else f"feature_{idx}",
            rule_count=int(importance_counts[idx]),
            coverage_weight=float(importance_weight[idx]),
        )
        for idx, _ in importance_counts.most_common()
    ]
    return "\n".join(lines) + "\n", importance


def _fuzzy_set_name(tree, feature: int, fuzzy_set: int) -> str:
    try:
        return str(tree.fuzzy_partitions[feature][fuzzy_set].name)
    except Exception:
        return f"set_{fuzzy_set}"


def run_e3(
    bundle: ArtifactBundle,
    *,
    methods: list[str],
    seed: int,
    alpha: float,
    output_dir: Path,
) -> None:
    _, train, val, test = encoded_splits(bundle)
    classes = np.arange(len(np.unique(np.concatenate([train.y, val.y, test.y]))))
    rows = []
    risk_rows = []
    for method in methods:
        print(f"[cub_cbm:e3] fit source={bundle.source} detector={bundle.detector_seed} method={method}", flush=True)
        model = make_model(method, seed)
        model.fit(train.C, train.y)
        P = predict_proba_aligned(model, test.C, classes)
        aurc, covs, risks = selective_risk_coverage(P, P.max(1), test.y, classes)
        rows.append(
            _selective_row(bundle, method, seed, "max_proba", aurc, P, test.y, classes)
        )
        for cov, risk in zip(covs, risks):
            risk_rows.append(_risk_row(bundle, method, "max_proba", cov, risk))

        if method in CONFIGS or method == "ferl-deep":
            tree = getattr(model, "tree_", None)
            if tree is not None:
                firing = np.maximum(tree.firing_strength(test.C), 1e-8)
                aurc_f, covs_f, risks_f = selective_risk_coverage(P, firing, test.y, classes)
                rows.append(
                    _selective_row(bundle, method, seed, "firing", aurc_f, P, test.y, classes)
                )
                for cov, risk in zip(covs_f, risks_f):
                    risk_rows.append(_risk_row(bundle, method, "firing", cov, risk))

                conformal_model = tree
                cp = ConformalFERL(conformal_model, score="firing").calibrate(val.C, val.y, alpha=alpha)
                sets = cp.predict_set(test.C)
                rows.append(
                    dict(
                        subset=bundle.subset,
                        source=bundle.source,
                        detector_seed=bundle.detector_seed,
                        method=method,
                        seed=seed,
                        score="conformal_firing",
                        aurc=np.nan,
                        accuracy=accuracy_score(test.y, P.argmax(1)),
                        macro_f1=f1_score(test.y, P.argmax(1), average="macro", zero_division=0),
                        coverage=coverage(sets, test.y, conformal_model.classes_),
                        avg_set_size=avg_set_size(sets),
                        abstention_rate=abstention_rate(sets),
                        accepted_accuracy=_accepted_accuracy(sets, P, test.y, conformal_model.classes_),
                    )
                )
            if getattr(model, "credal", False) or method == "ferl-deep":
                sets = model.predict_set(test.C)
                rows.append(
                    dict(
                        subset=bundle.subset,
                        source=bundle.source,
                        detector_seed=bundle.detector_seed,
                        method=method,
                        seed=seed,
                        score="native_credal",
                        aurc=np.nan,
                        accuracy=accuracy_score(test.y, P.argmax(1)),
                        macro_f1=f1_score(test.y, P.argmax(1), average="macro", zero_division=0),
                        coverage=coverage(sets, test.y, model.classes_),
                        avg_set_size=avg_set_size(sets),
                        abstention_rate=abstention_rate(sets),
                        accepted_accuracy=_accepted_accuracy(sets, P, test.y, model.classes_),
                    )
                )

    selective = pd.DataFrame(rows)
    risk_curve = pd.DataFrame(risk_rows)
    _append_csv(output_dir / "e3_selective.csv", selective)
    _append_csv(output_dir / "e3_risk_coverage.csv", risk_curve)
    plot_risk_coverage(risk_curve, output_dir / f"e3_risk_coverage_{bundle.subset}_{bundle.source}.png")


def _selective_row(bundle, method, seed, score, aurc, P, y, classes) -> dict:
    return dict(
        subset=bundle.subset,
        source=bundle.source,
        detector_seed=bundle.detector_seed,
        method=method,
        seed=seed,
        score=score,
        aurc=aurc,
        accuracy=accuracy_score(y, P.argmax(1)),
        macro_f1=f1_score(y, P.argmax(1), average="macro", zero_division=0),
        coverage=np.nan,
        avg_set_size=np.nan,
        abstention_rate=np.nan,
        accepted_accuracy=np.nan,
    )


def _risk_row(bundle, method, score, cov, risk) -> dict:
    return dict(
        subset=bundle.subset,
        source=bundle.source,
        detector_seed=bundle.detector_seed,
        method=method,
        score=score,
        coverage=float(cov),
        risk=float(risk),
    )


def _accepted_accuracy(sets: np.ndarray, P: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    sizes = sets.sum(axis=1)
    accepted = sizes == 1
    if not np.any(accepted):
        return float("nan")
    idx = {c: i for i, c in enumerate(classes)}
    cols = np.array([idx[v] for v in y])
    return float((P.argmax(1)[accepted] == cols[accepted]).mean())


def plot_risk_coverage(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for (method, score), group in df.groupby(["method", "score"]):
        group = group.sort_values("coverage")
        ax.plot(group["coverage"], group["risk"], marker="o", linewidth=1.3, label=f"{method}:{score}")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def run_e4(
    bundle: ArtifactBundle,
    *,
    artifact_dir: Path,
    detector_seeds: list[int],
    seed: int,
    alpha: float,
    output_dir: Path,
    noise_levels: list[float],
    missing_levels: list[float],
) -> None:
    if bundle.source != "predicted":
        return
    rows = []
    if detector_seeds and bundle.detector_seed == detector_seeds[0]:
        rows.extend(_e4_detector_swap(artifact_dir, bundle.subset, detector_seeds, seed, alpha))
    rows.extend(_e4_noise(bundle, seed, alpha, noise_levels))
    rows.extend(_e4_missing(bundle, seed, alpha, missing_levels))
    df = pd.DataFrame(rows)
    _append_csv(output_dir / "e4_robustness.csv", df)
    for kind in ("detector_swap", "concept_noise", "missing_concepts"):
        plot_e4_curve(
            df[df["degradation"] == kind],
            output_dir / f"e4_{kind}_{bundle.subset}.png",
            title=kind.replace("_", " "),
        )


def _e4_detector_swap(
    artifact_dir: Path,
    subset: str,
    detector_seeds: list[int],
    seed: int,
    alpha: float,
) -> list[dict]:
    rows = []
    if len(detector_seeds) < 2:
        return rows
    for train_seed in detector_seeds:
        print(f"[cub_cbm:e4] detector swap train_seed={train_seed}", flush=True)
        train_bundle = read_bundle(
            artifact_dir, source="predicted", detector_seed=train_seed, subset=subset
        )
        _, train, val, _ = encoded_splits(train_bundle)
        model = make("ferl-medium", random_state=seed)
        model.fit(train.C, train.y)
        cp = ConformalFERL(model.tree_, score="firing").calibrate(val.C, val.y, alpha=alpha)
        for test_seed in detector_seeds:
            print(f"[cub_cbm:e4] detector swap train_seed={train_seed} test_seed={test_seed}", flush=True)
            test_bundle = read_bundle(
                artifact_dir, source="predicted", detector_seed=test_seed, subset=subset
            )
            enc = LabelEncoder().fit(
                np.concatenate([train_bundle.train.y, train_bundle.val.y, test_bundle.test.y])
            )
            test_y = enc.transform(test_bundle.test.y)
            P = predict_proba_aligned(model, test_bundle.test.C, model.classes_)
            sets = cp.predict_set(test_bundle.test.C)
            rows.append(
                _robust_row(
                    subset=subset,
                    source="predicted",
                    detector_seed=train_seed,
                    degradation="detector_swap",
                    level=float(test_seed),
                    method="ferl-medium",
                    P=P,
                    y=test_y,
                    classes=model.classes_,
                    sets=sets,
                    extra={"test_detector_seed": test_seed},
                )
            )
    return rows


def _e4_noise(bundle: ArtifactBundle, seed: int, alpha: float, levels: list[float]) -> list[dict]:
    _, train, val, test = encoded_splits(bundle)
    model = make("ferl-medium", random_state=seed)
    model.fit(train.C, train.y)
    cp = ConformalFERL(model.tree_, score="firing").calibrate(val.C, val.y, alpha=alpha)
    rng = np.random.default_rng(seed)
    rows = []
    for rho in levels:
        print(f"[cub_cbm:e4] concept noise rho={rho}", flush=True)
        X = perturb_concepts(test.C, rho, rng)
        P = predict_proba_aligned(model, X, model.classes_)
        sets = cp.predict_set(X)
        rows.append(
            _robust_row(
                subset=bundle.subset,
                source=bundle.source,
                detector_seed=bundle.detector_seed,
                degradation="concept_noise",
                level=rho,
                method="ferl-medium",
                P=P,
                y=test.y,
                classes=model.classes_,
                sets=sets,
            )
        )
    return rows


def _e4_missing(bundle: ArtifactBundle, seed: int, alpha: float, levels: list[float]) -> list[dict]:
    _, train, val, test = encoded_splits(bundle)
    model = make("ferl-medium", random_state=seed)
    model.fit(train.C, train.y)
    cp = ConformalFERL(model.tree_, score="firing").calibrate(val.C, val.y, alpha=alpha)
    rng = np.random.default_rng(seed + 1009)
    rows = []
    for frac in levels:
        print(f"[cub_cbm:e4] missing concepts frac={frac}", flush=True)
        observed_mask = rng.random(test.C.shape) >= frac
        X = np.where(observed_mask, test.C, 0.5)
        P = model.tree_.predict_proba(X, observed_mask=observed_mask)
        sets = cp.predict_set_masked(X, observed_mask=observed_mask)
        rows.append(
            _robust_row(
                subset=bundle.subset,
                source=bundle.source,
                detector_seed=bundle.detector_seed,
                degradation="missing_concepts",
                level=frac,
                method="ferl-medium",
                P=P,
                y=test.y,
                classes=model.classes_,
                sets=sets,
            )
        )
    return rows


def perturb_concepts(X: np.ndarray, rho: float, rng: np.random.Generator) -> np.ndarray:
    Xn = np.array(X, copy=True)
    mask = rng.random(Xn.shape) < rho
    # Push selected concept activations to ambiguity instead of creating
    # out-of-distribution binary flips for probability-valued detector outputs.
    Xn[mask] = 0.5
    return Xn


def _robust_row(
    *,
    subset: str,
    source: str,
    detector_seed: int | None,
    degradation: str,
    level: float,
    method: str,
    P: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    sets: np.ndarray,
    extra: dict | None = None,
) -> dict:
    sizes = sets.sum(axis=1)
    accepted = sizes == 1
    cols = np.array([{c: i for i, c in enumerate(classes)}[v] for v in y])
    row = dict(
        subset=subset,
        source=source,
        detector_seed=detector_seed,
        degradation=degradation,
        level=level,
        method=method,
        accuracy=accuracy_score(y, P.argmax(1)),
        accuracy_on_accepted=float((P.argmax(1)[accepted] == cols[accepted]).mean()) if accepted.any() else np.nan,
        abstention_rate=abstention_rate(sets),
        conformal_coverage=coverage(sets, y, classes),
        avg_set_size=avg_set_size(sets),
    )
    if extra:
        row.update(extra)
    return row


def plot_e4_curve(df: pd.DataFrame, path: Path, title: str) -> None:
    if df.empty or "level" not in df:
        return
    agg = df.groupby("level", dropna=False)[
        ["accuracy", "accuracy_on_accepted", "abstention_rate", "conformal_coverage"]
    ].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for col in ["accuracy", "accuracy_on_accepted", "abstention_rate", "conformal_coverage"]:
        ax.plot(agg["level"], agg[col], marker="o", label=col)
    ax.set_title(title)
    ax.set_xlabel("Degradation level")
    ax.set_ylabel("Metric")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _append_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates()
    df.to_csv(path, index=False)


def create_smoke_artifact(path: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True, exist_ok=True)
    n_concepts = 12
    n_classes = 3
    concept_names = [f"concept_{idx:02d}" for idx in range(n_concepts)]
    (path / "concept_names.json").write_text(json.dumps(concept_names, indent=2) + "\n", encoding="utf-8")
    weights = rng.normal(size=(n_concepts, n_classes))

    def split(n: int, name: str) -> None:
        C = rng.binomial(1, 0.35, size=(n, n_concepts)).astype(float)
        logits = C @ weights + rng.normal(scale=0.35, size=(n, n_classes))
        y = logits.argmax(axis=1)
        ids = np.arange(n)
        np.savez(path / f"oracle_{name}.npz", C=C, y=y, ids=ids)
        for detector_seed in (0, 1):
            srng = np.random.default_rng(seed + 37 * detector_seed + len(name))
            P = np.clip(C * 0.75 + (1.0 - C) * 0.25 + srng.normal(scale=0.12, size=C.shape), 0.0, 1.0)
            np.savez(path / f"predicted_seed{detector_seed}_{name}.npz", C=P, y=y, ids=ids)

    split(90, "train")
    split(45, "val")
    split(60, "test")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, help="CUB concept artifact directory")
    parser.add_argument("--subset", help="Subset label for output tables; defaults to artifact dir name")
    parser.add_argument("--output-dir", type=Path, default=Path("results/cub_cbm"))
    parser.add_argument("--concept-sources", nargs="+", default=["oracle"], choices=["oracle", "predicted"])
    parser.add_argument("--detector-seeds", nargs="+", type=int, default=[])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--methods", nargs="+", default=list(ALL_METHODS))
    parser.add_argument("--experiments", nargs="+", default=["e1"], choices=["e1", "e2", "e3", "e4", "all"])
    parser.add_argument("--e2-config", default="ferl-medium", choices=sorted(CONFIGS))
    parser.add_argument("--frontier-depths", nargs="+", type=int, default=[1, 2, 3, 4, 5, 8])
    parser.add_argument("--learned-depth", type=int, default=None,
                        help="max_depth for ferl-deep in E1-E3 (match the extras, "
                             "e.g. 60); default keeps LearnedFuzzyTree's own default (12)")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--missing-levels", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Generate a small synthetic artifact under /tmp and run e1/e2/e3/e4.",
    )
    return parser.parse_args()


def main() -> int:
    global LEARNED_DEPTH
    args = parse_args()
    LEARNED_DEPTH = args.learned_depth
    if args.smoke:
        args.artifact_dir = create_smoke_artifact(Path("/tmp/ferl_cub_cbm_smoke"))
        args.concept_sources = ["oracle", "predicted"]
        args.detector_seeds = [0, 1]
        args.experiments = ["e1", "e2", "e3", "e4"]
        args.seeds = [0]
        args.output_dir = Path("/tmp/ferl_cub_cbm_smoke_results")
    if args.artifact_dir is None:
        raise SystemExit("--artifact-dir is required unless --smoke is used")
    experiments = {"e1", "e2", "e3", "e4"} if "all" in args.experiments else set(args.experiments)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundles: list[ArtifactBundle] = []
    if "oracle" in args.concept_sources:
        bundles.append(read_bundle(args.artifact_dir, source="oracle", subset=args.subset))
    if "predicted" in args.concept_sources:
        if not args.detector_seeds:
            raise SystemExit("--concept-sources predicted requires --detector-seeds")
        bundles.extend(
            read_bundle(
                args.artifact_dir,
                source="predicted",
                detector_seed=detector_seed,
                subset=args.subset,
            )
            for detector_seed in args.detector_seeds
        )

    for bundle in bundles:
        print(
            f"{bundle.subset} {bundle.source}"
            f"{'' if bundle.detector_seed is None else f' seed={bundle.detector_seed}'} "
            f"train={bundle.train.C.shape} val={bundle.val.C.shape} test={bundle.test.C.shape}"
        )
        if "e1" in experiments:
            run_e1(bundle, methods=args.methods, seeds=args.seeds, output_dir=args.output_dir)
        if "e2" in experiments:
            run_e2(
                bundle,
                ferl_config=args.e2_config,
                methods=args.methods,
                seed=args.seeds[0],
                output_dir=args.output_dir,
                frontier_depths=args.frontier_depths,
            )
        if "e3" in experiments:
            run_e3(
                bundle,
                methods=args.methods,
                seed=args.seeds[0],
                alpha=args.alpha,
                output_dir=args.output_dir,
            )
        if "e4" in experiments:
            run_e4(
                bundle,
                artifact_dir=args.artifact_dir,
                detector_seeds=args.detector_seeds,
                seed=args.seeds[0],
                alpha=args.alpha,
                output_dir=args.output_dir,
                noise_levels=args.noise_levels,
                missing_levels=args.missing_levels,
            )
    print(f"wrote CUB-CBM FERL outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
