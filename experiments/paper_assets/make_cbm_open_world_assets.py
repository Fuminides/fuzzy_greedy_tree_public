"""Generate paper tables for the detector-disjoint open-world CBM study (E16).

Run from the repository root::

    python experiments/paper_assets/make_cbm_open_world_assets.py

The aggregation follows the frozen E16 protocol: each metric is first averaged
over the five held-out-class folds for a detector seed, then summarized across
the three detector seeds. Individual images are never pooled across folds.
"""
from __future__ import annotations

import os

from pathlib import Path

import numpy as np
import pandas as pd


RESULTS = Path("results/cbm_open_world")
GENERATED = Path(os.environ.get("FERL_GEN_DIR", "paper/generated"))
DATASETS = (("cub", "CUB-200"), ("awa2", "AwA2"))
FOLDS = range(5)
SEEDS = range(3)
SIGNALS = (
    ("ferl_residual", "FERL residual"),
    ("ferl_ignorance", "FERL ignorance"),
    ("ferl_set_size", "FERL set size"),
    ("ferl_one_minus_maxp", r"FERL $1-\max p$"),
    ("lr_entropy", "LR entropy"),
    ("lr_one_minus_maxp", r"LR $1-\max p$"),
    ("mahalanobis", "Mahalanobis"),
    ("knn", r"$k$NN distance"),
    ("isolation_forest", "Isolation Forest"),
)
OOD_METRICS = ("auroc", "aupr_out", "fpr95", "ood_rejection")
ROUTER_METRICS = (
    "test_ferl_route_rate",
    "test_ferl_accuracy",
    "test_accuracy",
    "test_lr_accuracy",
)


def _validate_configurations(
    data: pd.DataFrame,
    *,
    key: list[str],
    rows_per_configuration: int,
) -> None:
    expected = {
        (dataset, fold, seed)
        for dataset, _ in DATASETS
        for fold in FOLDS
        for seed in SEEDS
    }
    observed = set(
        zip(
            data["dataset"].astype(str),
            data["fold"].astype(int),
            data["detector_seed"].astype(int),
        )
    )
    if observed != expected:
        raise RuntimeError(
            "incomplete E16 configurations: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    duplicates = int(data.duplicated(key).sum())
    if duplicates:
        raise RuntimeError(f"E16 results contain {duplicates} duplicate key rows")
    counts = data.groupby(["dataset", "fold", "detector_seed"]).size()
    if not counts.eq(rows_per_configuration).all():
        raise RuntimeError(
            "unexpected rows per E16 configuration: "
            f"{sorted(counts.unique().tolist())}"
        )


def _fold_then_seed(
    data: pd.DataFrame,
    *,
    groups: list[str],
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    """Macro-average folds within each seed, then summarize detector seeds."""
    per_seed = (
        data.groupby([*groups, "detector_seed"], as_index=False)[list(metrics)]
        .mean()
    )
    return per_seed.groupby(groups)[list(metrics)].agg(["mean", "std"])


def _pm(mean: float, std: float) -> str:
    return f"{100.0 * mean:.1f}\\,$\\pm$\\,{100.0 * std:.1f}"


def make_ood_table(data: pd.DataFrame) -> Path:
    _validate_configurations(
        data,
        key=["dataset", "fold", "detector_seed", "signal"],
        rows_per_configuration=len(SIGNALS),
    )
    if data[list(OOD_METRICS)].isna().any().any():
        raise RuntimeError("E16 OOD table contains missing primary metrics")
    observed_signals = set(data["signal"])
    expected_signals = {signal for signal, _ in SIGNALS}
    if observed_signals != expected_signals:
        raise RuntimeError(
            "unexpected E16 OOD signals: "
            f"missing={sorted(expected_signals - observed_signals)}, "
            f"unexpected={sorted(observed_signals - expected_signals)}"
        )

    summary = _fold_then_seed(
        data,
        groups=["dataset", "signal"],
        metrics=OOD_METRICS,
    )
    best: dict[tuple[str, str], float] = {}
    for dataset, _ in DATASETS:
        for metric in OOD_METRICS:
            values = summary.loc[dataset][(metric, "mean")]
            best[(dataset, metric)] = (
                float(values.min()) if metric == "fpr95" else float(values.max())
            )

    def cell(dataset: str, signal: str, metric: str) -> str:
        row = summary.loc[(dataset, signal)]
        value = _pm(row[(metric, "mean")], row[(metric, "std")])
        if np.isclose(row[(metric, "mean")], best[(dataset, metric)]):
            return rf"\textbf{{{value}}}"
        return value

    # One block per dataset, stacked vertically, so the table fits a
    # single-column journal page.
    rows: list[str] = []
    for d_index, (dataset, dataset_label) in enumerate(DATASETS):
        if d_index:
            rows.append(r"\midrule")
        rows.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textbf{{{dataset_label}}}}}\\")
        for index, (signal, label) in enumerate(SIGNALS):
            if index == 4:
                rows.append(r"\cmidrule(l){1-5}")
            values = [cell(dataset, signal, metric) for metric in OOD_METRICS]
            rows.append(f"{label} & " + " & ".join(values) + r"\\")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{Detector-class-disjoint open-world CBM detection.} "
        r"AUROC, AUPR-Out, FPR95, and unseen-class rejection at a threshold "
        r"fixed to accept $95\%$ of retained-class validation inputs (all on "
        r"the 0--100 scale). Within each dataset, the first four rows are FERL's "
        r"native signals and the rest are reference signals and dedicated "
        r"detectors fitted on the same calibrated concept vectors. We first "
        r"macro-average the five held-out-class folds within each detector "
        r"seed and report mean\,$\pm$\,std over three seeds. Higher is better "
        r"except for FPR95. Best entries per dataset and metric are bold.}",
        r"\label{tab:cbm-open-world-ood}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Signal & AUROC & AUPR-Out & FPR95 & Reject\\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    GENERATED.mkdir(parents=True, exist_ok=True)
    output = GENERATED / "tab_cbm_open_world_ood.tex"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def make_router_table(data: pd.DataFrame) -> Path:
    _validate_configurations(
        data,
        key=["dataset", "fold", "detector_seed"],
        rows_per_configuration=1,
    )
    if data[list(ROUTER_METRICS)].isna().any().any():
        raise RuntimeError("E16 learned-router table contains missing metrics")
    summary = _fold_then_seed(
        data,
        groups=["dataset"],
        metrics=ROUTER_METRICS,
    )
    rows = []
    for dataset, label in DATASETS:
        row = summary.loc[dataset]
        values = [
            _pm(row[(metric, "mean")], row[(metric, "std")])
            for metric in ROUTER_METRICS
        ]
        delta_mean = (
            row[("test_accuracy", "mean")]
            - row[("test_lr_accuracy", "mean")]
        )
        per_seed = (
            data[data["dataset"] == dataset]
            .groupby("detector_seed")[["test_accuracy", "test_lr_accuracy"]]
            .mean()
        )
        delta_std = (
            per_seed["test_accuracy"] - per_seed["test_lr_accuracy"]
        ).std()
        values.append(f"{100.0 * delta_mean:+.1f}\\,$\\pm$\\,{100.0 * delta_std:.1f}")
        rows.append(f"{label} & " + " & ".join(values) + r"\\")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Learned routing on retained classes in the strict "
        r"open-world protocol.} Percentages are macro-averaged over the five "
        r"held-out-class folds within each detector seed, followed by "
        r"mean\,$\pm$\,std over three detector seeds. The routing LR predicts "
        r"FERL correctness and does not inspect head agreement.}",
        r"\label{tab:cbm-open-world-router}",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Dataset & FERL route & FERL acc. & Routed acc. & LR acc. & "
        r"$\Delta$ vs.\ LR (pp)\\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    GENERATED.mkdir(parents=True, exist_ok=True)
    output = GENERATED / "tab_cbm_open_world_router.tex"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> None:
    ood = pd.read_csv(RESULTS / "e16_open_world_ood.csv")
    ood_table = make_ood_table(ood)
    print(f"wrote {ood_table}")
    router_path = RESULTS / "e16_open_world_router.csv"
    if router_path.exists():
        router_table = make_router_table(pd.read_csv(router_path))
        print(f"wrote {router_table}")


if __name__ == "__main__":
    main()
