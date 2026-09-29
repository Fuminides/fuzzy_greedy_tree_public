"""Seeded wall-clock scaling benchmark across representative datasets.

Unlike ``wallclock_runtime.py``, which gives one aggregate across five datasets,
this benchmark keeps a row per dataset and orders the paper table by the size of
the input matrix.  Each run is one seeded stratified train/test split; summaries
report medians and interquartile ranges so an unusually busy system or an easy
split does not define the claimed "typical" runtime.

Run from the repository root::

    PYTHONPATH=. python experiments/benchmark2/wallclock_scaling.py
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ferl.pipeline.run_configs import load_filtered
import models as M
from wallclock_runtime import DISPLAY_NAMES, FAST_FERL, FERL_MODELS

warnings.filterwarnings("ignore")

# Ordered by approximately increasing N*d in the unfiltered benchmark data.
# The set spans 178--5,300 samples, 2--57 features, and 2--7 classes while
# remaining suitable for a repeated workstation run of every baseline.
DEFAULT_DATASETS = [
    "glass",
    "wine",
    "heart",
    "australian",
    "banana",
    "ionosphere",
    "vehicle",
    "german",
    "segment",
    "spambase",
]
DEFAULT_MODELS = [
    "FERL-compact",
    "FERL-medium",
    "FERL-deep",
    "CART",
    "C45",
    "FIGS",
    "FURIA",
    "RuleFit",
]
DEFAULT_SEEDS = [11, 22, 33, 44, 55]
PRETTY = {"C45": "C4.5", **DISPLAY_NAMES}


class _FitTimedOut(TimeoutError):
    pass


@contextmanager
def _fit_timeout(seconds: float | None):
    """Bound a fit on Unix; zero/None disables the timer."""
    if not seconds or not hasattr(signal, "setitimer"):
        yield
        return

    def _raise_timeout(_signum, _frame):
        raise _FitTimedOut(f"fit exceeded {seconds:g} seconds")

    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _existing_keys(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    try:
        old = pd.read_csv(path)
    except Exception:
        return set()
    ok = old[old["status"] == "ok"]
    return set(zip(ok["dataset"], ok["seed"].astype(int), ok["model"]))


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _quantile(series, q):
    return float(series.quantile(q))


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    ok = rows[rows["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    ok["data_cells"] = ok["n_samples"] * ok["n_features"]
    ok["pred_ms_per_1k"] = ok["predict_s"] / ok["n_test"] * 1_000_000.0
    summary = ok.groupby(["dataset", "model"], sort=False).agg(
        method=("method", "first"),
        n_runs=("seed", "size"),
        n_samples=("n_samples", "first"),
        n_features=("n_features", "first"),
        n_classes=("n_classes", "first"),
        data_cells=("data_cells", "first"),
        complexity_median=("complexity", "median"),
        acc_median=("acc", "median"),
        fit_s_median=("train_s", "median"),
        fit_s_q1=("train_s", lambda x: _quantile(x, 0.25)),
        fit_s_q3=("train_s", lambda x: _quantile(x, 0.75)),
        predict_s_median=("predict_s", "median"),
        predict_s_q1=("predict_s", lambda x: _quantile(x, 0.25)),
        predict_s_q3=("predict_s", lambda x: _quantile(x, 0.75)),
        pred_ms_per_1k_median=("pred_ms_per_1k", "median"),
    ).reset_index()
    return summary.sort_values(["data_cells", "dataset", "model"], kind="stable")


def write_latex_table(summary: pd.DataFrame, path: Path, models: list[str], seeds: list[int]) -> None:
    """Write the supplementary training-time table from the summary artifact."""
    if summary.empty:
        return
    dataset_info = (
        summary[["dataset", "n_samples", "n_features", "n_classes", "data_cells"]]
        .drop_duplicates("dataset")
        .sort_values(["data_cells", "dataset"], kind="stable")
    )
    med = summary.pivot(index="dataset", columns="model", values="fit_s_median")
    nodes = summary.pivot(index="dataset", columns="model", values="complexity_median")

    ferl = [m for m in models if m in FERL_MODELS]
    base = [m for m in models if m not in FERL_MODELS]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{\textbf{Median fit time in seconds} over " + str(len(seeds))
        + r" seeded stratified 70/30 train/test splits, one CPU, one process. Datasets are "
        + r"ordered by the input-matrix size $N\!\times\!d$ (Table~\ref{tab:datasets}). "
        + r"FERL-compact and FERL-medium use the compiled \texttt{ferl\_fast} kernels; "
        + r"FERL-deep is pure NumPy. Per-seed timings and interquartile ranges are in the "
        + r"released CSV files.}",
        r"\label{tab:runtime}",
        r"\begin{tabular}{@{}l" + "r" * len(models) + r"@{}}",
        r"\toprule",
        r" & \multicolumn{" + str(len(ferl)) + r"}{c}{FERL} & "
        r"\multicolumn{" + str(len(base)) + r"}{c}{Baselines} \\",
        r"\cmidrule(lr){2-" + str(1 + len(ferl)) + r"}\cmidrule(l){" + str(2 + len(ferl))
        + "-" + str(1 + len(models)) + "}",
        "Dataset & " + " & ".join(PRETTY.get(model, model).replace("FERL-", "") for model in ferl + base)
        + r" \\",
        r"\midrule",
    ]
    for row in dataset_info.itertuples(index=False):
        values = []
        for model in ferl + base:
            value = med.loc[row.dataset, model] if model in med.columns else np.nan
            values.append("--" if pd.isna(value) else (f"{value:.3f}" if value < 1 else f"{value:.2f}" if value < 10 else f"{value:.1f}"))
        lines.append(f"{row.dataset} & " + " & ".join(values) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(args.out_dir)
    raw_path = out_dir / "wallclock_scaling.csv"
    summary_path = out_dir / "wallclock_scaling_summary.csv"
    if not args.resume and raw_path.exists():
        raw_path.unlink()
    completed = _existing_keys(raw_path) if args.resume else set()

    for dataset in args.datasets:
        try:
            X, y = load_filtered(dataset)
        except Exception as ex:
            print(f"{dataset}: SKIP load ({ex})", flush=True)
            continue
        n_classes = len(np.unique(y))
        print(f"{dataset}: N={len(y)} d={X.shape[1]} C={n_classes}", flush=True)
        for seed in args.seeds:
            tr, te = train_test_split(
                np.arange(len(y)),
                test_size=args.test_size,
                random_state=seed,
                stratify=y,
            )
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            for name in args.models:
                key = (dataset, seed, name)
                if key in completed:
                    continue
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "model": name,
                    "method": DISPLAY_NAMES.get(name, PRETTY.get(name, name)),
                    "n_samples": len(y),
                    "n_train": len(tr),
                    "n_test": len(te),
                    "n_features": X.shape[1],
                    "n_classes": n_classes,
                    "test_size": args.test_size,
                    "predict_repeats": 0,
                    "status": "ok",
                    "train_s": np.nan,
                    "predict_s": np.nan,
                    "acc": np.nan,
                    "complexity": np.nan,
                    "implementation": "",
                    "err": "",
                }
                try:
                    np.random.seed(seed)
                    estimator = M.build(name)
                    t0 = time.perf_counter()
                    with _fit_timeout(args.max_fit_seconds):
                        estimator.fit(Xtr, ytr)
                    row["train_s"] = time.perf_counter() - t0
                    row["implementation"] = getattr(estimator, "implementation_", "")
                    if name in FAST_FERL and row["implementation"] != "fast":
                        raise RuntimeError(
                            f"{DISPLAY_NAMES[name]} did not use the compiled ferl_fast backend"
                        )

                    # Repeat only cheap predictions: this stabilises sub-ms calls
                    # without multiplying genuinely slow FURIA/RuleFit inference.
                    t0 = time.perf_counter()
                    proba = estimator.predict_proba(Xte)
                    elapsed = time.perf_counter() - t0
                    repeats = 1
                    while repeats < args.predict_repeats and elapsed < args.min_predict_seconds:
                        t0 = time.perf_counter()
                        estimator.predict_proba(Xte)
                        elapsed += time.perf_counter() - t0
                        repeats += 1
                    row["predict_repeats"] = repeats
                    row["predict_s"] = elapsed / repeats
                    pred = M.classes_of(estimator)[np.argmax(proba, axis=1)]
                    row["acc"] = float(np.mean(pred == yte))
                    row["complexity"] = M.complexity(name, estimator)
                except Exception as ex:
                    row["status"] = "timeout" if isinstance(ex, _FitTimedOut) else "dnf"
                    row["err"] = f"{type(ex).__name__}: {str(ex)[:180]}"
                    print(
                        f"  seed={seed} {name}: {row['status'].upper()} {row['err']}",
                        flush=True,
                    )
                _append_row(raw_path, row)
            print(f"  seed {seed} done", flush=True)

    raw = pd.read_csv(raw_path)
    selected = raw[
        raw["dataset"].isin(args.datasets)
        & raw["model"].isin(args.models)
        & raw["seed"].isin(args.seeds)
    ]
    summary = summarise(selected)
    summary.to_csv(summary_path, index=False)
    if args.tex_out:
        write_latex_table(summary, Path(args.tex_out), args.models, args.seeds)
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")
    if args.tex_out:
        print(f"Wrote {args.tex_out}")
    return raw, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--predict-repeats", type=int, default=10)
    parser.add_argument("--min-predict-seconds", type=float, default=0.05)
    parser.add_argument("--max-fit-seconds", type=float, default=300.0)
    parser.add_argument("--out-dir", default="results/runtime_scaling")
    parser.add_argument("--tex-out", default="paper/generated/tab_runtime.tex")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
