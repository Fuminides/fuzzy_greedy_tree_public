"""Fresh wall-clock runtime benchmark for FERL variants and common baselines.

This intentionally does not reuse ``results/bench`` artifacts from
``harness.py`` because those may have been produced by an older implementation.
It fits each estimator from scratch, records fit and ``predict_proba`` times,
and writes fold-level plus model-level summaries.

Run from the repository root:

    PYTHONPATH=$(pwd) python experiments/benchmark2/wallclock_runtime.py
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ferl.pipeline.run_configs import load_filtered
import models as M

warnings.filterwarnings("ignore")

DEFAULT_DATASETS = ["iris", "wine", "heart", "vehicle", "banana"]
DEFAULT_MODELS = [
    "FERL-compact",
    "FERL-enhanced",
    "FERL-medium",
    "CART",
    "C45",
    "RuleFit",
    "FIGS",
    "FURIA",
    "LogReg",
    "MLP",
]

# Keep implementation keys in the raw artifacts, but use the paper's operating-
# point names in summaries and tables.  In particular, ``FERL-medium`` is
# the deepest of the three FuzzyCARTFast configurations timed by this script.
# Paper variants: FERL-compact and FERL-medium are the ferl-compact and
# ferl-medium configs, FERL-deep is LearnedFuzzyTree.
# ("FERL-enhanced" is the ferl-enhanced config, not a paper variant.)
DISPLAY_NAMES = {
    "FERL-compact": "FERL-compact",
    "FERL-medium": "FERL-medium",
    "FERL-deep": "FERL-deep",
}
FERL_MODELS = frozenset(DISPLAY_NAMES)
# FERL-compact/medium must run on the compiled ferl_fast kernels; FERL-deep
# (LearnedFuzzyTree) is pure NumPy and has no compiled backend.
FAST_FERL = frozenset({"FERL-compact", "FERL-medium"})


def _accuracy(est, X, y, proba):
    pred = M.classes_of(est)[np.argmax(proba, axis=1)]
    return float(np.mean(pred == y))


def run(datasets, models, folds, out_dir):
    rows = []
    os.makedirs(out_dir, exist_ok=True)

    for ds in datasets:
        try:
            X, y = load_filtered(ds)
        except Exception as ex:
            print(f"{ds}: SKIP load ({ex})", flush=True)
            continue
        if len(np.unique(y)) < 2 or len(y) < folds:
            print(f"{ds}: SKIP too few samples/classes", flush=True)
            continue

        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=33)
        print(f"{ds}: n={len(y)} d={X.shape[1]} C={len(np.unique(y))}", flush=True)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
            for name in models:
                row = {
                    "dataset": ds,
                    "fold": fold,
                    "model": name,
                    "n_train": len(ytr),
                    "n_test": len(yte),
                    "n_features": X.shape[1],
                    "n_classes": len(np.unique(y)),
                    "status": "ok",
                }
                try:
                    np.random.seed(fold)
                    est = M.build(name)
                    t0 = time.perf_counter()
                    est.fit(Xtr, ytr)
                    row["train_s"] = time.perf_counter() - t0

                    row["implementation"] = getattr(est, "implementation_", "")
                    if name in FAST_FERL and row["implementation"] != "fast":
                        raise RuntimeError(
                            f"{DISPLAY_NAMES[name]} did not use the compiled ferl_fast backend"
                        )

                    t0 = time.perf_counter()
                    proba = est.predict_proba(Xte)
                    row["predict_s"] = time.perf_counter() - t0

                    row["acc"] = _accuracy(est, Xte, yte, proba)
                    row["complexity"] = M.complexity(name, est)
                except Exception as ex:
                    row.update({
                        "status": "dnf",
                        "err": f"{type(ex).__name__}: {str(ex)[:160]}",
                        "train_s": np.nan,
                        "predict_s": np.nan,
                        "acc": np.nan,
                        "complexity": np.nan,
                        "implementation": "",
                    })
                    print(f"  {ds}/fold{fold}/{name}: DNF {row['err']}", flush=True)
                rows.append(row)
            print(f"  fold {fold} done", flush=True)

    df = pd.DataFrame(rows)
    fold_csv = os.path.join(out_dir, "wallclock_runtime.csv")
    summary_csv = os.path.join(out_dir, "wallclock_runtime_summary.csv")
    df.to_csv(fold_csv, index=False)

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print(f"No successful fits. Fold-level results: {fold_csv}")
        return df, pd.DataFrame()

    ok["pred_ms_per_1k"] = ok["predict_s"] / ok["n_test"] * 1000.0 * 1000.0
    ok["fit_predict_s"] = ok["train_s"] + ok["predict_s"]
    summary = ok.groupby("model", sort=False).agg(
        n_fits=("status", "size"),
        acc=("acc", "mean"),
        complexity=("complexity", "mean"),
        train_s_mean=("train_s", "mean"),
        train_s_median=("train_s", "median"),
        train_s_total=("train_s", "sum"),
        predict_s_mean=("predict_s", "mean"),
        fit_predict_s_mean=("fit_predict_s", "mean"),
        fit_predict_s_total=("fit_predict_s", "sum"),
        pred_ms_per_1k=("pred_ms_per_1k", "mean"),
    ).reset_index()
    summary.insert(1, "method", summary["model"].map(DISPLAY_NAMES).fillna(summary["model"]))
    summary.to_csv(summary_csv, index=False)

    print("\n=== Wall-clock summary ===")
    print(summary.to_string(index=False, formatters={
        "acc": "{:.4f}".format,
        "complexity": "{:.1f}".format,
        "train_s_mean": "{:.4f}".format,
        "train_s_median": "{:.4f}".format,
        "train_s_total": "{:.2f}".format,
        "predict_s_mean": "{:.5f}".format,
        "fit_predict_s_mean": "{:.4f}".format,
        "fit_predict_s_total": "{:.2f}".format,
        "pred_ms_per_1k": "{:.3f}".format,
    }))
    print(f"\nWrote {fold_csv}")
    print(f"Wrote {summary_csv}")
    return df, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    print(f"Wallclock benchmark | datasets={a.datasets} | models={a.models} | folds={a.folds}")
    run(a.datasets, a.models, a.folds, a.out_dir)


if __name__ == "__main__":
    main()
