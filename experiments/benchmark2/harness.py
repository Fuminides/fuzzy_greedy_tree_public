"""
Stage A (fit & persist). For each (dataset, fold, model) fit and save raw
artifacts; skip successful artifacts and retry failed/unreadable ones. Metrics
are computed offline by score.py from these files, so changing a metric/alpha
never refits anything.

Saved per (dataset, model, fold):
  proba_test, proba_cal : predict_proba on test and calibration folds
  y_test, y_cal, classes, C
  exact train/cal/test indices, timings, structural complexity, status
  JSON provenance manifest and compressed fitted-estimator archive

Usage (from repo root):
  python experiments/benchmark2/harness.py                 # all datasets/models
  python experiments/benchmark2/harness.py --datasets wine glass --models FERL CART
"""
import os
import sys
import time
import json
import platform
import subprocess
import argparse
import warnings
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from ferl.pipeline.run_configs import load_filtered, SELECTED_30
import artifact_schema as AS
import models as M

warnings.filterwarnings("ignore")
N_FOLDS = 5
CAL_FRAC = 0.25
OUT = os.path.join("results", "bench")
TIME_BUDGET = 600        # seconds/fit; record DNF if exceeded (soft: checked after)
ROOT = Path(__file__).resolve().parents[2]

SOURCE_REVISIONS = {
    "FuzzyUCS-DS": "jUCS:81f8bb6673436fef45fd31929836ce4f443de15b",
    "NeuRules": "supplement-sha256:54c23a2a0e5d6ec40116cf15596e1ffd7eb6b238312778cbbde32f8369afd77a",
}
PUBLIC_REPOS = {
    "SamRuLe": ("FERL_SAMRULE_REPO", "external/SamRuLe"),
    "SamRuLe-OVR": ("FERL_SAMRULE_REPO", "external/SamRuLe"),
    "RRL": ("FERL_RRL_REPO", "external/rrl"),
    "RL-Net": ("FERL_RLNET_REPO", "external/RLNet"),
}


def artifact_path(ds, model, fold):
    return os.path.join(OUT, ds, model, f"fold{fold}.npz")


def _git_revision(path):
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def _source_revision(name):
    if name in PUBLIC_REPOS:
        environment, default = PUBLIC_REPOS[name]
        path = Path(os.environ.get(environment, ROOT / default)).expanduser()
        return _git_revision(path)
    return SOURCE_REVISIONS.get(name, _git_revision(ROOT))


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _resolved_params(estimator):
    params = estimator.get_params(deep=False) if hasattr(estimator, "get_params") else {}
    resolved = {}
    for key, value in params.items():
        fitted_name = f"{key}_"
        resolved[key] = getattr(estimator, fitted_name, value)
    for key in (
        "n_rules_",
        "condition_complexity_",
        "complexity_",
        "best_validation_loss_",
        "sample_sizes_",
        "selected_width_",
        "width_scores_",
    ):
        if hasattr(estimator, key):
            resolved[key[:-1]] = getattr(estimator, key)
    return _jsonable(resolved)


def _rule_summary(name, estimator):
    if name == "NeuRules":
        return _jsonable(estimator.rules_)
    if name in ("SamRuLe", "SamRuLe-OVR"):
        models = []
        for fitted in estimator.models_:
            rules = []
            for condition, prediction in zip(fitted.conditions, fitted.predictions):
                conjunction = estimator.conjunctions_[condition]
                rules.append({
                    "predicates": [estimator.binarizer_.predicate_names_[i] for i in conjunction],
                    "prediction": int(prediction),
                })
            models.append({"rules": rules, "default": int(fitted.predictions[-1])})
        return models
    if name == "FuzzyUCS-DS":
        predicted = [int(np.argmax(rule.weights)) for rule in estimator.population_]
        return {
            "population": len(estimator.population_),
            "rules_per_class": np.bincount(predicted, minlength=len(estimator.classes_)),
            "covering_count": estimator.covering_count_,
            "subsumption_count": estimator.subsumption_count_,
        }
    if name == "RRL":
        return {"logic_conditions": estimator.condition_complexity_}
    if name == "RL-Net":
        return {
            "rules": estimator.n_rules_,
            "active_conditions": estimator.condition_complexity_,
            "best_validation_loss": estimator.best_validation_loss_,
        }
    return None


def _package_versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "scikit-learn", "torch", "imodels", "joblib"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _manifest(ds, name, fold, estimator, train_idx, cal_idx, test_idx, structure):
    params = estimator.get_params(deep=False) if hasattr(estimator, "get_params") else {}
    return {
        "schema_version": AS.SCHEMA_VERSION,
        "dataset": ds,
        "model": name,
        "fold": fold,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": {
            "outer_seed": 33,
            "calibration_seed": fold,
            "train_size": len(train_idx),
            "calibration_size": len(cal_idx),
            "test_size": len(test_idx),
        },
        "estimator_params": _jsonable(params),
        "resolved_params": _resolved_params(estimator),
        "environment": {
            key: value for key, value in sorted(os.environ.items()) if key.startswith("FERL_")
        },
        "source_revision": _source_revision(name),
        "project_revision": _git_revision(ROOT),
        "structure": structure,
        "rule_summary": _jsonable(_rule_summary(name, estimator)),
        "versions": _package_versions(),
        "model_archive": f"fold{fold}.joblib" if name in AS.ARCHIVED_MODELS else None,
    }


def _atomic_npz(path, **values):
    temporary = path + ".tmp.npz"
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _atomic_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _atomic_model(path, estimator):
    temporary = path + ".tmp"
    joblib.dump(estimator, temporary, compress=3)
    os.replace(temporary, path)


def run(datasets, model_names):
    failures = 0
    for ds in datasets:
        try:
            X, y = load_filtered(ds)
        except Exception as ex:
            failures += len(model_names)
            print(f"  {ds}: SKIP load ({ex})", flush=True); continue
        C = len(np.unique(y))
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=33)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            try:
                train_idx, cal_idx = train_test_split(
                    tr, test_size=CAL_FRAC, random_state=fold, stratify=y[tr])
            except ValueError:
                train_idx, cal_idx = train_test_split(
                    tr, test_size=CAL_FRAC, random_state=fold)
            Xtr, ytr = X[train_idx], y[train_idx]
            Xcal, ycal = X[cal_idx], y[cal_idx]
            Xte, yte = X[te], y[te]
            for name in model_names:
                p = artifact_path(ds, name, fold)
                if os.path.exists(p):
                    if AS.artifact_complete(p, ds, name, fold):
                        continue
                    print(f"    {ds}/{name}/f{fold}: retrying incomplete artifact", flush=True)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                try:
                    np.random.seed(fold)   # models drawing from NumPy's global RNG (e.g. FERL-medium) repeat
                    est = M.build(name)
                    t0 = time.perf_counter(); est.fit(Xtr, ytr); t_train = time.perf_counter() - t0
                    t0 = time.perf_counter(); pte = est.predict_proba(Xte); t_pred = time.perf_counter() - t0
                    pcal = est.predict_proba(Xcal)
                    complexity = M.complexity(name, est)
                    n_rules = float(getattr(est, "n_rules_", complexity))
                    conditions = float(getattr(est, "condition_complexity_", complexity))
                    avg_rule_length = conditions / n_rules if n_rules > 0 else 0.0
                    structure = {
                        "complexity": float(complexity),
                        "n_rules": n_rules,
                        "condition_complexity": conditions,
                        "avg_rule_length": avg_rule_length,
                    }
                    kw = dict(
                        schema_version=AS.SCHEMA_VERSION,
                        status="ok",
                        dataset=ds,
                        model=name,
                        fold=fold,
                        proba_test=pte,
                        proba_cal=pcal,
                        y_test=yte,
                        y_cal=ycal,
                        classes=M.classes_of(est),
                        C=C,
                        train_idx=np.asarray(train_idx, dtype=np.int64),
                        cal_idx=np.asarray(cal_idx, dtype=np.int64),
                        test_idx=np.asarray(te, dtype=np.int64),
                        train_s=t_train,
                        pred_s=t_pred,
                        complexity=complexity,
                        n_rules=n_rules,
                        condition_complexity=conditions,
                        avg_rule_length=avg_rule_length,
                    )
                    if name in M.SET_VALUED:                 # native set-valued output
                        kw["set_test"] = est.predict_set(Xte)
                    if hasattr(est, "predict_mass"):        # preserve native evidential output
                        kw["mass_test"] = est.predict_mass(Xte)
                        kw["mass_cal"] = est.predict_mass(Xcal)
                    if name in AS.ARCHIVED_MODELS:
                        _atomic_model(AS.model_path(p), est)
                    _atomic_json(
                        AS.metadata_path(p),
                        _manifest(ds, name, fold, est, train_idx, cal_idx, te, structure),
                    )
                    _atomic_npz(p, **kw)
                    if t_train > TIME_BUDGET:
                        print(f"    {ds}/{name}/f{fold}: SLOW {t_train:.0f}s", flush=True)
                except Exception as ex:
                    failures += 1
                    _atomic_npz(
                        p,
                        schema_version=AS.SCHEMA_VERSION,
                        status="dnf",
                        dataset=ds,
                        model=name,
                        fold=fold,
                        err=str(ex)[:500],
                        C=C,
                    )
                    print(f"    {ds}/{name}/f{fold} DNF: {str(ex)[:120]}", flush=True)
            print(f"  {ds} fold{fold} done", flush=True)
    return failures


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=SELECTED_30)
    ap.add_argument("--models", nargs="*", default=M.ALL_MODELS)
    ap.add_argument("--strict", action="store_true", help="exit nonzero if any fit or dataset load fails")
    ap.add_argument("--out-dir", default=OUT, help="artifact root (default results/bench)")
    a = ap.parse_args()
    OUT = a.out_dir
    print(f"Stage A: {len(a.datasets)} datasets x {a.models} x {N_FOLDS} folds -> {OUT}/")
    failures = run(a.datasets, a.models)
    if a.strict and failures:
        print(f"Stage A failed: {failures} fit/load failure(s)", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
