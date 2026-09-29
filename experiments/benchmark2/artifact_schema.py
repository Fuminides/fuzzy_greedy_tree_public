"""Versioned benchmark artifact paths and structural validation."""
from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np


SCHEMA_VERSION = 2
ARCHIVED_MODELS = {
    "SampledRuleList",
    "SamRuLe",
    "SamRuLe-OVR",
    "RRL",
    "RL-Net",
    "FuzzyUCS-DS",
    "NeuRules",
}
EVIDENTIAL_MODELS = {"FuzzyUCS-DS"}
REPOSITORY_ENV = {
    "SamRuLe": "FERL_SAMRULE_REPO",
    "SamRuLe-OVR": "FERL_SAMRULE_REPO",
    "RRL": "FERL_RRL_REPO",
    "RL-Net": "FERL_RLNET_REPO",
}
REPOSITORY_DEFAULT = {
    "SamRuLe": "external/SamRuLe",
    "SamRuLe-OVR": "external/SamRuLe",
    "RRL": "external/rrl",
    "RL-Net": "external/RLNet",
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_ARRAY_FIELDS = {
    "schema_version",
    "status",
    "dataset",
    "model",
    "fold",
    "proba_test",
    "proba_cal",
    "y_test",
    "y_cal",
    "classes",
    "C",
    "train_idx",
    "cal_idx",
    "test_idx",
    "train_s",
    "pred_s",
    "complexity",
    "n_rules",
    "condition_complexity",
    "avg_rule_length",
}


def metadata_path(npz_path):
    return os.path.splitext(npz_path)[0] + ".json"


def model_path(npz_path):
    return os.path.splitext(npz_path)[0] + ".joblib"


@contextmanager
def _import_path(path):
    if path is None:
        yield
        return
    path = str(Path(path).expanduser().resolve())
    old = list(sys.path)
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = old


def load_estimator(npz_path, repository=None):
    """Load a fitted estimator, adding its recorded public repository if needed."""
    manifest = metadata_path(npz_path)
    with open(manifest, encoding="utf-8") as stream:
        metadata = json.load(stream)
    model = metadata["model"]
    environment_name = REPOSITORY_ENV.get(model)
    if repository is None and environment_name is not None:
        repository = metadata.get("environment", {}).get(environment_name)
        if repository is None or not Path(repository).expanduser().exists():
            current = os.environ.get(environment_name)
            default = PROJECT_ROOT / REPOSITORY_DEFAULT[model]
            repository = current if current and Path(current).expanduser().exists() else default
    with _import_path(repository):
        return joblib.load(model_path(npz_path))


def archived_prediction_errors(npz_path, X, y=None):
    """Replay a fitted archive on its saved test indices and compare outputs."""
    try:
        with np.load(npz_path, allow_pickle=False) as artifact:
            indices = artifact["test_idx"]
            expected = artifact["proba_test"]
            train_idx = artifact["train_idx"]
            cal_idx = artifact["cal_idx"]
            saved_y_test = artifact["y_test"]
            saved_y_cal = artifact["y_cal"]
        complete_indices = np.sort(np.concatenate((train_idx, cal_idx, indices)))
        if not np.array_equal(complete_indices, np.arange(len(X))):
            return ["saved split indices do not partition the complete dataset"]
        if y is not None:
            y = np.asarray(y)
            if not np.array_equal(saved_y_test, y[indices]):
                return ["y_test is inconsistent with test_idx"]
            if not np.array_equal(saved_y_cal, y[cal_idx]):
                return ["y_cal is inconsistent with cal_idx"]
        estimator = load_estimator(npz_path)
        actual = np.asarray(estimator.predict_proba(np.asarray(X)[indices]), dtype=float)
        if actual.shape != expected.shape:
            return [f"archived prediction shape {actual.shape}; expected {expected.shape}"]
        if not np.allclose(actual, expected, rtol=1e-7, atol=1e-9):
            delta = float(np.max(np.abs(actual - expected)))
            return [f"archived predictions differ (max delta {delta:.3g})"]
        return []
    except Exception as ex:
        return [f"archived prediction replay failed ({ex})"]


def _scalar(artifact, key):
    value = np.asarray(artifact[key])
    if value.size != 1:
        raise ValueError(f"{key} is not scalar")
    return value.reshape(-1)[0]


def _validate_indices(artifact, errors):
    arrays = {}
    for key in ("train_idx", "cal_idx", "test_idx"):
        values = np.asarray(artifact[key])
        if values.ndim != 1 or values.dtype.kind not in "iu":
            errors.append(f"{key} must be a one-dimensional integer array")
            continue
        if len(values) == 0 or len(np.unique(values)) != len(values):
            errors.append(f"{key} must be non-empty and unique")
        arrays[key] = values
    if len(arrays) != 3:
        return
    if len(arrays["cal_idx"]) != len(artifact["y_cal"]):
        errors.append("cal_idx length does not match y_cal")
    if len(arrays["test_idx"]) != len(artifact["y_test"]):
        errors.append("test_idx length does not match y_test")
    for left, right in (
        ("train_idx", "cal_idx"),
        ("train_idx", "test_idx"),
        ("cal_idx", "test_idx"),
    ):
        if np.intersect1d(arrays[left], arrays[right]).size:
            errors.append(f"{left} and {right} overlap")


def _validate_probabilities(artifact, errors):
    classes = np.asarray(artifact["classes"])
    expected_classes = int(_scalar(artifact, "C"))
    if classes.ndim != 1 or len(np.unique(classes)) != len(classes):
        errors.append("classes must be a one-dimensional unique array")
        return
    if len(classes) != expected_classes:
        errors.append(f"classes={len(classes)} but C={expected_classes}")
    for split in ("test", "cal"):
        proba = np.asarray(artifact[f"proba_{split}"])
        target = np.asarray(artifact[f"y_{split}"])
        if target.ndim != 1:
            errors.append(f"y_{split} must be one-dimensional")
        if proba.ndim != 2 or proba.shape != (len(target), len(classes)):
            errors.append(
                f"malformed {split} shape {proba.shape}; "
                f"expected ({len(target)}, {len(classes)})"
            )
            continue
        if not np.isfinite(proba).all():
            errors.append(f"non-finite {split} probabilities")
        elif (proba < 0).any() or not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
            errors.append(f"invalid or unnormalized {split} probabilities")
        if not np.isin(target, classes).all():
            errors.append(f"y_{split} contains labels absent from classes")


def _validate_evidence(artifact, errors):
    n_classes = len(artifact["classes"])
    for split in ("test", "cal"):
        key = f"mass_{split}"
        if key not in artifact.files:
            errors.append(f"missing evidential field {key}")
            continue
        mass = np.asarray(artifact[key])
        expected = (len(artifact[f"y_{split}"]), n_classes + 1)
        if mass.shape != expected:
            errors.append(f"malformed {key} shape {mass.shape}; expected {expected}")
        elif (
            not np.isfinite(mass).all()
            or (mass < -1e-12).any()
            or not np.allclose(mass.sum(axis=1), 1.0, atol=1e-6)
        ):
            errors.append(f"invalid or unnormalized {key}")
        else:
            pignistic = mass[:, :n_classes] + mass[:, [-1]] / n_classes
            if not np.allclose(pignistic, artifact[f"proba_{split}"], atol=1e-6):
                errors.append(f"{key} is inconsistent with proba_{split}")
    if "set_test" not in artifact.files:
        errors.append("missing evidential field set_test")
    else:
        prediction_set = np.asarray(artifact["set_test"])
        expected = (len(artifact["y_test"]), n_classes)
        if prediction_set.shape != expected:
            errors.append(f"malformed set_test shape {prediction_set.shape}; expected {expected}")
        elif prediction_set.dtype != np.bool_:
            errors.append("set_test must be boolean")
        elif not prediction_set.any(axis=1).all():
            errors.append("set_test contains an empty prediction set")
        elif "mass_test" in artifact.files:
            mass = np.asarray(artifact["mass_test"])
            if mass.shape == (len(artifact["y_test"]), n_classes + 1):
                belief = mass[:, :n_classes]
                plausibility = belief + mass[:, [-1]]
                expected_set = plausibility >= belief.max(axis=1, keepdims=True) - 1e-12
                if not np.array_equal(prediction_set, expected_set):
                    errors.append("set_test is inconsistent with mass_test")


def artifact_errors(
    path, dataset=None, model=None, fold=None, require_sidecars=True, load_archive=False
):
    """Return structural errors for one successful artifact."""
    errors = []
    if not os.path.isfile(path):
        return ["missing"]
    try:
        with np.load(path, allow_pickle=False) as artifact:
            if "status" not in artifact.files:
                return ["missing status"]
            if str(_scalar(artifact, "status")) != "ok":
                return [f"status={str(_scalar(artifact, 'status'))}"]
            missing = REQUIRED_ARRAY_FIELDS.difference(artifact.files)
            if missing:
                return [f"missing fields {sorted(missing)}"]
            if int(_scalar(artifact, "schema_version")) != SCHEMA_VERSION:
                errors.append(
                    f"schema={int(_scalar(artifact, 'schema_version'))}; "
                    f"expected {SCHEMA_VERSION}"
                )
            stored_dataset = str(_scalar(artifact, "dataset"))
            stored_model = str(_scalar(artifact, "model"))
            stored_fold = int(_scalar(artifact, "fold"))
            if dataset is not None and stored_dataset != dataset:
                errors.append(f"dataset={stored_dataset}; expected {dataset}")
            if model is not None and stored_model != model:
                errors.append(f"model={stored_model}; expected {model}")
            if fold is not None and stored_fold != fold:
                errors.append(f"fold={stored_fold}; expected {fold}")
            for key in (
                "train_s",
                "pred_s",
                "complexity",
                "n_rules",
                "condition_complexity",
                "avg_rule_length",
            ):
                value = float(_scalar(artifact, key))
                if not np.isfinite(value) or value < 0:
                    errors.append(f"{key} must be finite and non-negative")
            _validate_probabilities(artifact, errors)
            _validate_indices(artifact, errors)
            if stored_model in EVIDENTIAL_MODELS:
                _validate_evidence(artifact, errors)
    except Exception as ex:
        return [f"unreadable ({ex})"]

    target_model = model or stored_model
    if require_sidecars and target_model in ARCHIVED_MODELS:
        manifest = metadata_path(path)
        archive = model_path(path)
        if not os.path.isfile(manifest):
            errors.append("missing metadata manifest")
        else:
            try:
                with open(manifest, encoding="utf-8") as stream:
                    metadata = json.load(stream)
                for key in (
                    "schema_version",
                    "dataset",
                    "model",
                    "fold",
                    "split",
                    "estimator_params",
                    "resolved_params",
                    "environment",
                    "source_revision",
                    "project_revision",
                    "structure",
                    "versions",
                ):
                    if key not in metadata:
                        errors.append(f"manifest missing {key}")
                if metadata.get("schema_version") != SCHEMA_VERSION:
                    errors.append("manifest schema mismatch")
                if dataset is not None and metadata.get("dataset") != dataset:
                    errors.append("manifest dataset mismatch")
                if model is not None and metadata.get("model") != model:
                    errors.append("manifest model mismatch")
                if fold is not None and metadata.get("fold") != fold:
                    errors.append("manifest fold mismatch")
            except Exception as ex:
                errors.append(f"unreadable metadata manifest ({ex})")
        if not os.path.isfile(archive) or os.path.getsize(archive) == 0:
            errors.append("missing fitted-estimator archive")
        elif load_archive:
            try:
                estimator = load_estimator(path)
                if not hasattr(estimator, "classes_"):
                    errors.append("fitted-estimator archive has no classes_")
            except Exception as ex:
                errors.append(f"unreadable fitted-estimator archive ({ex})")
    return errors


def artifact_complete(path, dataset, model, fold):
    return not artifact_errors(
        path,
        dataset,
        model,
        fold,
        load_archive=model in ARCHIVED_MODELS,
    )
