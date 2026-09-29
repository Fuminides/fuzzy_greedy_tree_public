"""Validate benchmark artifacts before final scoring or paper aggregation."""
from __future__ import annotations

import argparse
import os

from ferl.pipeline.run_configs import SELECTED_30, load_filtered
import artifact_schema as AS


COMPARISON_MODELS = ["SamRuLe-OVR", "RRL", "RL-Net", "FuzzyUCS-DS", "NeuRules"]


def validate(root, datasets, models, n_folds=5, check_models=False):
    errors = []
    for dataset in datasets:
        X = y = None
        if check_models:
            try:
                X, y = load_filtered(dataset)
            except Exception as ex:
                errors.append(f"{dataset}: cannot load data for archive replay ({ex})")
        for model in models:
            for fold in range(n_folds):
                label = f"{dataset}/{model}/fold{fold}"
                path = os.path.join(root, dataset, model, f"fold{fold}.npz")
                artifact_failures = AS.artifact_errors(
                    path, dataset, model, fold, load_archive=False
                )
                errors.extend(
                    f"{label}: {error}"
                    for error in artifact_failures
                )
                if (
                    not artifact_failures
                    and check_models
                    and X is not None
                    and model in AS.ARCHIVED_MODELS
                ):
                    errors.extend(
                        f"{label}: {error}"
                        for error in AS.archived_prediction_errors(path, X, y)
                    )
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.path.join("results", "bench"))
    parser.add_argument("--datasets", nargs="*", default=SELECTED_30)
    parser.add_argument("--models", nargs="*", default=COMPARISON_MODELS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="check archive presence but do not deserialize fitted estimators",
    )
    args = parser.parse_args()

    errors = validate(
        args.root,
        args.datasets,
        args.models,
        args.folds,
        check_models=not args.skip_model_load,
    )
    if errors:
        print(f"Comparison artifact validation failed with {len(errors)} error(s):")
        for error in errors:
            print(f"  {error}")
        raise SystemExit(1)

    expected = len(args.datasets) * len(args.models) * args.folds
    print(f"Comparison artifacts valid: {expected} folds under {args.root}")


if __name__ == "__main__":
    main()
