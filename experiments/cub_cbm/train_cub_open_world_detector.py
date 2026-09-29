"""Train detector-class-disjoint CUB concept artifacts for E16.

This adapter uses the official CUB X->C implementation in the sibling
``symbol_sanity`` checkout.  For each class fold it filters the detector's
training manifest, trains without any held-out-species image, then scores the
unchanged full train/validation/test manifests.  The resulting NPZ files remain
aligned with the ordinary CUB artifact bundle while their provenance records
which classes the image detector never saw.

Run from the FERL repository root.  This is a GPU/HPC stage; the FERL/LR E16
evaluation is a separate CPU stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cub_cbm.make_awa2_artifact import _write_split
from experiments.cub_cbm.open_world import make_class_folds, write_provenance


def _read_manifest(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schema = json.loads((directory / "schema.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (directory / "metadata.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    return schema, rows


def _write_filtered_manifest(
    directory: Path,
    *,
    schema: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "metadata.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rebase_image_paths(
    rows: list[dict[str, Any]],
    *,
    cub_root: Path,
) -> list[dict[str, Any]]:
    """Replace machine-specific manifest paths with this machine's CUB root."""
    rebased = []
    for row in rows:
        updated = dict(row)
        updated["image_path"] = str(
            cub_root / "images" / str(row["class_name"]) / Path(row["image_path"]).name
        )
        rebased.append(updated)
    return rebased


def _class_names(schema: dict[str, Any]) -> dict[int, str]:
    names: dict[int, str] = {}
    for original, compact in schema.get("class_to_label", {}).items():
        names[int(compact)] = str(schema.get("classes", {}).get(str(original), original))
    return names


def _score_and_write(
    *,
    manifest_dir: Path,
    artifact_dir: Path,
    prefix: str,
    checkpoint: Path,
    task: str,
    batch_size: int,
    device: str,
) -> None:
    from symbol_sanity.uncertainty import collect_detector_probabilities

    for split in ("train", "val", "test"):
        split_dir = manifest_dir / split
        _, rows = _read_manifest(split_dir)
        probabilities = collect_detector_probabilities(
            dataset_dir=split_dir,
            detector_path=checkpoint,
            batch_size=batch_size,
            device=device,
        )
        scores = np.asarray(
            probabilities.cpu().numpy()
            if hasattr(probabilities, "cpu")
            else probabilities,
            dtype=float,
        )
        labels = np.asarray(
            [int(row["task_labels"][task]) for row in rows],
            dtype=np.int64,
        )
        ids = np.asarray(
            [int(row.get("image_id", index)) for index, row in enumerate(rows)],
            dtype=np.int64,
        )
        _write_split(artifact_dir, prefix, split, scores, labels, ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cub-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--symbol-sanity-root",
        type=Path,
        default=REPO_ROOT.parent / "symbol_sanity",
    )
    parser.add_argument("--task", default="species")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--fold-seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=299)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="retrain even when the fold/seed checkpoint already exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbol_src = args.symbol_sanity_root.resolve() / "src"
    if not symbol_src.is_dir():
        raise SystemExit(f"symbol_sanity source directory not found: {symbol_src}")
    sys.path.insert(0, str(symbol_src))
    from symbol_sanity.neural_synthetic import train_official_synthetic_detector

    if not (args.cub_root / "images").is_dir():
        raise SystemExit(f"CUB image directory not found: {args.cub_root / 'images'}")
    full_manifests = {
        split: _read_manifest(args.manifest_dir / split)
        for split in ("train", "val", "test")
    }
    train_schema, raw_train_rows = full_manifests["train"]
    train_rows = _rebase_image_paths(raw_train_rows, cub_root=args.cub_root)
    if args.task not in train_schema.get("tasks", {}):
        raise ValueError(f"task {args.task!r} is absent from the CUB manifest")
    concept_names = list(train_schema["concept_names"])
    bundle_concepts = json.loads(
        (args.artifact_dir / "concept_names.json").read_text(encoding="utf-8")
    )
    if bundle_concepts != concept_names:
        raise ValueError("CUB manifest and artifact concept names differ")

    labels = np.asarray(
        [int(row["task_labels"][args.task]) for row in train_rows],
        dtype=int,
    )
    classes = np.unique(labels)
    folds = make_class_folds(classes, n_folds=args.n_folds, seed=args.fold_seed)
    selected_folds = args.folds if args.folds is not None else list(range(len(folds)))
    class_names = _class_names(train_schema)
    observed = set(int(label) for label in classes)
    checkpoint_dir = args.artifact_dir / "open_world_detectors"
    manifest_root = args.artifact_dir / "open_world_manifests"

    for fold in selected_folds:
        if fold < 0 or fold >= len(folds):
            raise ValueError(f"open-world fold {fold} is outside [0, {len(folds)})")
        heldout = folds[fold]
        trained = tuple(sorted(observed - set(heldout)))
        retained_rows = [
            row
            for row in train_rows
            if int(row["task_labels"][args.task]) not in set(heldout)
        ]
        for seed in args.seeds:
            prefix = f"openworld_fold{fold}_seed{seed}"
            filtered_train = manifest_root / prefix / "train"
            _write_filtered_manifest(
                filtered_train,
                schema=train_schema,
                rows=retained_rows,
            )
            rebased_full_manifest = manifest_root / prefix / "full"
            for split, (split_schema, split_rows) in full_manifests.items():
                _write_filtered_manifest(
                    rebased_full_manifest / split,
                    schema=split_schema,
                    rows=_rebase_image_paths(split_rows, cub_root=args.cub_root),
                )
            checkpoint = checkpoint_dir / f"{prefix}.pt"
            if args.force or not checkpoint.exists():
                print(
                    f"[cub:openworld] train fold={fold} seed={seed} "
                    f"heldout={list(heldout)} images={len(retained_rows)}",
                    flush=True,
                )
                train_official_synthetic_detector(
                    dataset_dir=filtered_train,
                    output_path=checkpoint,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    seed=seed,
                    device=args.device,
                    image_size=args.image_size,
                    pretrained=not args.no_pretrained,
                    freeze=args.freeze,
                )
            else:
                print(f"[cub:openworld] reuse {checkpoint}", flush=True)
            _score_and_write(
                manifest_dir=rebased_full_manifest,
                artifact_dir=args.artifact_dir,
                prefix=prefix,
                checkpoint=checkpoint,
                task=args.task,
                batch_size=args.batch_size,
                device=args.device,
            )
            write_provenance(
                args.artifact_dir,
                prefix=prefix,
                dataset="cub",
                fold=fold,
                detector_seed=seed,
                heldout_classes=heldout,
                detector_train_classes=trained,
                class_names=class_names,
                extra={
                    "fold_seed": args.fold_seed,
                    "n_folds": args.n_folds,
                    "detector_train_images": len(retained_rows),
                    "detector_checkpoint": str(checkpoint),
                    "detector_epochs": args.epochs,
                    "pretrained_imagenet": not args.no_pretrained,
                    "source_manifest": str(args.manifest_dir),
                    "runtime_cub_root": str(args.cub_root),
                },
            )
            print(f"[cub:openworld] wrote {prefix}_*.npz", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
