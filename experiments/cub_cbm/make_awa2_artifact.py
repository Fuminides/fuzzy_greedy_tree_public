"""Build a concept-bottleneck artifact bundle for Animals with Attributes 2
(AwA2) in the same format the CUB-CBM runners consume, so ``run_cub_cbm.py``
and ``run_cub_cbm_extras.py`` work on AwA2 unchanged (just ``--artifact-dir``).

Artifact format (identical to the CUB koh112 bundles):
    concept_names.json                       -- list[str], the 85 predicates
    oracle_{train,val,test}.npz              -- C (n,85) in [0,1], y, ids
    predicted_seed<k>_{train,val,test}.npz   -- detector scores (attached later)
    export_summary.json                      -- provenance

Protocol: this is a CBM *classification* study, not zero-shot. We use all 50
classes with a per-image stratified train/val/test split (default 60/20/20),
matching the CUB setup (all classes, per-image splits) rather than the AwA2
40/10 seen/unseen zero-shot split.

Oracle concepts are class-level: every image of a class gets that class's row
of the AwA2 predicate matrix (binary by default; ``--oracle continuous`` uses
the 0-100 relative-attribute matrix rescaled to [0,1]). This mirrors CUB's
class-level-majority oracle -- so, as with CUB oracle, the oracle regime is a
50-way lookup over the attribute signatures and is trivially separable; the
interesting regime is the detector-predicted one.

Predicted concepts come from an attribute detector trained OUTSIDE this repo
(an 85-way sigmoid head on an image backbone, exactly as CUB's predicted seeds
were produced elsewhere). Attach a detector's per-image scores with
``--predicted-scores <seed>:<file.npz>`` where the npz has ``ids`` (image ids,
matching this bundle) and ``scores`` (n,85) in [0,1]; rows are realigned to
each split by id.

Examples
--------
Real AwA2 (oracle only, ready for the predicted detector later):
    python3 experiments/cub_cbm/make_awa2_artifact.py \
        --awa2-root ../Animals_with_Attributes2 \
        --out-dir results/awa2_cbm_artifacts/awa2

Attach a trained detector's scores as seed 0:
    python3 experiments/cub_cbm/make_awa2_artifact.py \
        --awa2-root ../Animals_with_Attributes2 \
        --out-dir results/awa2_cbm_artifacts/awa2 \
        --predicted-scores 0:detector_seed0_scores.npz

Synthetic AwA2-shaped bundle to smoke-test the pipeline at scale:
    python3 experiments/cub_cbm/make_awa2_artifact.py --synthetic \
        --out-dir /tmp/awa2_synth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# --- reading standard AwA2 metadata ------------------------------------------

def _read_named_list(path: Path) -> list[str]:
    """Parse AwA2 'classes.txt' / 'predicates.txt': lines '<index>\\t<name>'."""
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        names.append(parts[1] if len(parts) > 1 else parts[0])
    return names


def _read_predicate_matrix(path: Path, n_classes: int, n_pred: int) -> np.ndarray:
    M = np.loadtxt(path, dtype=float)
    if M.shape != (n_classes, n_pred):
        raise ValueError(f"{path}: expected {(n_classes, n_pred)}, got {M.shape}")
    return M


def _enumerate_images(awa2_root: Path, class_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (image_ids, class_label) over JPEGImages/<class>/*.jpg. Image id is
    a stable index into the sorted global file listing."""
    jpeg_dir = awa2_root / "JPEGImages"
    if not jpeg_dir.is_dir():
        raise FileNotFoundError(
            f"{jpeg_dir} not found. Provide the AwA2 image tree, or use --synthetic "
            "to build a scale-test bundle without images."
        )
    ids, labels, _ = enumerate_awa2(awa2_root, class_names)
    return ids, labels


def enumerate_awa2(awa2_root: Path, class_names: list[str]):
    """(ids, labels, paths) over JPEGImages/<class>/*.jpg in a fixed order
    (classes.txt order, sorted files per class). Image id = index into this
    global listing; the detector trainer reuses this so its exported scores
    align to the oracle bundle's ids."""
    jpeg_dir = awa2_root / "JPEGImages"
    if not jpeg_dir.is_dir():
        raise FileNotFoundError(f"{jpeg_dir} not found.")
    labels, paths = [], []
    for label, cls in enumerate(class_names):
        files = sorted((jpeg_dir / cls).glob("*.jpg"))
        if not files:
            raise FileNotFoundError(f"no images for class '{cls}' under {jpeg_dir}")
        paths.extend(files)
        labels.extend([label] * len(files))
    ids = np.arange(len(paths), dtype=np.int64)
    return ids, np.asarray(labels, dtype=np.int64), paths


# --- splitting & writing ------------------------------------------------------

def stratified_split(y: np.ndarray, fracs: tuple[float, float, float], seed: int):
    """Per-class stratified train/val/test index arrays."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n = len(idx)
        n_tr = max(1, int(round(fracs[0] * n)))
        n_va = max(1, int(round(fracs[1] * n))) if n - n_tr >= 2 else 0
        tr.append(idx[:n_tr])
        va.append(idx[n_tr:n_tr + n_va])
        te.append(idx[n_tr + n_va:])
    return (np.concatenate(tr), np.concatenate(va), np.concatenate(te))


def _write_split(out_dir: Path, prefix: str, split: str, C: np.ndarray,
                 y: np.ndarray, ids: np.ndarray) -> None:
    np.savez(out_dir / f"{prefix}_{split}.npz",
             C=C.astype(np.float32), y=y.astype(np.int64), ids=ids.astype(np.int64))


def write_oracle(out_dir: Path, splits: dict, oracle_by_class: np.ndarray,
                 ids: np.ndarray, y: np.ndarray) -> dict:
    counts = {}
    for split, idx in splits.items():
        C = oracle_by_class[y[idx]]
        _write_split(out_dir, "oracle", split, C, y[idx], ids[idx])
        counts[split] = int(len(idx))
    return counts


def attach_predicted(out_dir: Path, seed: int, score_file: Path, splits: dict,
                     ids: np.ndarray, y: np.ndarray, n_concepts: int) -> None:
    """Realign a detector's per-image scores to each split by image id."""
    data = np.load(score_file, allow_pickle=False)
    score_ids = np.asarray(data["ids"], dtype=np.int64)
    scores = np.asarray(data["scores"], dtype=float)
    if scores.shape[1] != n_concepts:
        raise ValueError(f"{score_file}: scores have {scores.shape[1]} cols, expected {n_concepts}")
    pos = {int(i): r for r, i in enumerate(score_ids)}
    for split, idx in splits.items():
        rows = np.array([pos[int(ids[j])] for j in idx])
        C = np.clip(scores[rows], 0.0, 1.0)
        _write_split(out_dir, f"predicted_seed{seed}", split, C, y[idx], ids[idx])


# --- synthetic AwA2-shaped bundle (scale test, no images needed) --------------

def build_synthetic(out_dir: Path, seed: int, n_classes: int = 50,
                    n_concepts: int = 85, per_class: int = 120) -> tuple:
    rng = np.random.default_rng(seed)
    # distinct class-level attribute signatures (like the real predicate matrix)
    oracle_by_class = (rng.random((n_classes, n_concepts)) < 0.4).astype(float)
    for _ in range(100):
        if len({tuple(r) for r in oracle_by_class}) == n_classes:
            break
        oracle_by_class = (rng.random((n_classes, n_concepts)) < 0.4).astype(float)
    y = np.repeat(np.arange(n_classes), per_class).astype(np.int64)
    ids = np.arange(len(y), dtype=np.int64)
    concept_names = [f"attr_{j:02d}" for j in range(n_concepts)]
    # per-image predicted scores: class oracle bit pushed through a noisy sigmoid
    logits = 2.5 * (2 * oracle_by_class[y] - 1) + rng.normal(0, 1.5, (len(y), n_concepts))
    scores = 1.0 / (1.0 + np.exp(-logits))
    return oracle_by_class, y, ids, concept_names, scores


# --- CLI ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--awa2-root", type=Path, help="AwA2 dataset root (has classes.txt, "
                   "predicates.txt, predicate-matrix-*.txt, JPEGImages/)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--oracle", choices=["binary", "continuous"], default="binary")
    p.add_argument("--fracs", nargs=3, type=float, default=[0.6, 0.2, 0.2],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=0, help="split seed")
    p.add_argument("--predicted-scores", nargs="+", default=[],
                   metavar="SEED:FILE", help="attach detector scores, e.g. 0:scores0.npz")
    p.add_argument("--synthetic", action="store_true",
                   help="fabricate a 50x85 AwA2-shaped bundle (with predicted seed 0) "
                        "to scale-test the pipeline; ignores --awa2-root")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fracs = tuple(args.fracs)

    if args.synthetic:
        oracle_by_class, y, ids, concept_names, scores = build_synthetic(args.out_dir, args.seed)
        n_classes, n_concepts = oracle_by_class.shape
    else:
        if args.awa2_root is None:
            raise SystemExit("--awa2-root is required unless --synthetic is used")
        root = args.awa2_root
        class_names = _read_named_list(root / "classes.txt")
        concept_names = _read_named_list(root / "predicates.txt")
        n_classes, n_concepts = len(class_names), len(concept_names)
        mat = ("predicate-matrix-binary.txt" if args.oracle == "binary"
               else "predicate-matrix-continuous.txt")
        oracle_by_class = _read_predicate_matrix(root / mat, n_classes, n_concepts)
        if args.oracle == "continuous":
            oracle_by_class = oracle_by_class / 100.0
        oracle_by_class = np.clip(oracle_by_class, 0.0, 1.0)
        ids, y = _enumerate_images(root, class_names)
        scores = None

    (args.out_dir / "concept_names.json").write_text(
        json.dumps(concept_names, indent=2) + "\n", encoding="utf-8")
    tr, va, te = stratified_split(y, fracs, args.seed)
    splits = {"train": tr, "val": va, "test": te}
    counts = write_oracle(args.out_dir, splits, oracle_by_class, ids, y)

    detector_seeds = []
    if args.synthetic:
        np.savez(args.out_dir / "_synth_scores.npz", ids=ids, scores=scores)
        attach_predicted(args.out_dir, 0, args.out_dir / "_synth_scores.npz",
                         splits, ids, y, n_concepts)
        (args.out_dir / "_synth_scores.npz").unlink()
        detector_seeds = [0]
    for spec in args.predicted_scores:
        seed_s, _, file_s = spec.partition(":")
        attach_predicted(args.out_dir, int(seed_s), Path(file_s), splits, ids, y, n_concepts)
        detector_seeds.append(int(seed_s))

    summary = dict(
        artifact_dir=str(args.out_dir), dataset="AwA2",
        synthetic=bool(args.synthetic), oracle_source=args.oracle,
        num_classes=int(n_classes), num_concepts=int(n_concepts),
        splits=counts, split_fracs=list(fracs), split_seed=args.seed,
        detector_seeds=sorted(set(detector_seeds)),
        concept_names=str(args.out_dir / "concept_names.json"),
    )
    (args.out_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote AwA2 CBM artifact to {args.out_dir}: {n_classes} classes, "
          f"{n_concepts} concepts, splits {counts}, detector_seeds {sorted(set(detector_seeds))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
