"""Train an AwA2 attribute detector and export predicted concepts in the
CBM artifact format, so ``run_cub_cbm.py`` / ``run_cub_cbm_extras.py`` get a
detector-predicted regime for AwA2 (the analogue of CUB's ``predicted_seed<k>``).

The detector is a torchvision image backbone with an 85-way sigmoid head,
trained with BCE against AwA2's class-level binary attributes (every image of a
class shares that class's attribute vector, the standard AwA2 concept
supervision). Each ``--seeds`` value trains an independent detector and writes
``predicted_seed<k>_{train,val,test}.npz`` into the oracle bundle's directory,
with per-image sigmoid scores aligned by image id to the oracle splits.

This is the GPU part of the AwA2 pipeline; the oracle bundle must already exist
(``make_awa2_artifact.py``). Intended for the cluster (see awa2_cluster.sh).

Example (one seed, on a cluster with cached ImageNet weights):
    python3 experiments/cub_cbm/train_awa2_detector.py \
        --awa2-root ~/Datasets/Animals_with_Attributes2 \
        --artifact-dir results/awa2_cbm_artifacts/awa2 \
        --seeds 0 1 2 --epochs 15 --batch-size 128
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.cub_cbm.make_awa2_artifact import (
    _read_named_list,
    _read_predicate_matrix,
    _write_split,
    enumerate_awa2,
)
from experiments.cub_cbm.open_world import make_class_folds, write_provenance


def build_targets(awa2_root: Path):
    """(ids, labels, paths, concept_names, per_image_targets[N,85])."""
    class_names = _read_named_list(awa2_root / "classes.txt")
    concept_names = _read_named_list(awa2_root / "predicates.txt")
    A = _read_predicate_matrix(awa2_root / "predicate-matrix-binary.txt",
                               len(class_names), len(concept_names))
    A = (A >= 0.5).astype(np.float32)
    ids, labels, paths = enumerate_awa2(awa2_root, class_names)
    return ids, labels, paths, concept_names, A[labels]


def split_ids(artifact_dir: Path) -> dict:
    out = {}
    for split in ("train", "val", "test"):
        d = np.load(artifact_dir / f"oracle_{split}.npz", allow_pickle=False)
        out[split] = np.asarray(d["ids"], dtype=np.int64)
    return out


def _make_dataset(paths, targets, img_size, train):
    import torch
    from PIL import Image
    from torchvision import transforms

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if train:
        tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        tf = transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(), transforms.Normalize(mean, std)])

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(paths)

        def __getitem__(self, i):
            img = Image.open(paths[i]).convert("RGB")
            return tf(img), torch.from_numpy(targets[i])

    return DS()


def _build_model(arch: str, n_concepts: int, pretrained: bool):
    import torch.nn as nn
    import torchvision.models as M

    factory = getattr(M, arch)
    weights = "DEFAULT" if pretrained else None
    model = factory(weights=weights)
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, n_concepts)
    elif hasattr(model, "classifier"):
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, n_concepts)
    else:
        raise ValueError(f"don't know how to reshape head of {arch}")
    return model


def train_one_seed(seed, *, paths, targets, labels, splits, concept_names,
                   artifact_dir, arch, epochs, batch_size, lr, img_size,
                   num_workers, pretrained, device, limit_per_class,
                   heldout_classes=(), output_prefix=None):
    import torch
    from torch.utils.data import DataLoader, Subset

    torch.manual_seed(seed)
    np.random.seed(seed)
    n_concepts = targets.shape[1]

    heldout_classes = {int(label) for label in heldout_classes}
    tr_ids = splits["train"]
    if heldout_classes:
        tr_ids = tr_ids[~np.isin(labels[tr_ids], sorted(heldout_classes))]
    if not len(tr_ids):
        raise ValueError("detector training split is empty after class exclusion")
    if limit_per_class:  # smoke: cap images per class
        keep, seen = [], {}
        for i in tr_ids:
            c = int(labels[i])
            if seen.get(c, 0) < limit_per_class:
                keep.append(i); seen[c] = seen.get(c, 0) + 1
        tr_ids = np.array(keep, dtype=np.int64)

    full_train = _make_dataset(paths, targets, img_size, train=True)
    full_eval = _make_dataset(paths, targets, img_size, train=False)
    train_loader = DataLoader(Subset(full_train, tr_ids.tolist()), batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    model = _build_model(arch, n_concepts, pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    lossf = torch.nn.BCEWithLogitsLoss()
    # mixed precision roughly halves activation memory (fits ResNet50 on an
    # ~11GB GPU at batch 64) and speeds training; a no-op on CPU.
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    for ep in range(epochs):
        t0, tot = time.time(), 0.0
        for x, t in train_loader:
            x, t = x.to(device, non_blocking=True), t.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = lossf(model(x), t)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += float(loss) * len(x)
        sched.step()
        print(f"[awa2:det seed={seed}] epoch {ep+1}/{epochs} "
              f"bce={tot/max(1,len(tr_ids)):.4f} ({time.time()-t0:.0f}s)", flush=True)

    # score every image, deterministically
    model.eval()
    scores = np.zeros((len(paths), n_concepts), dtype=np.float32)
    eval_loader = DataLoader(full_eval, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        pos = 0
        for x, _ in eval_loader:
            p = torch.sigmoid(model(x.to(device, non_blocking=True))).float().cpu().numpy()
            scores[pos:pos + len(p)] = p
            pos += len(p)

    prefix = output_prefix or f"predicted_seed{seed}"
    for split, sids in splits.items():
        _write_split(artifact_dir, prefix, split,
                     scores[sids], labels[sids], sids)
    print(
        f"[awa2:det seed={seed}] wrote {prefix}_*.npz "
        f"(excluded classes={sorted(heldout_classes)})",
        flush=True,
    )
    return int(len(tr_ids))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--awa2-root", type=Path, required=True)
    p.add_argument("--artifact-dir", type=Path, required=True,
                   help="oracle bundle dir; predicted_seed<k>_*.npz written here")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--arch", default="resnet50")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)  # fits ResNet50 on ~11GB w/ AMP
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-pretrained", action="store_true",
                   help="random init (default uses cached ImageNet weights)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit-per-class", type=int, default=0,
                   help="cap train images/class for a fast smoke run (0 = all)")
    p.add_argument(
        "--open-world-folds",
        type=int,
        default=0,
        help=(
            "train class-disjoint open-world detectors over this many folds; "
            "writes openworld_fold<f>_seed<s> artifacts instead of ordinary seeds"
        ),
    )
    p.add_argument(
        "--folds",
        nargs="+",
        type=int,
        help="optional open-world fold indices to train (default: every fold)",
    )
    p.add_argument(
        "--fold-seed",
        type=int,
        default=0,
        help="deterministic class-fold partition seed",
    )
    return p.parse_args()


def main() -> int:
    import torch

    args = parse_args()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    torch.backends.cudnn.benchmark = True
    ids, labels, paths, concept_names, targets = build_targets(args.awa2_root)
    splits = split_ids(args.artifact_dir)
    # sanity: the oracle bundle and the detector head must share a concept space
    bundle_concepts = json.loads((args.artifact_dir / "concept_names.json").read_text())
    if len(bundle_concepts) != targets.shape[1]:
        raise SystemExit(f"concept mismatch: bundle has {len(bundle_concepts)}, "
                         f"detector head {targets.shape[1]}")
    print(
        f"AwA2 detector: {len(paths)} images, {targets.shape[1]} concepts, "
        f"train/val/test {[len(splits[s]) for s in ('train','val','test')]}, "
        f"device={device}",
        flush=True,
    )
    if args.open_world_folds:
        folds = make_class_folds(
            labels,
            n_folds=args.open_world_folds,
            seed=args.fold_seed,
        )
        selected_folds = (
            args.folds if args.folds is not None else list(range(len(folds)))
        )
        class_names = _read_named_list(args.awa2_root / "classes.txt")
        class_name_map = {index: name for index, name in enumerate(class_names)}
        observed = set(int(label) for label in np.unique(labels))
        for fold in selected_folds:
            if fold < 0 or fold >= len(folds):
                raise ValueError(f"open-world fold {fold} is outside [0, {len(folds)})")
            heldout = folds[fold]
            trained = tuple(sorted(observed - set(heldout)))
            for seed in args.seeds:
                prefix = f"openworld_fold{fold}_seed{seed}"
                n_train = train_one_seed(
                    seed,
                    paths=paths,
                    targets=targets,
                    labels=labels,
                    splits=splits,
                    concept_names=concept_names,
                    artifact_dir=args.artifact_dir,
                    arch=args.arch,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    img_size=args.img_size,
                    num_workers=args.num_workers,
                    pretrained=not args.no_pretrained,
                    device=device,
                    limit_per_class=args.limit_per_class,
                    heldout_classes=heldout,
                    output_prefix=prefix,
                )
                write_provenance(
                    args.artifact_dir,
                    prefix=prefix,
                    dataset="awa2",
                    fold=fold,
                    detector_seed=seed,
                    heldout_classes=heldout,
                    detector_train_classes=trained,
                    class_names=class_name_map,
                    extra={
                        "fold_seed": args.fold_seed,
                        "n_folds": args.open_world_folds,
                        "detector_train_images": n_train,
                        "detector_architecture": args.arch,
                        "detector_epochs": args.epochs,
                        "pretrained_imagenet": not args.no_pretrained,
                    },
                )
        print(
            f"done: open-world folds {selected_folds}, seeds {args.seeds} "
            f"in {args.artifact_dir}"
        )
    else:
        for seed in args.seeds:
            train_one_seed(
                seed,
                paths=paths,
                targets=targets,
                labels=labels,
                splits=splits,
                concept_names=concept_names,
                artifact_dir=args.artifact_dir,
                arch=args.arch,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                img_size=args.img_size,
                num_workers=args.num_workers,
                pretrained=not args.no_pretrained,
                device=device,
                limit_per_class=args.limit_per_class,
            )
        print(f"done: predicted seeds {args.seeds} in {args.artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
