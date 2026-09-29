"""
Stage B (metrics). Offline pass over the artifacts saved by harness.py — no
refitting. Computes accuracy / complexity / speed / selective (AURC, acc@cov) /
conformal-APS (coverage, size) / credal utilities (determinacy, set-acc, u65/u80)
at a chosen alpha, aggregates over datasets x folds, writes a tidy CSV.

Usage:  python experiments/benchmark2/score.py [--alpha 0.1]
"""
import os
import glob
import argparse
import warnings
import re
import numpy as np
import pandas as pd
import metrics as MET

warnings.filterwarnings("ignore")
OUT = os.path.join("results", "bench")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--bench-dir", default=OUT, help="artifact root written by harness.py")
    ap.add_argument("--per-fold-csv", default="results/benchmark2_per_fold.csv")
    ap.add_argument("--summary-csv", default="results/benchmark2_summary.csv")
    ap.add_argument("--replace-models", nargs="*", default=None,
                    help="only score these models' artifacts and merge them into the existing "
                         "per-fold CSV: rows with the same (dataset, model, fold) are replaced, "
                         "all other rows are kept")
    a = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(a.bench_dir, "*", "*", "fold*.npz"))):
        if a.replace_models is not None and path.split(os.sep)[-2] not in a.replace_models:
            continue
        ds, model = path.split(os.sep)[-3], path.split(os.sep)[-2]
        match = re.fullmatch(r"fold(\d+)\.npz", os.path.basename(path))
        if match is None:
            continue
        fold = int(match.group(1))
        d = np.load(path, allow_pickle=False)
        if str(d["status"]) != "ok":
            rows.append(dict(dataset=ds, model=model, fold=fold, status="dnf")); continue
        cls = d["classes"]; yte = d["y_test"]
        pte, pcal, ycal = d["proba_test"], d["proba_cal"], d["y_cal"]
        if pte.shape[1] != len(cls) or pcal.shape[1] != len(cls):
            rows.append(dict(dataset=ds, model=model, fold=fold, status="malformed")); continue
        conf = MET.credal_utils(MET.aps_sets(pcal, ycal, pte, cls, alpha=a.alpha), yte, cls)
        raps = MET.credal_utils(MET.raps_sets(pcal, ycal, pte, cls, alpha=a.alpha), yte, cls)
        mond = MET.credal_utils(MET.mondrian_sets(pcal, ycal, pte, cls, alpha=a.alpha), yte, cls)
        if "set_test" in d.files:                      # native set-valued output
            nat = MET.credal_utils(d["set_test"], yte, cls); src = "native"
        else:
            nat = conf; src = "conformal"
        # Stage-B calibration transforms over the saved cal-fold probabilities.
        try:
            ece_va = MET.ece(MET.venn_abers_cal(pcal, ycal, pte, cls), yte, cls)
        except Exception:
            ece_va = np.nan
        if "mass_test" in d.files:
            evidence = MET.evidential_metrics(d["mass_test"], yte, cls)
        else:
            evidence = dict(
                ignorance=np.nan,
                true_belief=np.nan,
                true_plausibility=np.nan,
                evidence_aurc=np.nan,
            )
        complexity = float(d["complexity"])
        n_rules = float(d["n_rules"]) if "n_rules" in d.files else complexity
        conditions = (
            float(d["condition_complexity"])
            if "condition_complexity" in d.files else complexity
        )
        avg_rule_length = (
            float(d["avg_rule_length"])
            if "avg_rule_length" in d.files
            else conditions / n_rules if n_rules > 0 else 0.0
        )
        rows.append(dict(
            dataset=ds, model=model, fold=fold, status="ok", C=int(d["C"]), set_src=src,
            acc=MET.accuracy(pte, yte, cls),
            complexity=complexity, n_rules=n_rules,
            condition_complexity=conditions, avg_rule_length=avg_rule_length,
            train_s=float(d["train_s"]), pred_s=float(d["pred_s"]),
            aurc=MET.aurc(pte, yte, cls),
            acc90=MET.acc_at_coverage(pte, yte, cls, 0.90),
            ece_raw=MET.ece(pte, yte, cls),
            ece_iso=MET.ece(MET.isotonic_cal(pcal, ycal, pte, cls), yte, cls),
            ece_va=ece_va,
            conf_determinacy=conf["determinacy"],
            conf_cov=conf["coverage"], conf_size=conf["mean_size"],
            conf_u65=conf["u65"], conf_u80=conf["u80"],
            raps_cov=raps["coverage"], raps_size=raps["mean_size"],
            mond_cov=mond["coverage"], mond_size=mond["mean_size"],
            determinacy=nat["determinacy"], set_cov=nat["coverage"], set_acc=nat["set_acc"],
            mean_size=nat["mean_size"], disc_acc=nat["disc_acc"], u65=nat["u65"], u80=nat["u80"]))
        rows[-1].update(evidence)

    df = pd.DataFrame(rows)
    if a.replace_models is not None:
        old = pd.read_csv(a.per_fold_csv)
        key = ["dataset", "model", "fold"]
        new_keys = pd.MultiIndex.from_frame(df[key])
        keep = ~pd.MultiIndex.from_frame(old[key]).isin(new_keys)
        df = pd.concat([old[keep], df], ignore_index=True)
    df.to_csv(a.per_fold_csv, index=False)
    ok = df[df.status == "ok"]
    dnf = df[df.status == "dnf"]
    cols = ["acc", "complexity", "n_rules", "condition_complexity", "avg_rule_length",
            "train_s", "pred_s", "aurc", "acc90",
            "ece_raw", "ece_iso", "ece_va",
            "conf_determinacy", "conf_cov", "conf_size", "conf_u65", "conf_u80",
            "raps_cov", "raps_size", "mond_cov", "mond_size",
            "determinacy", "set_cov", "set_acc", "mean_size", "u65", "u80",
            "ignorance", "true_belief", "true_plausibility", "evidence_aurc"]
    agg = ok.groupby("model")[cols].mean()
    agg["set_src"] = ok.groupby("model")["set_src"].agg(lambda s: s.mode().iat[0])
    agg.to_csv(a.summary_csv)
    print(f"=== SOTA benchmark (alpha={a.alpha}, mean over datasets x folds) ===")
    print(f"{'model':11s} {'acc':>6s} {'cplx':>8s} {'AURC':>6s} {'src':>9s} "
          f"{'determ':>6s} {'setcov':>6s} {'setsz':>6s} {'u65':>5s} {'u80':>5s}")
    for m, r in agg.iterrows():
        print(f"{m:11s} {r.acc:6.3f} {r.complexity:8.1f} {r.aurc:6.3f} {r.set_src:>9s} "
              f"{r.determinacy:6.2f} {r.set_cov:6.2f} {r.mean_size:6.2f} {r.u65:5.2f} {r.u80:5.2f}")
    print(f"\n--- calibration (ECE) & conformal family ---")
    print(f"{'model':11s} {'ECEraw':>6s} {'ECEiso':>6s} {'ECEva':>6s} "
          f"{'APScov':>6s} {'APSsz':>6s} {'RAPScov':>7s} {'RAPSsz':>6s} {'Mndcov':>6s} {'Mndsz':>6s}")
    for m, r in agg.iterrows():
        print(f"{m:11s} {r.ece_raw:6.3f} {r.ece_iso:6.3f} {r.ece_va:6.3f} "
              f"{r.conf_cov:6.2f} {r.conf_size:6.2f} {r.raps_cov:7.2f} {r.raps_size:6.2f} "
              f"{r.mond_cov:6.2f} {r.mond_size:6.2f}")
    evidence = agg[agg["ignorance"].notna()]
    if len(evidence):
        print("\n--- native evidential mass metrics ---")
        print(f"{'model':11s} {'ignor':>6s} {'trueBel':>7s} {'truePl':>7s} {'evAURC':>7s}")
        for m, r in evidence.iterrows():
            print(
                f"{m:11s} {r.ignorance:6.3f} {r.true_belief:7.3f} "
                f"{r.true_plausibility:7.3f} {r.evidence_aurc:7.3f}"
            )
    if len(dnf):
        print("\nDNF:", dnf.groupby("model").size().to_dict())


if __name__ == "__main__":
    main()
