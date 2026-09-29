"""Supplementary: characteristics table for the 30 tabular benchmark datasets.

Reads the dataset names from the main benchmark run and their #instances /
#features / #classes from the Keel headers, and emits paper/generated/ (FERL_GEN_DIR overrides)
tab_datasets.tex.

Run from the repo root:
    python experiments/paper_assets/make_dataset_list.py
"""
import os
from pathlib import Path
import pandas as pd

GEN = Path(os.environ.get("FERL_GEN_DIR", "paper/generated"))
GEN.mkdir(parents=True, exist_ok=True)
KEEL = os.environ.get("KEEL_DIR", "../keel_datasets")


def stats(name):
    f = os.path.join(KEEL, name, name + ".dat")
    n_feat = None
    labels, n_inst, in_hdr = set(), 0, True
    for line in open(f):
        s = line.strip()
        low = s.lower()
        if low.startswith("@inputs"):
            n_feat = len([t for t in s.split(None, 1)[1].split(",") if t.strip()])
        if low.startswith("@data"):
            in_hdr = False
            continue
        if in_hdr or not s:
            continue
        n_inst += 1
        labels.add(s.split(",")[-1].strip())
    return n_inst, n_feat, len(labels)


def main():
    from ferl.pipeline.run_configs import SELECTED_30
    names = sorted(SELECTED_30)   # the benchmark CSV also holds an extra dataset
    rows = [(n, *stats(n)) for n in names]

    def fmt(n, inst, feat, C):
        return f"{n} & {inst} & {feat} & {C}\\\\"

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{The thirty tabular benchmark datasets used in the main "
        r"results, with number of instances, input features (after one-hot "
        r"encoding of categoricals) and classes.}",
        r"\label{tab:datasets}", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
        r"Dataset & \#Inst. & \#Feat. & \#Cls.\\", r"\midrule",
        *[fmt(*r) for r in rows],
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (GEN / "tab_datasets.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote tab_datasets.tex ({len(rows)} datasets)")


if __name__ == "__main__":
    main()
