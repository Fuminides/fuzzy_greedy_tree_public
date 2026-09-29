"""Fast MDLP fuzzification: C kernel for the cut-point search.

Drop-in replacement for ``fuzzification_mdlp.learn_partitions_mdlp`` that uses
the compiled ``mdlp_cuts_c`` kernel for the per-feature cut search (the hot part)
while reusing the reference's trapezoid construction and ex_fuzzy object building
verbatim, so the produced partitions are identical.

    from ferl_fast.fuzzification_fast import learn_partitions_mdlp_fast
    parts = learn_partitions_mdlp_fast(X_train, y_train)
    clf = FuzzyCARTFast(parts, ...)
"""
from __future__ import annotations

import numpy as np
from ex_fuzzy import fuzzy_sets as fs

# Reuse the reference trapezoid builder so set shapes are byte-for-byte identical.
from ferl.fuzzification.fuzzification_mdlp import cuts_to_trapezoids
from ferl_fast import _kernels


def mdlp_cuts_fast(x: np.ndarray, y: np.ndarray) -> list[float]:
    """C-kernel MDLP cut points for one feature (sorted), matching mdlp_cuts."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    xs = np.ascontiguousarray(x[order], dtype=np.float64)
    # classes -> 0..C-1 in x order, same as the reference's np.unique(..., inverse).
    classes, ys = np.unique(np.asarray(y)[order], return_inverse=True)
    ys = np.ascontiguousarray(ys.astype(np.int32))
    cuts = _kernels.mdlp_cuts_c(xs, ys, int(len(classes)))
    return cuts.tolist()


def learn_partitions_mdlp_fast(X: np.ndarray, y: np.ndarray, overlap_frac: float = 0.8,
                               fallback_median: bool = True) -> list:
    """Supervised trapezoidal partitions via MDLP, cut search in C.

    Mirrors ``fuzzification_mdlp.learn_partitions_mdlp`` exactly except the
    per-feature cut search runs in the compiled kernel.
    """
    X = np.asarray(X, dtype=float)
    variables = []
    for j in range(X.shape[1]):
        col = X[:, j]
        lo, hi = float(col.min()), float(col.max())
        cuts = mdlp_cuts_fast(col, y)
        if not cuts and fallback_median and hi > lo:
            cuts = [float(np.median(col))]
        sets = cuts_to_trapezoids(cuts, lo, hi, overlap_frac=overlap_frac)
        variables.append(fs.fuzzyVariable(name=f"feature_{j}", fuzzy_sets=sets))
    return variables
