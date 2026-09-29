# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Compiled hot-path kernels for ferl_fast.

This file currently contains only a self-test stub used to validate that the
Python<->C build pipeline (Cython -> C -> .so, typed memoryviews, the numpy
C-API and a ``nogil`` reduction loop) works end to end on this machine. The
real split-scoring and vote-accumulation kernels are added once the toolchain
is proven.
"""
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.math cimport INFINITY, log2, pow
from cython.parallel cimport prange

cnp.import_array()


cdef inline double _seg_entropy(long *cnt, int C, double total) noexcept nogil:
    """-sum p*log2 p over classes with positive count (p = cnt[k]/total)."""
    cdef int k
    cdef double e = 0.0, p
    for k in range(C):
        if cnt[k] > 0:
            p = cnt[k] / total
            e -= p * log2(p)
    return e


def mdlp_cuts_c(double[::1] xs, int[::1] ys, int C):
    """MDLP (Fayyad-Irani) cut points for one already-sorted feature column.

    C port of ``fuzzification_mdlp.mdlp_cuts``. ``xs`` is the feature sorted
    ascending; ``ys`` are the corresponding class indices (0..C-1) in the same
    order. Maintains running class counts in a single left-to-right sweep per
    segment (no per-level cumsum/onehot allocation), reproducing the reference's
    gain, leftmost-argmax selection and MDL stopping criterion. Returns the
    sorted cut values as a float64 array.
    """
    cdef Py_ssize_t n = xs.shape[0]
    if n < 2:
        return np.empty(0, dtype=np.float64)

    cdef long *total = <long*>malloc(C * sizeof(long))
    cdef long *left = <long*>malloc(C * sizeof(long))
    cdef Py_ssize_t *stk_lo = <Py_ssize_t*>malloc((n + 2) * sizeof(Py_ssize_t))
    cdef Py_ssize_t *stk_hi = <Py_ssize_t*>malloc((n + 2) * sizeof(Py_ssize_t))
    cdef double *cuts = <double*>malloc(n * sizeof(double))
    if total == NULL or left == NULL or stk_lo == NULL or stk_hi == NULL or cuts == NULL:
        free(total); free(left); free(stk_lo); free(stk_hi); free(cuts)
        raise MemoryError()

    cdef Py_ssize_t sp = 0, ncuts = 0
    cdef Py_ssize_t lo, hi, m, j, best_i, s, i
    cdef int k, nz, k1, k2, best_k1, best_k2
    cdef double base_ent, el, er, gain, best_gain, best_el, best_er
    cdef double nl, nr, mm, delta, threshold, p, lc, rc

    stk_lo[0] = 0
    stk_hi[0] = n
    sp = 1

    with nogil:
        while sp > 0:
            sp -= 1
            lo = stk_lo[sp]
            hi = stk_hi[sp]
            m = hi - lo
            if m < 2:
                continue

            for k in range(C):
                total[k] = 0
            for s in range(lo, hi):
                total[ys[s]] += 1
            nz = 0
            for k in range(C):
                if total[k] > 0:
                    nz += 1
            if nz < 2:
                continue

            mm = <double>m
            base_ent = _seg_entropy(total, C, mm)

            for k in range(C):
                left[k] = 0
            best_gain = -INFINITY
            best_i = -1
            best_el = 0.0; best_er = 0.0; best_k1 = 0; best_k2 = 0

            for j in range(0, m - 1):
                left[ys[lo + j]] += 1            # left = counts of seg[0:j+1]
                if xs[lo + j + 1] != xs[lo + j]:  # cannot cut between equal values
                    nl = <double>(j + 1)
                    nr = mm - nl
                    el = 0.0; k1 = 0
                    for k in range(C):
                        lc = left[k]
                        if lc > 0:
                            p = lc / nl
                            el -= p * log2(p)
                            k1 += 1
                    er = 0.0; k2 = 0
                    for k in range(C):
                        rc = total[k] - left[k]
                        if rc > 0:
                            p = rc / nr
                            er -= p * log2(p)
                            k2 += 1
                    gain = base_ent - (nl / mm) * el - (nr / mm) * er
                    if gain > best_gain:          # leftmost max (strict >)
                        best_gain = gain
                        best_i = lo + j + 1
                        best_el = el; best_er = er; best_k1 = k1; best_k2 = k2

            if best_i < 0:
                continue

            delta = log2(pow(3.0, <double>nz) - 2.0) - (nz * base_ent - best_k1 * best_el - best_k2 * best_er)
            threshold = (log2(mm - 1.0) + delta) / mm
            if best_gain <= threshold:
                continue

            cuts[ncuts] = (xs[best_i - 1] + xs[best_i]) / 2.0
            ncuts += 1
            stk_lo[sp] = lo; stk_hi[sp] = best_i; sp += 1
            stk_lo[sp] = best_i; stk_hi[sp] = hi; sp += 1

    out = np.empty(ncuts, dtype=np.float64)
    cdef double[::1] outv = out
    for i in range(ncuts):
        outv[i] = cuts[i]

    free(total); free(left); free(stk_lo); free(stk_hi); free(cuts)
    out.sort()
    return out


def learned_best_cut_sorted(double[::1] xs, int[::1] ys, double[::1] w,
                            int n_classes, double parent, double W):
    """Weighted-Gini best threshold for one already-sorted feature column.

    This is the allocation-free inner loop used by performance-mode learned
    splits. It mirrors ``tree_learning._learned_best_cut`` once the caller has
    sorted ``x`` with stable mergesort. Returns ``(gain, threshold, ok)``.
    """
    cdef Py_ssize_t n = xs.shape[0]
    if n < 2 or W <= 1e-12:
        return 0.0, 0.0, 0

    cdef double *left = <double*>malloc(n_classes * sizeof(double))
    cdef double *total = <double*>malloc(n_classes * sizeof(double))
    if left == NULL or total == NULL:
        free(left); free(total)
        raise MemoryError()

    cdef Py_ssize_t i, k
    cdef double Wl = 0.0, Wr, gini_l, gini_r, p, gain
    cdef double best_gain = -INFINITY
    cdef double best_thr = 0.0

    with nogil:
        for k in range(n_classes):
            left[k] = 0.0
            total[k] = 0.0
        for i in range(n):
            total[ys[i]] += w[i]

        for i in range(n - 1):
            Wl += w[i]
            left[ys[i]] += w[i]
            if xs[i + 1] == xs[i]:
                continue
            Wr = W - Wl
            if Wl <= 1e-12 or Wr <= 1e-12:
                continue
            gini_l = 1.0
            gini_r = 1.0
            for k in range(n_classes):
                p = left[k] / Wl
                gini_l -= p * p
                p = (total[k] - left[k]) / Wr
                gini_r -= p * p
            gain = parent - (Wl / W) * gini_l - (Wr / W) * gini_r
            if gain > best_gain:
                best_gain = gain
                best_thr = 0.5 * (xs[i] + xs[i + 1])

    free(left); free(total)
    if best_gain <= 0.0:
        return 0.0, 0.0, 0
    return best_gain, best_thr, 1


def fill_memberships(double[:, ::1] X,
                     int[::1] feat,
                     unsigned char[::1] kind,
                     double[::1] pa,
                     double[::1] pb,
                     double[::1] pc,
                     double[::1] pd,
                     double[:, ::1] out,
                     int num_threads=1):
    """Fill ``out[:, j]`` with the membership of feature ``feat[j]`` to set j.

    Bit-exact C port of ex_fuzzy's ``trapezoidal_membership`` / categorical
    membership for the numpy path. ``kind[j]`` selects:
      * 0 -> trapezoid: clip(min((x-a)/(b-a), (d-x)/(d-c)), 0, 1), with
        ``pb``/``pd`` already carrying ex_fuzzy's b+=eps / d+=eps adjustments;
      * 1 -> equality (categorical or singleton trapezoid a==d): x == pa[j].

    Divisions (not reciprocal multiplies) are used so the result matches numpy
    elementwise. Parallelised over samples (each row independent), so results
    are identical to serial regardless of ``num_threads``. ``out`` must be
    preallocated ``(n_samples, m)``.
    """
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t m = feat.shape[0]
    cdef Py_ssize_t s, j, f
    cdef double a, b, c, d, x, aux1, aux2, mv

    for s in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
        for j in range(m):
            f = feat[j]
            if kind[j] == 1:
                out[s, j] = 1.0 if X[s, f] == pa[j] else 0.0
            elif kind[j] == 2:
                # LearnedRampSet, direction='below':
                # clip((center + h - x) / (2h), 0, 1)
                x = (pa[j] + pb[j] - X[s, f]) / (2.0 * pb[j])
                if x < 0.0:
                    x = 0.0
                elif x > 1.0:
                    x = 1.0
                out[s, j] = x
            elif kind[j] == 3:
                # LearnedRampSet, direction='above': 1 - below.
                x = (pa[j] + pb[j] - X[s, f]) / (2.0 * pb[j])
                if x < 0.0:
                    x = 0.0
                elif x > 1.0:
                    x = 1.0
                out[s, j] = 1.0 - x
            else:
                a = pa[j]; b = pb[j]; c = pc[j]; d = pd[j]
                x = X[s, f]
                aux1 = (x - a) / (b - a)
                aux2 = (d - x) / (d - c)
                mv = aux1 if aux1 < aux2 else aux2
                if mv < 0.0:
                    mv = 0.0
                elif mv > 1.0:
                    mv = 1.0
                out[s, j] = mv


def fill_memberships_masked(double[:, ::1] X,
                            int[::1] feat,
                            unsigned char[::1] kind,
                            double[::1] pa,
                            double[::1] pb,
                            double[::1] pc,
                            double[::1] pd,
                            unsigned char[:, ::1] observed,
                            double[::1] uniform,
                            double[:, ::1] out,
                            int num_threads=1):
    """Like ``fill_memberships`` but with per-sample missing features.

    Where ``observed[s, feat[j]] == 0`` the membership is ``uniform[j]`` (=
    1/n_partitions of that feature) exactly as ex_fuzzy's observed-mask handling
    in ``_predict_proba_all_nodes``; otherwise the normal trapezoid/equality
    membership is computed.
    """
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t m = feat.shape[0]
    cdef Py_ssize_t s, j, f
    cdef double a, b, c, d, x, aux1, aux2, mv

    for s in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
        for j in range(m):
            f = feat[j]
            if observed[s, f] == 0:
                out[s, j] = uniform[j]
            elif kind[j] == 1:
                out[s, j] = 1.0 if X[s, f] == pa[j] else 0.0
            elif kind[j] == 2:
                x = (pa[j] + pb[j] - X[s, f]) / (2.0 * pb[j])
                if x < 0.0:
                    x = 0.0
                elif x > 1.0:
                    x = 1.0
                out[s, j] = x
            elif kind[j] == 3:
                x = (pa[j] + pb[j] - X[s, f]) / (2.0 * pb[j])
                if x < 0.0:
                    x = 0.0
                elif x > 1.0:
                    x = 1.0
                out[s, j] = 1.0 - x
            else:
                a = pa[j]; b = pb[j]; c = pc[j]; d = pd[j]
                x = X[s, f]
                aux1 = (x - a) / (b - a)
                aux2 = (d - x) / (d - c)
                mv = aux1 if aux1 < aux2 else aux2
                if mv < 0.0:
                    mv = 0.0
                elif mv > 1.0:
                    mv = 1.0
                out[s, j] = mv


cdef inline void _node_memb_one_sample(Py_ssize_t s, double[:, ::1] memb,
                                       int[::1] cand, int[::1] starts,
                                       double[:, ::1] outM, Py_ssize_t K) noexcept nogil:
    cdef Py_ssize_t k, p
    cdef double pm
    for k in range(K):
        pm = 1.0
        for p in range(starts[k], starts[k + 1]):
            pm = pm * memb[s, cand[p]]
        outM[s, k] = pm


def node_membership_matrix(double[:, ::1] memb,
                           int[::1] cand,
                           int[::1] starts,
                           double[:, ::1] outM,
                           int num_threads=1):
    """Per-node path-membership matrix M[s, k] for ``node_activation_matrix``.

    ``M[s,k]`` is the product of the candidate memberships along node k's path
    (CSR ``cand[starts[k]:starts[k+1]]``). Parallel over samples; deterministic.
    """
    cdef Py_ssize_t n = memb.shape[0]
    cdef Py_ssize_t K = starts.shape[0] - 1
    cdef Py_ssize_t s
    for s in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
        _node_memb_one_sample(s, memb, cand, starts, outM, K)


cdef inline void _ds_one_sample(Py_ssize_t s, double[:, ::1] M, double[:, ::1] cons,
                                double[:, ::1] bp, double[:, ::1] be,
                                double[:, ::1] pls, double[::1] ig,
                                Py_ssize_t K, Py_ssize_t Cn) noexcept nogil:
    """One sample's DS combination (kept out of the prange body so the local
    accumulators are not misread as cross-thread reductions)."""
    cdef Py_ssize_t k, c
    cdef double qtheta = 1.0, mk, om, total, v, mc, mth
    for c in range(Cn):
        bp[s, c] = 1.0
    for k in range(K):
        mk = M[s, k]
        om = 1.0 - mk
        qtheta = qtheta * om
        for c in range(Cn):
            bp[s, c] = bp[s, c] * (mk * cons[k, c] + om)
    total = 0.0
    for c in range(Cn):
        v = bp[s, c] - qtheta
        if v < 0.0:
            v = 0.0
        be[s, c] = v
        total = total + v
    total = total + qtheta
    if total <= 0.0:
        total = 1.0
    mth = qtheta / total
    for c in range(Cn):
        mc = be[s, c] / total
        be[s, c] = mc
        pls[s, c] = mc + mth
        bp[s, c] = mc + mth / Cn
    ig[s] = mth


def predict_ds_combine(double[:, ::1] M, double[:, ::1] cons, int num_threads=1):
    """Dempster-Shafer combination over rule nodes (closed form, streaming).

    C port of the combination in ``FuzzyCART.predict_ds`` for the singleton+Theta
    mass structure. Avoids the reference's (N, K, C) intermediate tensor by
    streaming the commonality products over K with O(C) scratch:
        Qc[c]   = prod_k ( M[s,k]*cons[k,c] + (1 - M[s,k]) )
        Qtheta  = prod_k ( 1 - M[s,k] )
    then m_c = clip(Qc - Qtheta, 0)/total, m_theta = Qtheta/total with
    total = sum_c m_c_un + Qtheta. Returns (betp, bel, pl, ignorance). The betp
    and bel output buffers double as Qc / m_c_un scratch. Parallel over samples.
    """
    cdef Py_ssize_t N = M.shape[0]
    cdef Py_ssize_t K = M.shape[1]
    cdef Py_ssize_t Cn = cons.shape[1]
    cdef Py_ssize_t s

    betp = np.empty((N, Cn), dtype=np.float64)
    bel = np.empty((N, Cn), dtype=np.float64)
    pl = np.empty((N, Cn), dtype=np.float64)
    ign = np.empty(N, dtype=np.float64)
    cdef double[:, ::1] bp = betp
    cdef double[:, ::1] be = bel
    cdef double[:, ::1] pls = pl
    cdef double[::1] ig = ign

    for s in prange(N, nogil=True, num_threads=num_threads, schedule='static'):
        _ds_one_sample(s, M, cons, bp, be, pls, ig, K, Cn)
    return betp, bel, pl, ign


cdef inline void _ds_one_sample_masses(Py_ssize_t s, double[:, ::1] mu,
                                       double[:, ::1] a_eff, double[::1] t_eff,
                                       double[:, ::1] bp, double[:, ::1] be,
                                       double[:, ::1] pls, double[::1] ig,
                                       Py_ssize_t K, Py_ssize_t Cn) noexcept nogil:
    """One sample's generalized (per-node Theta + singleton pseudo-mass) DS
    combination. Identical tail to ``_ds_one_sample``; only the two per-node
    factors differ (incremental rule): the node's effective Theta mass is
    ``f_th = 1 - mu*(1 - t_eff)`` and its singleton pseudo-mass is ``a_eff``."""
    cdef Py_ssize_t k, c
    cdef double qtheta = 1.0, muk, f_th, total, v, mc, mth
    for c in range(Cn):
        bp[s, c] = 1.0
    for k in range(K):
        muk = mu[s, k]
        f_th = 1.0 - muk * (1.0 - t_eff[k])
        qtheta = qtheta * f_th
        for c in range(Cn):
            bp[s, c] = bp[s, c] * (muk * a_eff[k, c] + f_th)
    total = 0.0
    for c in range(Cn):
        v = bp[s, c] - qtheta
        if v < 0.0:
            v = 0.0
        be[s, c] = v
        total = total + v
    total = total + qtheta
    if total <= 0.0:
        total = 1.0
    mth = qtheta / total
    for c in range(Cn):
        mc = be[s, c] / total
        be[s, c] = mc
        pls[s, c] = mc + mth
        bp[s, c] = mc + mth / Cn
    ig[s] = mth


def predict_ds_combine_masses(double[:, ::1] mu, double[:, ::1] a_eff,
                              double[::1] t_eff, int num_threads=1):
    """Generalized streaming Dempster combiner for the incremental rule.

    Unlike ``predict_ds_combine`` (which assumes each node's base Theta mass is 0,
    i.e. ``om = 1 - mu``), incremental nodes carry a per-node effective Theta mass
    ``t_eff[k]`` and a per-node singleton pseudo-mass ``a_eff[k, c]`` (a
    commonality-derived residual the final normalization rescales -- it does NOT
    sum to 1 with ``t_eff``, which is why this cannot reuse the cons kernel).

    Per sample, streams the commonality products over K with O(C) scratch:
        f_th    = 1 - mu[s,k]*(1 - t_eff[k])
        Qc[c]   = prod_k ( mu[s,k]*a_eff[k,c] + f_th )
        Qtheta  = prod_k f_th
    then m_c = clip(Qc - Qtheta, 0)/total, m_theta = Qtheta/total with
    total = sum_c m_c_un + Qtheta. Returns (betp, bel, pl, ignorance). Parallel
    over samples (each row independent -> bit-exact regardless of thread count).
    """
    cdef Py_ssize_t N = mu.shape[0]
    cdef Py_ssize_t K = mu.shape[1]
    cdef Py_ssize_t Cn = a_eff.shape[1]
    cdef Py_ssize_t s

    betp = np.empty((N, Cn), dtype=np.float64)
    bel = np.empty((N, Cn), dtype=np.float64)
    pl = np.empty((N, Cn), dtype=np.float64)
    ign = np.empty(N, dtype=np.float64)
    cdef double[:, ::1] bp = betp
    cdef double[:, ::1] be = bel
    cdef double[:, ::1] pls = pl
    cdef double[::1] ig = ign

    for s in prange(N, nogil=True, num_threads=num_threads, schedule='static'):
        _ds_one_sample_masses(s, mu, a_eff, t_eff, bp, be, pls, ig, K, Cn)
    return betp, bel, pl, ign


cdef inline void _ds_one_sample_cautious(Py_ssize_t s, double[:, ::1] M, double[:, ::1] cons,
                                         double[:, ::1] bp, double[:, ::1] be,
                                         double[:, ::1] pls, double[::1] ig,
                                         Py_ssize_t K, Py_ssize_t Cn) noexcept nogil:
    """Denoeux's cautious rule over all K nodes (idempotent). Streams the
    per-class min of the canonical weights w_n({c}) = (1-mu)/(mu p_n(c) + 1-mu)
    with O(C) scratch, then m({c}) = (1/w_c - 1)/S, m(Theta) = 1/S where
    S = (1 - C) + sum_c 1/w_c. ``be[s]`` holds the running min (then 1/w_c);
    ``bp[s]`` doubles as scratch. Mirrors ``FuzzyCART._cautious_mass``."""
    cdef Py_ssize_t k, c
    cdef double mk, om, term, w, inv, S, mth, mc
    for c in range(Cn):
        be[s, c] = INFINITY
    for k in range(K):
        mk = M[s, k]
        om = 1.0 - mk
        for c in range(Cn):
            term = mk * cons[k, c] + om
            if term < 1e-12:
                term = 1e-12
            w = om / term
            if w < be[s, c]:
                be[s, c] = w
    S = 1.0 - Cn
    for c in range(Cn):
        w = be[s, c]
        if w < 1e-12:
            w = 1e-12
        elif w > 1.0:
            w = 1.0
        inv = 1.0 / w
        bp[s, c] = inv
        S = S + inv
    mth = 1.0 / S
    for c in range(Cn):
        mc = (bp[s, c] - 1.0) / S
        be[s, c] = mc
        pls[s, c] = mc + mth
        bp[s, c] = mc + mth / Cn
    ig[s] = mth


def predict_ds_combine_cautious(double[:, ::1] M, double[:, ::1] cons, int num_threads=1):
    """Cautious (Denoeux) DS combination over all rule nodes. Streaming O(C)
    scratch version of ``FuzzyCART._cautious_mass`` applied to every column.
    Returns (betp, bel, pl, ignorance). Parallel over samples (bit-exact)."""
    cdef Py_ssize_t N = M.shape[0]
    cdef Py_ssize_t K = M.shape[1]
    cdef Py_ssize_t Cn = cons.shape[1]
    cdef Py_ssize_t s

    betp = np.empty((N, Cn), dtype=np.float64)
    bel = np.empty((N, Cn), dtype=np.float64)
    pl = np.empty((N, Cn), dtype=np.float64)
    ign = np.empty(N, dtype=np.float64)
    cdef double[:, ::1] bp = betp
    cdef double[:, ::1] be = bel
    cdef double[:, ::1] pls = pl
    cdef double[::1] ig = ign

    for s in prange(N, nogil=True, num_threads=num_threads, schedule='static'):
        _ds_one_sample_cautious(s, M, cons, bp, be, pls, ig, K, Cn)
    return betp, bel, pl, ign


cdef inline void _ds_one_sample_hybrid(Py_ssize_t s, double[:, ::1] M, double[:, ::1] cons,
                                       int[::1] cols, int[::1] lstart, Py_ssize_t n_leaves,
                                       double[:, ::1] bp, double[:, ::1] be,
                                       double[:, ::1] pls, double[::1] ig,
                                       Py_ssize_t Cn) noexcept nogil:
    """Structure-aware rule: cautious within each root->leaf chain (the chain's
    columns ``cols[lstart[li]:lstart[li+1]]``), then Dempster across chains.
    ``bp[s]`` accumulates the across-chain commonality Qc; ``be[s]`` is per-chain
    cautious scratch. Mirrors the hybrid branch of ``FuzzyCART.predict_ds``."""
    cdef Py_ssize_t li, idx, k, c
    cdef double qt = 1.0, mk, om, term, w, inv, S, mt, mc, total, v, mth
    for c in range(Cn):
        bp[s, c] = 1.0
    for li in range(n_leaves):
        for c in range(Cn):
            be[s, c] = INFINITY
        for idx in range(lstart[li], lstart[li + 1]):
            k = cols[idx]
            mk = M[s, k]
            om = 1.0 - mk
            for c in range(Cn):
                term = mk * cons[k, c] + om
                if term < 1e-12:
                    term = 1e-12
                w = om / term
                if w < be[s, c]:
                    be[s, c] = w
        S = 1.0 - Cn
        for c in range(Cn):
            w = be[s, c]
            if w < 1e-12:
                w = 1e-12
            elif w > 1.0:
                w = 1.0
            inv = 1.0 / w
            be[s, c] = inv
            S = S + inv
        mt = 1.0 / S
        for c in range(Cn):
            mc = (be[s, c] - 1.0) / S
            bp[s, c] = bp[s, c] * (mc + mt)
        qt = qt * mt
    total = 0.0
    for c in range(Cn):
        v = bp[s, c] - qt
        if v < 0.0:
            v = 0.0
        be[s, c] = v
        total = total + v
    total = total + qt
    if total <= 0.0:
        total = 1.0
    mth = qt / total
    for c in range(Cn):
        mc = be[s, c] / total
        be[s, c] = mc
        pls[s, c] = mc + mth
        bp[s, c] = mc + mth / Cn
    ig[s] = mth


def predict_ds_combine_hybrid(double[:, ::1] M, double[:, ::1] cons,
                              int[::1] cols, int[::1] lstart, int num_threads=1):
    """Hybrid (cautious-within-chain, Dempster-across-chains) DS combination.

    ``cols`` is the concatenation of each leaf chain's node-column indices (in
    the reference's leaf order, ancestor columns ascending); ``lstart`` is the
    CSR row offsets (length n_leaves + 1). Returns (betp, bel, pl, ignorance).
    Parallel over samples (bit-exact)."""
    cdef Py_ssize_t N = M.shape[0]
    cdef Py_ssize_t Cn = cons.shape[1]
    cdef Py_ssize_t n_leaves = lstart.shape[0] - 1
    cdef Py_ssize_t s

    betp = np.empty((N, Cn), dtype=np.float64)
    bel = np.empty((N, Cn), dtype=np.float64)
    pl = np.empty((N, Cn), dtype=np.float64)
    ign = np.empty(N, dtype=np.float64)
    cdef double[:, ::1] bp = betp
    cdef double[:, ::1] be = bel
    cdef double[:, ::1] pls = pl
    cdef double[::1] ig = ign

    for s in prange(N, nogil=True, num_threads=num_threads, schedule='static'):
        _ds_one_sample_hybrid(s, M, cons, cols, lstart, n_leaves, bp, be, pls, ig, Cn)
    return betp, bel, pl, ign


def accumulate_soft_votes(double[:, ::1] memb,
                          int[::1] cand,
                          int[::1] starts,
                          double[:, ::1] node_probs,
                          int n_classes,
                          int num_threads=1):
    """Soft-mode vote accumulation across all non-root nodes.

    Faithful C port of the ``use_soft`` branch of
    ``FuzzyCART._predict_proba_all_nodes``. For each sample and each non-root
    node, the node's path membership ``pm`` (product of per-candidate
    memberships along its root->node path) weights the node's class-probability
    vector; these are summed into ``class_mem`` with ``pm`` summed into
    ``total``.

    The redundant per-node membership recomputation of the reference is avoided:
    ``memb`` holds each candidate's membership once and paths just index into it.

    To stay bit-identical to the reference, the caller must supply:
      * candidates within each node's path in root->node order, and
      * nodes (the CSR rows) in the reference accumulation order
        (stable sort by node prediction), root excluded.

    Parameters
    ----------
    memb : (n, C) float64, C-contiguous
        Per-candidate membership, candidate ``c`` is column ``c``.
    cand : (P,) int32
        Flattened path candidate indices for all nodes (CSR values).
    starts : (n_nodes + 1,) int32
        CSR row offsets into ``cand``; node ``j`` owns ``cand[starts[j]:starts[j+1]]``.
    node_probs : (n_nodes, n_classes) float64
        Per-node class-probability vectors.

    Returns
    -------
    (class_mem, total) : (n, n_classes) float64, (n,) float64
        Unnormalised vote accumulators (matching ``return_votes=True``).
    """
    cdef Py_ssize_t n = memb.shape[0]
    cdef Py_ssize_t n_nodes = node_probs.shape[0]
    cdef Py_ssize_t s, j, p, k
    cdef double pm

    class_mem = np.zeros((n, n_classes), dtype=np.float64)
    total = np.zeros(n, dtype=np.float64)
    cdef double[:, ::1] cm = class_mem
    cdef double[::1] tot = total

    # Parallelised over samples: each row is written by exactly one thread and
    # accumulates nodes in the same order as serial, so the result is identical.
    for s in prange(n, nogil=True, num_threads=num_threads, schedule='static'):
        for j in range(n_nodes):
            pm = 1.0
            for p in range(starts[j], starts[j + 1]):
                pm *= memb[s, cand[p]]
            for k in range(n_classes):
                cm[s, k] += pm * node_probs[j, k]
            tot[s] += pm
    return class_mem, total


def accumulate_node_votes(double[:, ::1] M,
                          double[:, ::1] node_probs,
                          int[::1] pred_idx,
                          int[::1] child_cols,
                          int[::1] child_starts,
                          unsigned char[::1] is_internal,
                          int mode_code,
                          double epsilon,
                          int n_classes,
                          int num_threads=1):
    """Accumulate all-node probabilities for the non-default prediction modes.

    ``mode_code``:
      * 0: soft probabilities, no internal-node gate (root should be omitted by
        the caller to match reference ``prediction_mode='soft'``);
      * 1: soft probabilities with internal-node gate (``soft_gate``);
      * 2: hard one-hot votes with internal-node gate (``hard_gate``);
      * 3: hard one-hot votes without gate (reference proba path for
        ``prediction_mode='winner'``).

    ``M`` is the already-computed node path-membership matrix in the same node
    order as the other arrays. Child CSR indices are in this same node order.
    """
    cdef Py_ssize_t N = M.shape[0]
    cdef Py_ssize_t K = M.shape[1]
    cdef Py_ssize_t s, j, c, p, child
    cdef double valid
    cdef bint gate = mode_code == 1 or mode_code == 2
    cdef bint use_soft = mode_code == 0 or mode_code == 1
    cdef int pi

    class_mem = np.zeros((N, n_classes), dtype=np.float64)
    total = np.zeros(N, dtype=np.float64)
    cdef double[:, ::1] cm = class_mem
    cdef double[::1] tot = total

    for s in prange(N, nogil=True, num_threads=num_threads, schedule='static'):
        for j in range(K):
            valid = M[s, j]
            if gate and is_internal[j] != 0:
                for p in range(child_starts[j], child_starts[j + 1]):
                    child = child_cols[p]
                    if M[s, child] > epsilon:
                        valid = 0.0
                        break
            if use_soft:
                for c in range(n_classes):
                    cm[s, c] += valid * node_probs[j, c]
                tot[s] += valid
            else:
                pi = pred_idx[j]
                if pi >= 0:
                    cm[s, pi] += valid
                    tot[s] += valid

    return class_mem, total


def select_best_nodes(double[:, ::1] M,
                      int[::1] pred_idx,
                      int[::1] child_cols,
                      int[::1] child_starts,
                      unsigned char[::1] is_internal,
                      int[::1] order,
                      double epsilon,
                      int num_threads=1):
    """Winner-style all-node prediction with internal-node constraints.

    Mirrors ``FuzzyCART._predict_all_nodes`` for fully-observed inputs. ``order``
    is the exact reference processing order: leaves first, then internal nodes
    deepest-to-root. Ties keep the earlier node because updates use strict ``>``.
    Returns prediction class indices, best memberships, and selected node cols.
    """
    cdef Py_ssize_t N = M.shape[0]
    cdef Py_ssize_t O = order.shape[0]
    cdef Py_ssize_t s, oi, j, p, child
    cdef double valid, best
    cdef int best_pred, best_col

    pred = np.empty(N, dtype=np.int32)
    best_m = np.empty(N, dtype=np.float64)
    best_node = np.empty(N, dtype=np.int32)
    cdef int[::1] pr = pred
    cdef double[::1] bm = best_m
    cdef int[::1] bn = best_node

    for s in prange(N, nogil=True, num_threads=num_threads, schedule='static'):
        best = -1.0
        best_pred = -1
        best_col = -1
        for oi in range(O):
            j = order[oi]
            valid = M[s, j]
            if is_internal[j] != 0:
                for p in range(child_starts[j], child_starts[j + 1]):
                    child = child_cols[p]
                    if M[s, child] > epsilon:
                        valid = 0.0
                        break
            if valid > best:
                best = valid
                best_pred = pred_idx[j]
                best_col = <int>j
        pr[s] = best_pred
        bm[s] = best
        bn[s] = best_col
    return pred, best_m, best_node


cdef inline double _learned_ramp_mu(double x, double center, double h,
                                    int direction) noexcept nogil:
    cdef double mu = (center + h - x) / (2.0 * h)
    if mu < 0.0:
        mu = 0.0
    elif mu > 1.0:
        mu = 1.0
    if direction == 1:
        mu = 1.0 - mu
    return mu


def score_learned_ramps(double[:, ::1] X,
                        unsigned char[::1] valid,
                        double[::1] centers,
                        double[::1] hs,
                        double[::1] existing,
                        int[::1] y_int,
                        double[:, ::1] base_votes,
                        int n_classes,
                        double cov_thresh,
                        double acc_pre,
                        double best_cci_init,
                        unsigned char[::1] uncovered,
                        double coverage_weight,
                        int n_uncovered):
    """Score learned ramp candidates for one node.

    For every valid feature, scores both directions ('below', 'above') in the
    same additive soft-vote CCI space as the reference performance mode.
    Returns ``(feature, direction, cci, purity, coverage, child_idx)`` where
    direction 0 is below and 1 is above.
    """
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]
    cdef Py_ssize_t f, s, k
    cdef int direction
    cdef double center, h, mu, full, tot, cov, gini, p, purity
    cdef double cci, acc_new, mv
    cdef Py_ssize_t child_idx, new_pred, correct_new, cov_hits

    cdef double best_cci = best_cci_init
    cdef double best_purity = INFINITY
    cdef int best_feature = -1
    cdef int best_direction = 0
    cdef double best_coverage = 0.0
    cdef int best_child_idx = 0

    cdef double *cnt = <double*>malloc(n_classes * sizeof(double))
    cdef double *probs = <double*>malloc(n_classes * sizeof(double))
    if cnt == NULL or probs == NULL:
        free(cnt); free(probs)
        raise MemoryError()

    with nogil:
        for f in range(n_features):
            if valid[f] == 0:
                continue
            center = centers[f]
            h = hs[f]
            if h <= 0.0:
                continue

            for direction in range(2):
                tot = 0.0
                for k in range(n_classes):
                    cnt[k] = 0.0
                for s in range(n):
                    mu = _learned_ramp_mu(X[s, f], center, h, direction)
                    full = mu * existing[s]
                    if full != 0.0:
                        tot += full
                        cnt[y_int[s]] += full

                cov = tot / n
                if cov < cov_thresh:
                    continue

                if tot > 0.0:
                    child_idx = 0
                    mv = cnt[0]
                    for k in range(1, n_classes):
                        if cnt[k] > mv:
                            mv = cnt[k]
                            child_idx = k
                    for k in range(n_classes):
                        probs[k] = cnt[k] / tot
                    gini = 1.0
                    for k in range(n_classes):
                        p = cnt[k] / tot
                        gini -= p * p
                    purity = gini
                else:
                    child_idx = 0
                    for k in range(n_classes):
                        probs[k] = 1.0 / n_classes
                    purity = INFINITY

                if tot == 0.0:
                    cci = 0.0
                else:
                    correct_new = 0
                    for s in range(n):
                        mu = _learned_ramp_mu(X[s, f], center, h, direction)
                        full = mu * existing[s]
                        new_pred = 0
                        mv = base_votes[s, 0] + full * probs[0]
                        for k in range(1, n_classes):
                            p = base_votes[s, k] + full * probs[k]
                            if p > mv:
                                mv = p
                                new_pred = k
                        if new_pred == y_int[s]:
                            correct_new += 1
                    acc_new = (<double>correct_new) / n
                    if acc_pre == 0.0:
                        cci = acc_new
                    else:
                        cci = (acc_new - acc_pre) / acc_pre

                if coverage_weight > 0.0 and n_uncovered > 0:
                    cov_hits = 0
                    for s in range(n):
                        if uncovered[s] != 0:
                            mu = _learned_ramp_mu(X[s, f], center, h, direction)
                            full = mu * existing[s]
                            if full > 1e-3:
                                cov_hits += 1
                    cci = cci + coverage_weight * ((<double>cov_hits) / n_uncovered)

                if cci > best_cci:
                    best_cci = cci
                    best_purity = purity
                    best_feature = <int>f
                    best_direction = direction
                    best_coverage = cov
                    best_child_idx = <int>child_idx
                elif cci == best_cci and purity < best_purity:
                    best_cci = cci
                    best_purity = purity
                    best_feature = <int>f
                    best_direction = direction
                    best_coverage = cov
                    best_child_idx = <int>child_idx

    free(cnt); free(probs)
    return (best_feature, best_direction, best_cci, best_purity,
            best_coverage, best_child_idx)


def score_node_purity(double[:, ::1] memb_flat,
                      int[::1] cand_feature,
                      int[::1] cand_fz,
                      unsigned char[::1] legal,
                      double[::1] existing,
                      int[::1] y_int,
                      int n_classes,
                      double cov_thresh,
                      double father_purity):
    """Score legal candidate splits for one node by fuzzy-Gini purity gain.

    C port of the inner loop of ``FuzzyCART._node_purity_checks``. For each
    candidate, improvement = ``father_purity - purity`` where purity is the
    weighted-Gini of the candidate's path membership; a candidate with coverage
    below ``cov_thresh`` or zero total membership has purity = +inf (improvement
    -inf, never selected), matching ``compute_fuzzy_purity``. Selection is the
    reference's: strictly greater improvement wins, feature-major scan.

    ``father_purity`` is passed in (computed once in Python with the reference
    ``compute_fuzzy_purity`` so it is bit-identical to the reference's value).

    Returns ``(best_feature, best_fz, best_improvement, best_coverage,
    best_child_idx)``; ``best_feature == -1`` means no candidate was taken.
    """
    cdef Py_ssize_t C = memb_flat.shape[0]
    cdef Py_ssize_t n = memb_flat.shape[1]
    cdef Py_ssize_t c, s, k
    cdef double tot, full, cov, gini, p, purity, improvement, mv
    cdef Py_ssize_t child_idx

    cdef double best_improvement = -INFINITY
    cdef int best_feature = -1
    cdef int best_fz = -1
    cdef double best_coverage = 0.0
    cdef int best_child_idx = 0

    cdef double *cnt = <double*>malloc(n_classes * sizeof(double))
    if cnt == NULL:
        raise MemoryError()

    with nogil:
        for c in range(C):
            if legal[c] == 0:
                continue

            tot = 0.0
            for k in range(n_classes):
                cnt[k] = 0.0
            for s in range(n):
                full = memb_flat[c, s] * existing[s]
                if full != 0.0:
                    tot += full
                    cnt[y_int[s]] += full

            cov = tot / n
            if tot == 0.0 or cov < cov_thresh:
                # purity = +inf -> improvement = -inf, never selected.
                continue

            gini = 1.0
            for k in range(n_classes):
                p = cnt[k] / tot
                gini -= p * p
            purity = gini
            improvement = father_purity - purity

            child_idx = 0
            mv = cnt[0]
            for k in range(1, n_classes):
                if cnt[k] > mv:
                    mv = cnt[k]
                    child_idx = k

            if improvement > best_improvement:
                best_improvement = improvement
                best_feature = cand_feature[c]
                best_fz = cand_fz[c]
                best_coverage = cov
                best_child_idx = <int>child_idx

    free(cnt)
    return (best_feature, best_fz, best_improvement, best_coverage, best_child_idx)


def score_node_cci(double[:, ::1] memb_flat,
                   int[::1] cand_feature,
                   int[::1] cand_fz,
                   unsigned char[::1] legal,
                   double[::1] existing,
                   int[::1] y_int,
                   double[:, ::1] base_votes,
                   int n_classes,
                   double cov_thresh,
                   double acc_pre,
                   double best_cci_init,
                   unsigned char[::1] uncovered,
                   double coverage_weight,
                   int n_uncovered):
    """Score every legal candidate split for a single node (consistent-CCI).

    Faithful C port of the inner loop of ``FuzzyCART._node_cci_checks`` for the
    default ``consistent_cci`` / ``soft`` configuration. Returns the best
    candidate using the reference tie-break: maximise CCI, then minimise the
    fuzzy-Gini purity, scanning candidates in feature-major / fuzzy-set-minor
    order so ties resolve identically to the pure-Python implementation.

    Parameters
    ----------
    memb_flat : (C, n) float64, C-contiguous
        Per-candidate membership of each sample. Candidate ``c`` corresponds to
        ``(cand_feature[c], cand_fz[c])`` and rows are ordered feature-major.
    legal : (C,) uint8
        1 if candidate is allowed for this node (father_path AND child_splits).
    existing : (n,) float64
        The node's existing path membership.
    y_int : (n,) int32
        Class indices in ``0..n_classes-1`` (argmax space of ``classes_``).
    base_votes : (n, n_classes) float64
        Unnormalised baseline soft-vote sums for the frozen tree.
    acc_pre : float
        Baseline accuracy ``mean(argmax(base_votes) == y_int)`` (constant per scan).
    best_cci_init : float
        Initial best CCI (``-inf`` while ``tree_rules <= 3`` else ``0.0``).

    Returns
    -------
    tuple
        ``(best_feature, best_fz, best_cci, best_purity, best_coverage,
        best_child_idx)``. ``best_feature == -1`` means no candidate was taken.
    """
    cdef Py_ssize_t C = memb_flat.shape[0]
    cdef Py_ssize_t n = memb_flat.shape[1]
    cdef Py_ssize_t c, s, k
    cdef double tot, mv, full, cov, gini, p
    cdef Py_ssize_t child_idx, new_pred, correct_new, cov_hits
    cdef double acc_new, cci, purity

    cdef double best_cci = best_cci_init
    cdef double best_purity = INFINITY
    cdef int best_feature = -1
    cdef int best_fz = -1
    cdef double best_coverage = 0.0
    cdef int best_child_idx = 0

    cdef double *cnt = <double*>malloc(n_classes * sizeof(double))
    cdef double *probs = <double*>malloc(n_classes * sizeof(double))
    if cnt == NULL or probs == NULL:
        if cnt != NULL:
            free(cnt)
        if probs != NULL:
            free(probs)
        raise MemoryError()

    with nogil:
        for c in range(C):
            if legal[c] == 0:
                continue

            # First pass: total weight + weighted class counts.
            tot = 0.0
            for k in range(n_classes):
                cnt[k] = 0.0
            for s in range(n):
                full = memb_flat[c, s] * existing[s]
                if full != 0.0:
                    tot += full
                    cnt[y_int[s]] += full

            cov = tot / n
            if cov < cov_thresh:
                continue

            # Majority class (argmax of weighted counts, ties -> lowest index)
            # and normalised class probabilities. tot == 0 reproduces the
            # reference: uniform probs, child class 0, cci 0.0, purity +inf.
            if tot > 0.0:
                child_idx = 0
                mv = cnt[0]
                for k in range(1, n_classes):
                    if cnt[k] > mv:
                        mv = cnt[k]
                        child_idx = k
                for k in range(n_classes):
                    probs[k] = cnt[k] / tot
                gini = 1.0
                for k in range(n_classes):
                    p = cnt[k] / tot
                    gini -= p * p
                purity = gini
            else:
                child_idx = 0
                for k in range(n_classes):
                    probs[k] = 1.0 / n_classes
                purity = INFINITY

            # CCI: re-argmax in the additive soft-vote space per sample.
            if tot == 0.0:
                cci = 0.0
            else:
                correct_new = 0
                for s in range(n):
                    full = memb_flat[c, s] * existing[s]
                    new_pred = 0
                    mv = base_votes[s, 0] + full * probs[0]
                    for k in range(1, n_classes):
                        p = base_votes[s, k] + full * probs[k]
                        if p > mv:
                            mv = p
                            new_pred = k
                    if new_pred == y_int[s]:
                        correct_new += 1
                acc_new = (<double>correct_new) / n
                if acc_pre == 0.0:
                    cci = acc_new
                else:
                    cci = (acc_new - acc_pre) / acc_pre

            if coverage_weight > 0.0 and n_uncovered > 0:
                cov_hits = 0
                for s in range(n):
                    if uncovered[s] != 0:
                        full = memb_flat[c, s] * existing[s]
                        if full > 1e-3:
                            cov_hits += 1
                cci = cci + coverage_weight * ((<double>cov_hits) / n_uncovered)

            # Reference selection: CCI desc, then purity asc.
            if cci > best_cci:
                best_cci = cci
                best_feature = cand_feature[c]
                best_fz = cand_fz[c]
                best_coverage = cov
                best_child_idx = <int>child_idx
                best_purity = purity
            elif cci == best_cci and purity < best_purity:
                best_cci = cci
                best_feature = cand_feature[c]
                best_fz = cand_fz[c]
                best_coverage = cov
                best_child_idx = <int>child_idx
                best_purity = purity

    free(cnt)
    free(probs)
    return (best_feature, best_fz, best_cci, best_purity, best_coverage, best_child_idx)


def score_node_cci_legacy(double[:, ::1] memb_flat,
                          int[::1] cand_feature,
                          int[::1] cand_fz,
                          unsigned char[::1] legal,
                          double[::1] existing,
                          int[::1] y_int,
                          int[::1] skeleton_idx,
                          int n_classes,
                          double cov_thresh,
                          double acc_pre,
                          double best_cci_init,
                          unsigned char[::1] uncovered,
                          double coverage_weight,
                          int n_uncovered):
    """Legacy non-consistent CCI scorer.

    This is the original FERL split criterion used by ``ferl-original``:
    candidate membership above 0.01 hard-overrides the frozen skeleton
    prediction with the child majority class, then CCI is the relative accuracy
    improvement. It shares the reference tie-break: CCI desc, fuzzy-Gini asc,
    feature-major scan order.
    """
    cdef Py_ssize_t C = memb_flat.shape[0]
    cdef Py_ssize_t n = memb_flat.shape[1]
    cdef Py_ssize_t c, s, k
    cdef double tot, mv, full, cov, gini, p, acc_new, cci, purity
    cdef Py_ssize_t child_idx, new_pred, correct_new, cov_hits

    cdef double best_cci = best_cci_init
    cdef double best_purity = INFINITY
    cdef int best_feature = -1
    cdef int best_fz = -1
    cdef double best_coverage = 0.0
    cdef int best_child_idx = 0

    cdef double *cnt = <double*>malloc(n_classes * sizeof(double))
    if cnt == NULL:
        raise MemoryError()

    with nogil:
        for c in range(C):
            if legal[c] == 0:
                continue

            # The legacy branch often falls through to zero/near-zero CCI ties.
            # The reference computes total weight and then one masked sum per
            # class, so cnt sums can differ from total by a few ULPs and affect
            # the purity tie-break. Preserve that order here.
            tot = 0.0
            for s in range(n):
                tot += memb_flat[c, s] * existing[s]
            for k in range(n_classes):
                cnt[k] = 0.0
                for s in range(n):
                    if y_int[s] == k:
                        cnt[k] += memb_flat[c, s] * existing[s]

            cov = tot / n
            if cov < cov_thresh:
                continue

            if tot > 0.0:
                child_idx = 0
                mv = cnt[0]
                for k in range(1, n_classes):
                    if cnt[k] > mv:
                        mv = cnt[k]
                        child_idx = k
                gini = 1.0
                for k in range(n_classes):
                    p = cnt[k] / tot
                    gini -= p * p
                purity = gini
            else:
                child_idx = 0
                purity = INFINITY

            if tot == 0.0:
                cci = 0.0
            else:
                correct_new = 0
                for s in range(n):
                    full = memb_flat[c, s] * existing[s]
                    if full > 0.01:
                        new_pred = child_idx
                    else:
                        new_pred = skeleton_idx[s]
                    if new_pred == y_int[s]:
                        correct_new += 1
                acc_new = (<double>correct_new) / n
                if acc_pre == 0.0:
                    cci = acc_new
                else:
                    cci = (acc_new - acc_pre) / acc_pre

            if coverage_weight > 0.0 and n_uncovered > 0:
                cov_hits = 0
                for s in range(n):
                    if uncovered[s] != 0:
                        full = memb_flat[c, s] * existing[s]
                        if full > 1e-3:
                            cov_hits += 1
                cci = cci + coverage_weight * ((<double>cov_hits) / n_uncovered)

            if cci > best_cci:
                best_cci = cci
                best_feature = cand_feature[c]
                best_fz = cand_fz[c]
                best_coverage = cov
                best_child_idx = <int>child_idx
                best_purity = purity
            elif cci == best_cci and purity < best_purity:
                best_cci = cci
                best_feature = cand_feature[c]
                best_fz = cand_fz[c]
                best_coverage = cov
                best_child_idx = <int>child_idx
                best_purity = purity

    free(cnt)
    return (best_feature, best_fz, best_cci, best_purity, best_coverage, best_child_idx)


def selftest_sum(double[::1] a):
    """Sum a contiguous float64 array inside a ``nogil`` loop.

    Exercises exactly the primitives the real kernels rely on: a typed
    C-contiguous memoryview and a GIL-free reduction. Returns a Python float.
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t n = a.shape[0]
    cdef double s = 0.0
    with nogil:
        for i in range(n):
            s += a[i]
    return s


def selftest_argmax_rows(double[:, ::1] votes):
    """Row-wise argmax over a 2D float64 array, returned as an int32 array.

    Mirrors the per-sample argmax the prediction/CCI kernels will perform, and
    checks that we can allocate and fill a numpy result array from Cython.
    """
    cdef Py_ssize_t n = votes.shape[0]
    cdef Py_ssize_t k = votes.shape[1]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t best_j
    cdef double best_v
    out = np.empty(n, dtype=np.int32)
    cdef int[::1] out_mv = out
    with nogil:
        for i in range(n):
            best_j = 0
            best_v = votes[i, 0]
            for j in range(1, k):
                if votes[i, j] > best_v:
                    best_v = votes[i, j]
                    best_j = j
            out_mv[i] = <int>best_j
    return out
