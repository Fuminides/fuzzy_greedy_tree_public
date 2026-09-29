"""FuzzyCARTFast: a Cython-accelerated drop-in for the reference FuzzyCART.

Design goal: identical tree structure to ``tree_learning.FuzzyCART``, with the
hot per-candidate split scoring moved into the compiled ``_kernels`` module.

To guarantee structural equivalence, this class *subclasses* the reference and
overrides only two methods:

* ``_build_cci_context`` - augments the reference scan context with the flat,
  C-contiguous arrays the kernel consumes (built once per split scan).
* ``_node_cci_checks``   - replaces the Python inner candidate loop with a call
  to ``_kernels.score_node_cci`` for the default consistent-CCI / soft setup,
  and otherwise defers to the reference implementation.

Everything else (root construction, the greedy growth loop, ``_split_node``,
prediction and pruning) is inherited unchanged, so the only thing that can
differ from the reference is the arithmetic of the split score - which the
kernel reproduces in IEEE double precision.
"""
from __future__ import annotations

import os
import numpy as np
import ex_fuzzy.fuzzy_sets as fs

import ferl.core.tree_learning as _tl
from ferl.core.tree_learning import FuzzyCART
from ferl_fast import _kernels

# ex_fuzzy.fuzzy_sets.trapezoidal_membership default epsilon (10E-5).
_TRAP_EPS = 10e-5

# OpenMP thread selection. Parallelism only pays off on large batches; below the
# work threshold the kernels run serial (num_threads=1) so the many small calls
# during training don't pay thread-spawn overhead. Capped to leave headroom on
# shared machines. Override per-instance via FuzzyCARTFast.max_threads.
_CPU = os.cpu_count() or 1
_DEFAULT_MAX_THREADS = min(8, _CPU)
_PARALLEL_WORK_MIN = 300_000  # elementary ops (n * inner) below which we stay serial


def _pick_threads(work, cap):
    if work < _PARALLEL_WORK_MIN or cap <= 1:
        return 1
    return cap


class FuzzyCARTFast(FuzzyCART):
    """Drop-in replacement for FuzzyCART with a compiled split-scoring kernel."""

    # Max OpenMP threads for the parallel prediction/membership kernels. Set to
    # 1 to force serial. Defaults to a capped fraction of available cores.
    max_threads = _DEFAULT_MAX_THREADS
    exact_low_gain_ties = False

    def fit(self, X: np.array, y: np.array, patience: int = 3):
        for attr in (
            "_class_to_fast_idx", "_cand_layout_cache", "_feat_offsets_cache",
            "_memb_tables_cache", "_last_partition_signature",
            "_ds_layout_cache",
        ):
            if hasattr(self, attr):
                delattr(self, attr)
        return super().fit(X, y, patience=patience)

    def _invalidate_leaf_cache(self):
        super()._invalidate_leaf_cache()
        if hasattr(self, "_ds_layout_cache"):
            delattr(self, "_ds_layout_cache")

    def _prune_subtree(self, node: dict, X: np.array, y: np.array):
        super()._prune_subtree(node, X, y)
        self._invalidate_leaf_cache()

    def _restore_tree(self, tree_backup: dict):
        super()._restore_tree(tree_backup)
        self._invalidate_leaf_cache()

    def _partition_signature(self):
        """Identity signature for cached arrays derived from fuzzy partitions."""
        return tuple(tuple(id(fset) for fset in fuzzy_var)
                     for fuzzy_var in self.fuzzy_partitions)

    def _candidate_layout(self):
        """Per-candidate (feature, fuzzy_set) index arrays, feature-major.

        Depends only on the partition sizes, so it is computed once and cached.
        Row ``c`` of the flattened membership matrix corresponds to
        ``(cand_feature[c], cand_fz[c])``.
        """
        sig = self._partition_signature()
        cached = getattr(self, "_cand_layout_cache", None)
        if cached is not None and cached[0] == sig:
            return cached[1]
        feats = []
        fzs = []
        for f, fuzzy_var in enumerate(self.fuzzy_partitions):
            for fz in range(len(fuzzy_var)):
                feats.append(f)
                fzs.append(fz)
        layout = (
            np.asarray(feats, dtype=np.int32),
            np.asarray(fzs, dtype=np.int32),
        )
        self._cand_layout_cache = (sig, layout)
        return layout

    def _get_cached_memberships(self, X: np.array) -> dict:
        """Kernel-backed version of the reference per-scan membership cache.

        Returns the same structure as the reference (dict feature_idx ->
        (n_fuzzy_sets, n_samples)) so scoring is unchanged, but computes every
        candidate's membership in one C call instead of per-set ex_fuzzy calls.
        """
        sig = self._partition_signature()
        if (self._last_X_shape == X.shape and
                getattr(self, "_last_partition_signature", None) == sig and
                len(self._membership_cache) > 0):
            return self._membership_cache

        offsets = self._feature_offsets()
        C = int(offsets[-1])
        full = self._kernel_memberships(X, np.arange(C))  # (n, C)
        cache = {}
        for f in range(len(self.fuzzy_partitions)):
            block = full[:, int(offsets[f]):int(offsets[f + 1])].T  # (n_fz, n)
            cache[f] = np.ascontiguousarray(block)
        self._membership_cache = cache
        self._last_X_shape = X.shape
        self._last_partition_signature = sig
        return cache

    def _membership_tables(self):
        """Per-global-candidate membership parameters for the C kernel.

        Built once (partitions are fixed). For candidate g = offsets[f] + fz:
          kind[g] = 1 -> equality test ``x == pa[g]`` (categoricalFS, or a
                         singleton trapezoid where a == d);
          kind[g] = 2 -> learned ramp, direction='below';
          kind[g] = 3 -> learned ramp, direction='above';
          kind[g] = 0 -> trapezoid with (pa, pb, pc, pd) = (a, b', c, d') where
                         b' / d' carry ex_fuzzy's b+=eps / d+=eps adjustments.
        Bit-exactness against ex_fuzzy is checked in _validate_fast / the
        membership self-test before this path is trusted.
        """
        sig = self._partition_signature()
        cached = getattr(self, "_memb_tables_cache", None)
        if cached is not None and cached[0] == sig:
            return cached[1]

        feat, kind, pa, pb, pc, pd = [], [], [], [], [], []
        for f, fuzzy_var in enumerate(self.fuzzy_partitions):
            for fset in fuzzy_var:
                feat.append(f)
                if isinstance(fset, _tl.LearnedRampSet):
                    kind.append(2 if fset.direction == "below" else 3)
                    pa.append(float(fset.center)); pb.append(float(fset.h)); pc.append(0.0); pd.append(0.0)
                    continue
                params = getattr(fset, "membership_parameters", None)
                if isinstance(fset, fs.categoricalFS) or params is None:
                    kind.append(1)
                    pa.append(float(fset.category)); pb.append(0.0); pc.append(0.0); pd.append(0.0)
                    continue
                a, b, c, d = (float(v) for v in params)
                if a == d:  # singleton trapezoid -> equality on a
                    kind.append(1)
                    pa.append(a); pb.append(0.0); pc.append(0.0); pd.append(0.0)
                else:
                    if b == a:
                        b = b + _TRAP_EPS
                    if c == d:
                        d = d + _TRAP_EPS
                    kind.append(0)
                    pa.append(a); pb.append(b); pc.append(c); pd.append(d)

        tables = dict(
            feat=np.asarray(feat, dtype=np.int32),
            kind=np.asarray(kind, dtype=np.uint8),
            pa=np.asarray(pa, dtype=np.float64),
            pb=np.asarray(pb, dtype=np.float64),
            pc=np.asarray(pc, dtype=np.float64),
            pd=np.asarray(pd, dtype=np.float64),
        )
        self._memb_tables_cache = (sig, tables)
        return tables

    def _kernel_memberships(self, X, used_global, observed_mask=None):
        """Membership of X to each candidate in ``used_global`` -> (n, len(used)).

        ``used_global`` is a sequence of global candidate indices. Uses the C
        ``fill_memberships`` kernel instead of per-set ex_fuzzy Python calls.
        If ``observed_mask`` is given (and not all-True) the masked kernel is
        used: unobserved features contribute uniform membership 1/n_partitions.
        """
        t = self._membership_tables()
        idx = np.asarray(used_global, dtype=np.intp)
        feat = np.ascontiguousarray(t['feat'][idx])
        kind = np.ascontiguousarray(t['kind'][idx])
        pa = np.ascontiguousarray(t['pa'][idx])
        pb = np.ascontiguousarray(t['pb'][idx])
        pc = np.ascontiguousarray(t['pc'][idx])
        pd = np.ascontiguousarray(t['pd'][idx])
        out = np.empty((X.shape[0], idx.shape[0]), dtype=np.float64)
        Xc = np.ascontiguousarray(X, dtype=np.float64)
        nt = _pick_threads(out.size, self.max_threads)
        if observed_mask is None:
            _kernels.fill_memberships(Xc, feat, kind, pa, pb, pc, pd, out, nt)
        else:
            offs = self._feature_offsets()
            nparts = (offs[1:] - offs[:-1]).astype(np.float64)  # per feature
            uniform = np.ascontiguousarray(1.0 / nparts[feat])
            obs = np.ascontiguousarray(observed_mask.astype(np.uint8))
            _kernels.fill_memberships_masked(
                Xc, feat, kind, pa, pb, pc, pd, obs, uniform, out, nt)
        return out

    def _feature_offsets(self):
        """Cumulative candidate offsets per feature (feature-major flat index).

        Candidate index of ``(feature f, fuzzy set fz)`` is ``offsets[f] + fz``;
        ``offsets[-1]`` is the total candidate count ``C``.
        """
        sig = self._partition_signature()
        cached = getattr(self, "_feat_offsets_cache", None)
        if cached is not None and cached[0] == sig:
            return cached[1]
        offs = [0]
        for fuzzy_var in self.fuzzy_partitions:
            offs.append(offs[-1] + len(fuzzy_var))
        offs = np.asarray(offs, dtype=np.int64)
        self._feat_offsets_cache = (sig, offs)
        return offs

    def _label_to_index(self, label):
        """Class-index lookup with ``-1`` for the root's construction sentinel."""
        if not hasattr(self, "_class_to_fast_idx"):
            self._class_to_fast_idx = {c: i for i, c in enumerate(self.classes_)}
        return self._class_to_fast_idx.get(label, -1)

    def _encode_labels(self, y):
        if not hasattr(self, "_class_to_fast_idx"):
            self._class_to_fast_idx = {c: i for i, c in enumerate(self.classes_)}
        return np.asarray([self._class_to_fast_idx[v] for v in y], dtype=np.int32)

    def _learned_best_cut_fast(self, x, y_int, w, parent, W):
        order = np.argsort(x, kind="mergesort")
        xs = np.ascontiguousarray(np.asarray(x, dtype=np.float64)[order])
        ys = np.ascontiguousarray(np.asarray(y_int, dtype=np.int32)[order])
        ws = np.ascontiguousarray(np.asarray(w, dtype=np.float64)[order])
        gain, thr, ok = _kernels.learned_best_cut_sorted(
            xs, ys, ws, len(self.classes_), float(parent), float(W))
        return float(gain), (float(thr) if ok else None)

    def _node_kernel_layout(self, nodes, X, observed_mask=None, missing_policy="uniform"):
        """Build node-path matrices and metadata for prediction kernels.

        Parameters
        ----------
        nodes : list[dict]
            Entries from ``_extract_all_nodes`` in the exact order the caller
            wants the kernel to process.
        missing_policy : {"uniform", "one", "none"}
            ``uniform`` matches ``_predict_proba_all_nodes``. ``one`` is the
            legacy winner path's missing-feature behavior. ``none`` assumes all
            features are observed.
        """
        N, K, C = X.shape[0], len(nodes), len(self.classes_)
        names = [nd['name'] for nd in nodes]
        offsets = self._feature_offsets()
        used = sorted({int(offsets[f]) + fz
                       for nd in nodes
                       for f, fz in zip(nd['path_features'], nd['path_fuzzy_sets'])})
        remap = {g: i for i, g in enumerate(used)}

        if used:
            if missing_policy == "uniform":
                fully = observed_mask is None or bool(np.all(observed_mask))
                memb = np.ascontiguousarray(self._kernel_memberships(
                    X, used, None if fully else observed_mask))
            elif missing_policy == "one":
                if observed_mask is not None and not bool(np.all(observed_mask)):
                    raise ValueError("missing_policy='one' is handled by the reference fallback")
                memb = np.ascontiguousarray(self._kernel_memberships(X, used, None))
            else:
                memb = np.ascontiguousarray(self._kernel_memberships(X, used, None))
        else:
            memb = np.empty((N, 0), dtype=np.float64)

        cand, starts = [], [0]
        probs = np.zeros((K, C), dtype=np.float64)
        pred_idx = np.empty(K, dtype=np.int32)
        path_lengths = np.empty(K, dtype=np.int32)
        for k, nd in enumerate(nodes):
            for f, fz in zip(nd['path_features'], nd['path_fuzzy_sets']):
                cand.append(remap[int(offsets[f]) + fz])
            starts.append(len(cand))
            node = self.node_dict_access.get(nd['name'])
            cp = node.get('class_probabilities') if node is not None else None
            probs[k] = cp if (cp is not None and len(cp) == C) else np.ones(C) / C
            pred_idx[k] = self._label_to_index(nd['prediction'])
            path_lengths[k] = nd['path_length']

        cand = np.asarray(cand, dtype=np.int32)
        starts = np.asarray(starts, dtype=np.int32)
        M = np.empty((N, K), dtype=np.float64)
        nt = _pick_threads(N * max(K, 1), self.max_threads)
        _kernels.node_membership_matrix(memb, cand, starts, M, nt)

        name_to_col = {name: i for i, name in enumerate(names)}
        child_cols, child_starts, is_internal = [], [0], np.zeros(K, dtype=np.uint8)
        for name in names:
            children = [name_to_col[ch] for ch in self._get_node_children_names(name)
                        if ch in name_to_col]
            if children:
                is_internal[len(child_starts) - 1] = 1
                child_cols.extend(children)
            child_starts.append(len(child_cols))

        return dict(
            M=np.ascontiguousarray(M),
            probs=np.ascontiguousarray(probs),
            pred_idx=np.ascontiguousarray(pred_idx),
            child_cols=np.asarray(child_cols, dtype=np.int32),
            child_starts=np.asarray(child_starts, dtype=np.int32),
            is_internal=np.ascontiguousarray(is_internal),
            path_lengths=path_lengths,
            names=names,
        )

    def _ds_node_layout(self):
        """Static non-root node layout for activation and DS kernels."""
        nodes = [nd for nd in self._extract_all_nodes() if nd['path_length'] > 0]
        names = tuple(nd['name'] for nd in nodes)
        sig = (self._partition_signature(), names, len(self.classes_))
        cached = getattr(self, "_ds_layout_cache", None)
        if cached is not None and cached.get("sig") == sig:
            return cached

        K, C = len(nodes), len(self.classes_)
        if K == 0:
            layout = dict(
                sig=sig,
                nodes=nodes,
                names=[],
                used=np.empty(0, dtype=np.int64),
                cand=np.empty(0, dtype=np.int32),
                starts=np.zeros(1, dtype=np.int32),
                cons=np.zeros((0, C), dtype=np.float64),
                support=np.empty(0, dtype=np.float64),
                leaf_keep=np.empty(0, dtype=bool),
                hybrid_cols=np.empty(0, dtype=np.int32),
                hybrid_lstart=np.zeros(1, dtype=np.int32),
                parent=np.empty(0, dtype=np.int32),
            )
            self._ds_layout_cache = layout
            return layout

        offsets = self._feature_offsets()
        used = sorted({int(offsets[f]) + fz
                       for nd in nodes
                       for f, fz in zip(nd['path_features'], nd['path_fuzzy_sets'])})
        remap = {g: i for i, g in enumerate(used)}
        cand, starts = [], [0]
        cons = np.zeros((K, C), dtype=np.float64)
        support = np.zeros(K, dtype=np.float64)
        names_list = list(names)
        for k, nd in enumerate(nodes):
            for f, fz in zip(nd['path_features'], nd['path_fuzzy_sets']):
                cand.append(remap[int(offsets[f]) + fz])
            starts.append(len(cand))
            node = self.node_dict_access[nd['name']]
            cp = node.get('class_probabilities')
            cons[k] = cp if (cp is not None and len(cp) == C) else np.ones(C) / C
            support[k] = node['coverage'] * self._n_train

        leaf_keep = np.asarray([not self._node_has_children(n) for n in names_list], dtype=bool)
        hybrid_cols, hybrid_lstart = self._hybrid_chains(names_list)
        parent = np.full(K, -1, dtype=np.int32)
        for kk, nk in enumerate(names_list):
            blen = -1
            for j, nj in enumerate(names_list):
                if j != kk and nk.startswith(nj + "_") and len(nj) > blen:
                    parent[kk], blen = j, len(nj)

        layout = dict(
            sig=sig,
            nodes=nodes,
            names=names_list,
            used=np.asarray(used, dtype=np.int64),
            cand=np.asarray(cand, dtype=np.int32),
            starts=np.asarray(starts, dtype=np.int32),
            cons=np.ascontiguousarray(cons),
            support=np.ascontiguousarray(support),
            leaf_keep=leaf_keep,
            hybrid_cols=hybrid_cols,
            hybrid_lstart=hybrid_lstart,
            parent=np.ascontiguousarray(parent),
        )
        self._ds_layout_cache = layout
        return layout

    def _predict_proba_all_nodes(self, X: np.array, observed_mask: np.array,
                                 epsilon: float = 1e-6, return_votes: bool = False):
        """Kernel-backed all-node probability prediction.

        The default ``soft`` mode keeps the original specialised accumulator.
        ``soft_gate``, ``hard_gate`` and the hard non-gated probability path used
        by ``winner`` use the generic gated vote kernel.
        """
        mode = getattr(self, 'prediction_mode', 'soft_gate')
        use_soft = mode in ('soft_gate', 'soft')
        gate_internal = mode in ('soft_gate', 'hard_gate')

        if mode not in ('soft', 'soft_gate', 'hard_gate', 'winner'):
            return super()._predict_proba_all_nodes(X, observed_mask, epsilon, return_votes)
        fully_observed = bool(np.all(observed_mask))

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        offsets = self._feature_offsets()

        if mode != 'soft':
            nodes = self._extract_all_nodes()
            if mode == 'winner':
                nodes = [nd for nd in nodes if nd['path_length'] > 0]
                mode_code = 3
            elif use_soft:
                mode_code = 1
            else:
                mode_code = 2

            if len(nodes) == 0:
                if return_votes:
                    return (np.zeros((n_samples, n_classes)), np.zeros(n_samples))
                return np.full((n_samples, n_classes), 1.0 / n_classes)

            # Reference accumulation order sorts the node-membership dictionary
            # by prediction before applying gate/hard/soft semantics.
            nodes = sorted(nodes, key=lambda d: d['prediction'])
            layout = self._node_kernel_layout(nodes, X, observed_mask, missing_policy="uniform")
            nt = _pick_threads(n_samples * len(nodes) * n_classes, self.max_threads)
            class_mem, total = _kernels.accumulate_node_votes(
                layout['M'], layout['probs'], layout['pred_idx'],
                layout['child_cols'], layout['child_starts'], layout['is_internal'],
                mode_code, float(epsilon), n_classes, nt)

            if return_votes:
                return class_mem, total

            nonzero = total > 0
            proba = np.zeros((n_samples, n_classes))
            proba[nonzero] = class_mem[nonzero] / total[nonzero, np.newaxis]
            proba[~nonzero] = 1.0 / n_classes
            return proba

        # Non-root nodes in the reference accumulation order: insertion order is
        # _extract_all_nodes (path-length sorted), then a stable sort by node
        # prediction -- matching the reference's node_items.sort.
        all_nodes = self._extract_all_nodes()
        nz = [nd for nd in all_nodes if nd['path_length'] > 0]
        nz.sort(key=lambda d: d['prediction'])  # stable

        if len(nz) == 0:
            # Root-only tree: reference skips root in soft mode -> zero votes.
            if return_votes:
                return (np.zeros((n_samples, n_classes)), np.zeros(n_samples))
            return np.full((n_samples, n_classes), 1.0 / n_classes)

        # Compute membership only for the (feature, fuzzy set) candidates that
        # actually occur on some node path. This keeps the kernel's membership
        # work <= the reference (which evaluates per node, with redundancy)
        # instead of eagerly evaluating all n_features * n_partitions sets.
        used = sorted({int(offsets[f]) + fz
                       for nd in nz
                       for f, fz in zip(nd['path_features'], nd['path_fuzzy_sets'])})
        remap = {g: i for i, g in enumerate(used)}

        # Membership of all used candidates in one C call (vs per-set ex_fuzzy).
        # Pass the mask only when features are actually missing.
        memb = np.ascontiguousarray(self._kernel_memberships(
            X, used, None if fully_observed else observed_mask))

        cand = []
        starts = [0]
        probs = []
        for nd in nz:
            for f, fz in zip(nd['path_features'], nd['path_fuzzy_sets']):
                cand.append(remap[int(offsets[f]) + fz])
            starts.append(len(cand))
            probs.append(self.node_dict_access[nd['name']]['class_probabilities'])

        cand = np.asarray(cand, dtype=np.int32)
        starts = np.asarray(starts, dtype=np.int32)
        probs = np.ascontiguousarray(np.asarray(probs, dtype=np.float64))

        nt = _pick_threads(n_samples * len(nz) * n_classes, self.max_threads)
        class_mem, total = _kernels.accumulate_soft_votes(
            memb, cand, starts, probs, n_classes, nt)

        if return_votes:
            return class_mem, total

        # Identical normalisation to the reference.
        nonzero = total > 0
        proba = np.zeros((n_samples, n_classes))
        proba[nonzero] = class_mem[nonzero] / total[nonzero, np.newaxis]
        proba[~nonzero] = 1.0 / n_classes
        return proba

    def _predict_all_nodes(self, X: np.array, observed_mask: np.array,
                           epsilon: float = 1e-6):
        """Kernel-backed winner prediction for fully observed inputs."""
        if observed_mask is not None and not bool(np.all(observed_mask)):
            return super()._predict_all_nodes(X, observed_mask, epsilon)

        nodes = self._extract_all_nodes()
        if len(nodes) == 0:
            return super()._predict_all_nodes(X, observed_mask, epsilon)

        layout = self._node_kernel_layout(nodes, X, observed_mask, missing_policy="one")
        leaves = [i for i, name in enumerate(layout['names'])
                  if not self._node_has_children(name)]
        internal = [i for i, name in enumerate(layout['names'])
                    if self._node_has_children(name)]
        internal.sort(key=lambda i: layout['path_lengths'][i], reverse=True)
        order = np.asarray(leaves + internal, dtype=np.int32)

        nt = _pick_threads(X.shape[0] * len(nodes), self.max_threads)
        pred_idx, memberships, node_cols = _kernels.select_best_nodes(
            layout['M'], layout['pred_idx'], layout['child_cols'],
            layout['child_starts'], layout['is_internal'], order,
            float(epsilon), nt)

        if np.issubdtype(self.classes_.dtype, np.number):
            predictions = np.full(X.shape[0], -1, dtype=self.classes_.dtype)
        else:
            predictions = np.full(X.shape[0], -1, dtype=object)
        valid = pred_idx >= 0
        predictions[valid] = self.classes_[pred_idx[valid]]
        paths = np.empty(X.shape[0], dtype=object)
        for i, col in enumerate(node_cols):
            paths[i] = layout['names'][col] if col >= 0 else 'root'
        return predictions, memberships, paths

    def node_activation_matrix(self, X: np.array, observed_mask: np.array = None,
                               membership_floor: float = 0.0):
        """Kernel-backed per-node firing matrix (reference node order preserved).

        Node order is ``_extract_all_nodes`` (path-length sorted), non-root --
        NOT the prediction-sorted order used by soft inference -- so M / cons
        align with the reference for the DS combination.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if observed_mask is None:
            observed_mask = np.ones_like(X, dtype=bool)

        layout = self._ds_node_layout()
        N, K, C = X.shape[0], len(layout['names']), len(self.classes_)
        if K == 0:
            return np.zeros((N, 0)), np.zeros((0, C)), []

        fully = bool(np.all(observed_mask))
        memb = np.ascontiguousarray(
            self._kernel_memberships(X, layout['used'], None if fully else observed_mask))
        if membership_floor > 0.0:
            memb = np.maximum(memb, membership_floor)

        M = np.empty((N, K), dtype=np.float64)
        nt = _pick_threads(N * K, self.max_threads)
        _kernels.node_membership_matrix(memb, layout['cand'], layout['starts'], M, nt)
        return M, layout['cons'].copy(), list(layout['names'])

    def predict_ds(self, X: np.array, observed_mask: np.array = None, leaves_only: bool = False,
                   rule: str = "dempster", reliability_k: float = None, prior_strength: float = None,
                   reliability_vec: np.array = None, top_p: float = None):
        """Kernel-backed Dempster-Shafer combination (no (N,K,C) tensor).

        Faithful port of ``FuzzyCART.predict_ds``: the cheap O(K*C) prep (top-p
        nucleus routing, reliability / Dirichlet / learned discounts) runs in
        numpy exactly as the reference; the heavy (N,K,C) combination is the only
        part in C. Every rule -- ``dempster``, ``cautious``, ``hybrid``,
        ``incremental`` / ``incremental_local`` -- has a streaming kernel, so this
        no longer falls back to the reference combination. The prep mirrors the
        reference line-for-line; re-sync if the reference's prep changes.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        M, cons, names = self.node_activation_matrix(X, observed_mask)
        layout = self._ds_node_layout()
        support = layout['support']
        parent = layout['parent']
        if leaves_only and M.shape[1] > 0:
            keep = layout['leaf_keep']
            M, cons = M[:, keep], cons[keep]
            names = [n for n, kp in zip(names, keep) if kp]
            support = support[keep]
            parent = None

        if top_p is not None and M.shape[1] > 0:
            # nucleus routing: keep each node's top classes until the cumulative
            # consequent reaches top_p, route the tail to Theta (firing discount).
            order = np.argsort(-cons, axis=1)
            sc = np.take_along_axis(cons, order, axis=1)
            before = np.cumsum(sc, axis=1) - sc
            kept = np.zeros_like(cons)
            np.put_along_axis(kept, order, np.where(before < top_p, sc, 0.0), axis=1)
            rtp = kept.sum(1)
            cons = kept / np.clip(rtp[:, None], 1e-12, None)
            M = M * rtp[None, :]

        k = self.reliability_k if reliability_k is None else reliability_k
        M_raw = M.copy()                                   # firing only (pre-reliability)
        r_vec = np.ones(M.shape[1])                        # per-node reliability rho_n
        if reliability_vec is not None and M.shape[1] > 0:
            r_vec = np.asarray(reliability_vec, dtype=float)
            M = M * r_vec[None, :]
        elif prior_strength is not None and M.shape[1] > 0:
            a0, Cc = prior_strength, cons.shape[1]
            counts = cons * support[:, None]
            cons = (a0 + counts) / (Cc * a0 + support[:, None])
            r_vec = support / (Cc * a0 + support)
            M = M * r_vec[None, :]
        elif k is not None and M.shape[1] > 0:
            r_vec = support / (support + k)
            M = M * r_vec[None, :]

        C = len(self.classes_)
        if M.shape[1] == 0:  # no rules -> total ignorance (matches reference)
            ign = np.ones(X.shape[0])
            betp = np.full((X.shape[0], C), 1.0 / C)
            return betp, np.zeros((X.shape[0], C)), np.ones((X.shape[0], C)), ign

        M = np.ascontiguousarray(M, dtype=np.float64)
        cons = np.ascontiguousarray(cons, dtype=np.float64)
        nt = _pick_threads(M.shape[0] * M.shape[1] * C, self.max_threads)

        if rule in ("incremental", "incremental_local"):
            return self._predict_ds_incremental(
                M_raw, cons, names, r_vec, rule, nt, support=support, parent=parent)
        if rule == "cautious":
            return _kernels.predict_ds_combine_cautious(M, cons, nt)
        if rule == "hybrid":
            if leaves_only:
                cols, lstart = self._hybrid_chains(names)
            else:
                cols, lstart = layout['hybrid_cols'], layout['hybrid_lstart']
            return _kernels.predict_ds_combine_hybrid(M, cons, cols, lstart, nt)
        return _kernels.predict_ds_combine(M, cons, nt)

    def _hybrid_chains(self, names):
        """CSR (cols, lstart) of each leaf chain's ancestor+self column indices,
        in the reference's leaf order with ascending ancestor columns -- the
        structure the hybrid kernel consumes. Mirrors the leaf/ancestor selection
        in ``FuzzyCART.predict_ds``'s hybrid branch."""
        leaves = [i for i, n in enumerate(names)
                  if not any(o != n and o.startswith(n + "_") for o in names)]
        cols, lstart = [], [0]
        for li in leaves:
            ln = names[li]
            cols.extend(j for j, n in enumerate(names) if ln == n or ln.startswith(n + "_"))
            lstart.append(len(cols))
        return (np.asarray(cols, dtype=np.int32), np.asarray(lstart, dtype=np.int32))

    def _predict_ds_incremental(self, M_raw, cons, names, r_vec, rule, nt,
                                support=None, parent=None):
        """Residual (incremental) DS evidence on the fast path.

        Ports the O(K*C)/O(K**2) prep of ``FuzzyCART.predict_ds``'s incremental
        branch verbatim (residual extraction, validity test, parent lattice,
        local conditional firing), then streams the firing-discounted Dempster
        product through the ``predict_ds_combine_masses`` kernel. ``r_vec`` is the
        already-discounted per-node reliability (the beta=10 default applies only
        when it is all ones, exactly as the reference).
        """
        local = (rule == "incremental_local")
        K = M_raw.shape[1]

        r_inc = r_vec
        if np.allclose(r_inc, 1.0):                        # need t_n>0 to divide
            supp = support
            if supp is None:
                supp = np.array([self.node_dict_access[n]['coverage'] for n in names]) * self._n_train
            r_inc = supp / (supp + 10.0)                   # default beta=10
        t = np.clip(1.0 - r_inc, 1e-9, 1.0)                # (K,) Theta mass
        a = r_inc[:, None] * cons                          # (K,C) singleton mass
        g = a + t[:, None]                                 # (K,C) base commonality

        # immediate active parent of each node (longest name that is a prefix)
        if parent is None:
            parent = np.full(K, -1, dtype=int)
            for kk, nk in enumerate(names):
                blen = -1
                for j, nj in enumerate(names):
                    if j != kk and nk.startswith(nj + "_") and len(nj) > blen:
                        parent[kk], blen = j, len(nj)
        else:
            parent = np.asarray(parent, dtype=int)

        a_eff = a.copy()                                   # (K,C) effective singleton mass
        t_eff = t.copy()                                   # (K,)  effective Theta mass
        is_residual = np.zeros(K, dtype=bool)
        n_nonroot = int((parent >= 0).sum())
        n_invalid = 0
        for kk in range(K):
            p = parent[kk]
            if p < 0:                                      # root component: full mass
                continue
            q_res = g[kk] / np.clip(g[p], 1e-12, None)     # (C,)
            q_res_th = float(np.clip(t[kk] / t[p], 0.0, 1.0))
            a_res = q_res - q_res_th
            if np.all(a_res >= -1e-9):                      # additive refinement
                a_eff[kk] = np.clip(a_res, 0.0, None)
                t_eff[kk] = q_res_th
                is_residual[kk] = True
            else:                                          # exception -> keep full mass
                n_invalid += 1
        self._last_incremental_diag = {
            'n_nonroot': n_nonroot, 'n_invalid': n_invalid,
            'residual_invalid_rate': (n_invalid / n_nonroot) if n_nonroot else 0.0,
        }

        mu = M_raw.copy()                                  # (N,K) per-sample firing
        if local and is_residual.any():
            pr = parent[is_residual]
            mu[:, is_residual] = M_raw[:, is_residual] / np.clip(M_raw[:, pr], 1e-12, None)
            mu = np.clip(mu, 0.0, 1.0)

        mu = np.ascontiguousarray(mu, dtype=np.float64)
        a_eff = np.ascontiguousarray(a_eff, dtype=np.float64)
        t_eff = np.ascontiguousarray(t_eff, dtype=np.float64)
        return _kernels.predict_ds_combine_masses(mu, a_eff, t_eff, nt)

    def _learned_cci_candidates(self, node, ctx):
        """Fast performance-mode learned split search."""
        if node.get('_learned_exhausted'):
            node['aux_purity_cache'] = {'cci': 0.0, 'feature': -1, 'fuzzy_set': -1,
                'coverage': 0.0, 'split_criterion': 0.0, 'child_decision': None, 'purity': 0.0}
            return float('-inf'), float('inf')
        if node.get('_learned_aux') is not None:
            node['aux_purity_cache'] = node['_learned_aux']
            return node['_learned_aux']['cci'], node['_learned_aux']['purity']
        if "_kernel" not in ctx or ctx["_kernel"].get("base_votes") is None:
            return super()._learned_cci_candidates(node, ctx)

        X_sample = np.ascontiguousarray(ctx['X_sample'], dtype=np.float64)
        sample_indices = ctx['sample_indices']
        kd = ctx["_kernel"]
        y_int = kd["y_int"]
        base_votes = kd["base_votes"]
        uncovered = kd["uncovered"]
        coverage_weight = kd["coverage_weight"]
        n_uncovered = kd["n_uncovered"]

        if sample_indices is not None:
            existing = node['existing_membership'][sample_indices]
        else:
            existing = node['existing_membership']
        existing = np.ascontiguousarray(existing, dtype=np.float64)

        n, n_features = X_sample.shape
        best_cci = float('-inf') if self.tree_rules <= 3 else 0.0
        best_purity = float('inf')
        best_feature, best_center, best_h, best_coverage = -1, 0.0, 0.0, 0.0
        child_decision = node['prediction']

        W = float(existing.sum())
        region = existing > 1e-6
        n_eff = int(region.sum())
        if W > 1e-9 and n_eff >= 4:
            X_region = X_sample[region]
            y_region = np.ascontiguousarray(y_int[region], dtype=np.int32)
            wm = np.ascontiguousarray(existing[region], dtype=np.float64)
            Wm = float(wm.sum())
            pwm = wm / Wm
            counts = np.bincount(y_region, weights=wm, minlength=len(self.classes_))
            p = counts / Wm
            parent = float(1.0 - np.sum(p ** 2))

            valid = np.zeros(n_features, dtype=np.uint8)
            centers = np.zeros(n_features, dtype=np.float64)
            hs = np.zeros(n_features, dtype=np.float64)
            ones = np.ones(n_eff, dtype=np.float64)

            for feature in range(n_features):
                xfm = np.ascontiguousarray(X_region[:, feature], dtype=np.float64)
                _gain, thr = self._learned_best_cut_fast(xfm, y_region, wm, parent, Wm)
                if thr is None:
                    continue

                if self.learned_width == 'bootstrap':
                    thetas = []
                    for _ in range(self.learned_n_boot):
                        idx = np.random.choice(n_eff, n_eff, p=pwm)
                        _g, tb = self._learned_best_cut_fast(
                            xfm[idx], y_region[idx], ones, 1.0, float(n_eff))
                        if tb is not None:
                            thetas.append(tb)
                    if len(thetas) < 2:
                        continue
                    center, h = float(np.mean(thetas)), float(np.std(thetas))
                else:
                    mean = float((wm * xfm).sum() / Wm)
                    std = float(np.sqrt((wm * (xfm - mean) ** 2).sum() / Wm))
                    center, h = float(thr), float(self.learned_width) * std
                h = max(h, 1e-3 * (float(xfm.max() - xfm.min()) + 1e-9))
                valid[feature] = 1
                centers[feature] = center
                hs[feature] = h

            if valid.any():
                (bf, _bdir, bcci, bpur, bcov, bchild) = _kernels.score_learned_ramps(
                    X_sample,
                    np.ascontiguousarray(valid),
                    np.ascontiguousarray(centers),
                    np.ascontiguousarray(hs),
                    existing,
                    np.ascontiguousarray(y_int, dtype=np.int32),
                    np.ascontiguousarray(base_votes, dtype=np.float64),
                    len(self.classes_),
                    float(self.coverage_threshold),
                    float(kd["acc_pre"]),
                    float(best_cci),
                    uncovered,
                    float(coverage_weight),
                    int(n_uncovered),
                )
                if bf != -1:
                    best_cci, best_purity = float(bcci), float(bpur)
                    best_feature = int(bf)
                    best_center = float(centers[best_feature])
                    best_h = float(hs[best_feature])
                    best_coverage = float(bcov)
                    child_decision = self.classes_[int(bchild)]

        if best_feature != -1:
            aux = {
                'cci': best_cci, 'feature': best_feature, 'fuzzy_set': -1,
                'learned': True, 'learned_center': best_center, 'learned_h': best_h,
                'coverage': best_coverage, 'split_criterion': best_cci,
                'child_decision': child_decision, 'purity': best_purity,
            }
        else:
            aux = {
                'cci': 0.0, 'feature': -1, 'fuzzy_set': -1, 'coverage': 0.0,
                'split_criterion': 0.0, 'child_decision': None, 'purity': 0.0,
            }
        node['aux_purity_cache'] = aux
        node['_learned_aux'] = aux
        return best_cci, best_purity

    def _build_cci_context(self, X: np.array, y: np.array) -> dict:
        """Build the per-scan context with kernel-backed memberships."""
        use_sampling = self.sample_for_splits
        if use_sampling is None:
            use_sampling = X.shape[0] > 50000

        if use_sampling and X.shape[0] > self.sample_size:
            n_samples = min(self.sample_size, X.shape[0])
            sample_indices = np.random.choice(X.shape[0], n_samples, replace=False)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
        else:
            sample_indices = None
            X_sample = X
            y_sample = y

        cached_memberships = self._get_cached_memberships(X_sample)
        consistent = (getattr(self, 'consistent_cci', False) and
                      getattr(self, 'prediction_mode', 'soft') == 'soft')
        ones_mask = np.ones_like(X_sample, dtype=bool)

        if consistent:
            base_votes, base_total = self._predict_proba_all_nodes(
                X_sample, ones_mask, return_votes=True)
            base_pred = self.classes_[np.argmax(base_votes, axis=1)]
            skeleton_yhat = base_pred
        else:
            skeleton_yhat = self.predict(X_sample)
            base_votes = None
            base_pred = None
            base_total = None

        if getattr(self, 'coverage_weight', 0.0) > 0.0:
            if base_total is None:
                _, base_total = self._predict_proba_all_nodes(
                    X_sample, ones_mask, return_votes=True)
            uncovered_mask = base_total <= 1e-8
        else:
            uncovered_mask = None

        ctx = {
            'sample_indices': sample_indices,
            'X_sample': X_sample,
            'y_sample': y_sample,
            'cached_memberships': cached_memberships,
            'skeleton_yhat': skeleton_yhat,
            'consistent': consistent,
            'base_votes': base_votes,
            'base_pred': base_pred,
            'uncovered_mask': uncovered_mask,
        }

        cand_feature, cand_fz = self._candidate_layout()
        memb_flat = self._flatten_memberships(ctx["cached_memberships"])

        # Encode labels into classes_ argmax space (classes_ is sorted by np.unique).
        y_sample = ctx["y_sample"]
        y_int = self._encode_labels(y_sample)
        uncovered = np.zeros(len(y_sample), dtype=np.uint8)
        n_uncovered = 0
        if uncovered_mask is not None:
            uncovered = np.ascontiguousarray(uncovered_mask.astype(np.uint8))
            n_uncovered = int(uncovered.sum())

        ctx["_kernel"] = {
            "memb_flat": memb_flat,
            "cand_feature": cand_feature,
            "cand_fz": cand_fz,
            "y_int": y_int,
            "n_classes": len(self.classes_),
            "uncovered": uncovered,
            "coverage_weight": float(getattr(self, 'coverage_weight', 0.0)),
            "n_uncovered": n_uncovered,
            "consistent": consistent,
        }
        if consistent:
            base_votes = np.ascontiguousarray(ctx["base_votes"], dtype=np.float64)
            base_pred_idx = np.argmax(base_votes, axis=1).astype(np.int32)
            ctx["_kernel"].update({
                "base_votes": base_votes,
                "acc_pre": float(np.mean(base_pred_idx == y_int)),
            })
        else:
            skeleton_idx = np.asarray(
                [self._label_to_index(v) for v in skeleton_yhat], dtype=np.int32)
            ctx["_kernel"].update({
                "skeleton_idx": np.ascontiguousarray(skeleton_idx),
                "acc_pre": float(np.mean(skeleton_idx == y_int)),
            })
        return ctx

    def _flatten_memberships(self, cached):
        """Per-feature membership blocks {f: (n_fz_f, n)} -> one (C, n) C-array,
        in the feature-major candidate order used everywhere else."""
        blocks = [cached[f] for f in range(len(self.fuzzy_partitions))]
        return np.ascontiguousarray(np.concatenate(blocks, axis=0), dtype=np.float64)

    def _node_purity_checks(self, node, X: np.array, y: np.array) -> float:
        """Kernel-backed purity (fuzzy-Gini gain) split scoring for one node."""
        # max_depth sentinel: identical to the reference.
        if node['depth'] >= self.max_depth:
            node['aux_purity_cache'] = {
                'feature': -1, 'fuzzy_set': -1, 'coverage': 0.0,
                'split_criterion': 0.0, 'child_decision': None, 'purity': 0.0
            }
            return float('-inf')

        # Split-eval subsampling branch -> reference (membership built ad hoc there).
        use_sampling = self.sample_for_splits
        if use_sampling is None:
            use_sampling = X.shape[0] > 50000
        if use_sampling and X.shape[0] > self.sample_size:
            return super()._node_purity_checks(node, X, y)

        existing = np.ascontiguousarray(node['existing_membership'], dtype=np.float64)
        memb_flat = self._flatten_memberships(self._get_cached_memberships(X))
        fp, cs = node['father_path'], node['child_splits']
        legal = np.concatenate([
            np.logical_and(fp[f], cs[f]) for f in range(len(fp))
        ]).astype(np.uint8)
        # father_purity via the reference helper so it is bit-identical.
        father_purity = float(_tl.compute_fuzzy_purity(existing, y, self.coverage_threshold))
        cand_feature, cand_fz = self._candidate_layout()
        y_int = np.searchsorted(self.classes_, y).astype(np.int32)

        (bf, bfz, bimp, bcov, bchild) = _kernels.score_node_purity(
            memb_flat, cand_feature, cand_fz, legal, existing, y_int,
            len(self.classes_), float(self.coverage_threshold), father_purity)

        if bf != -1:
            node['aux_purity_cache'] = {
                'feature': int(bf), 'fuzzy_set': int(bfz), 'coverage': float(bcov),
                'split_criterion': bimp, 'child_decision': self.classes_[bchild],
                'purity': bimp,
            }
        else:
            node['aux_purity_cache'] = {
                'feature': -1, 'fuzzy_set': -1, 'coverage': 0.0,
                'split_criterion': 0.0, 'child_decision': None, 'purity': 0.0
            }
        return bimp

    def _node_cci_checks(self, node, X: np.array, y: np.array, ctx: dict = None) -> float:
        """Kernel-backed split scoring for one node (consistent-CCI path)."""
        if ctx is None:
            ctx = self._build_cci_context(X, y)

        # max_depth sentinel: identical to the reference.
        if node['depth'] >= self.max_depth:
            node['aux_purity_cache'] = {
                'cci': 0.0, 'feature': -1, 'fuzzy_set': -1, 'coverage': 0.0,
                'split_criterion': 0.0, 'child_decision': None, 'purity': 0.0
            }
            return float('-inf'), float('inf')

        if self.split_mode == 'learned':
            return self._learned_cci_candidates(node, ctx)

        # Legacy / non-accelerated path -> exact reference behaviour.
        if "_kernel" not in ctx:
            return super()._node_cci_checks(node, X, y, ctx)

        kd = ctx["_kernel"]
        sample_indices = ctx["sample_indices"]
        if sample_indices is not None:
            existing = node['existing_membership'][sample_indices]
        else:
            existing = node['existing_membership']
        existing = np.ascontiguousarray(existing, dtype=np.float64)

        # legal candidate mask = father_path AND child_splits, feature-major.
        father_path = node['father_path']
        child_splits = node['child_splits']
        legal = np.concatenate([
            np.logical_and(father_path[f], child_splits[f])
            for f in range(len(father_path))
        ]).astype(np.uint8)

        best_cci_init = float('-inf') if self.tree_rules <= 3 else 0.0

        if kd.get("consistent", False):
            (best_feature, best_fz, best_cci, best_purity,
             best_coverage, best_child_idx) = _kernels.score_node_cci(
                kd["memb_flat"], kd["cand_feature"], kd["cand_fz"], legal,
                existing, kd["y_int"], kd["base_votes"], kd["n_classes"],
                float(self.coverage_threshold), kd["acc_pre"], best_cci_init,
                kd["uncovered"], kd["coverage_weight"], kd["n_uncovered"],
            )
            if (getattr(self, "exact_low_gain_ties", False) and
                    best_feature != -1 and best_cci <= self.min_improvement):
                return super()._node_cci_checks(node, X, y, ctx)
        else:
            (best_feature, best_fz, best_cci, best_purity,
             best_coverage, best_child_idx) = _kernels.score_node_cci_legacy(
                kd["memb_flat"], kd["cand_feature"], kd["cand_fz"], legal,
                existing, kd["y_int"], kd["skeleton_idx"], kd["n_classes"],
                float(self.coverage_threshold), kd["acc_pre"], best_cci_init,
                kd["uncovered"], kd["coverage_weight"], kd["n_uncovered"],
            )
            # The original non-consistent criterion spends its tail iterations
            # in zero/near-zero CCI ties where the reference's numpy summation
            # noise affects the chosen branch. Fall back only in that tie-prone
            # near-stop region; high-value legacy scans stay in the C kernel.
            if (getattr(self, "exact_low_gain_ties", False) and
                    best_feature != -1 and best_cci <= self.min_improvement):
                return super()._node_cci_checks(node, X, y, ctx)

        if best_feature != -1:
            node['aux_purity_cache'] = {
                'cci': best_cci,
                'feature': int(best_feature),
                'fuzzy_set': int(best_fz),
                'coverage': float(best_coverage),
                'split_criterion': best_cci,
                'child_decision': self.classes_[best_child_idx],
                'purity': best_purity,
            }
        else:
            node['aux_purity_cache'] = {
                'cci': 0.0, 'feature': -1, 'fuzzy_set': -1, 'coverage': 0.0,
                'split_criterion': 0.0, 'child_decision': None, 'purity': 0.0
            }

        return best_cci, best_purity
