"""
Adapters for methods from ``Comparison papers/``.

The rule-list papers mostly ship script-first research code rather than pip
packages.  These wrappers keep the benchmark protocol stable while avoiding
vendoring third-party repositories into this project.  Public-code methods look
for a clone path via environment variable first, then under ``external/``.
"""
from __future__ import annotations

import os
import math
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


EPS = 1e-12
ROOT = Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _candidate_repo(env_var: str, default_rel: str) -> Path | None:
    if os.environ.get(env_var):
        p = Path(os.environ[env_var]).expanduser().resolve()
        if p.exists():
            return p
    p = (ROOT / default_rel).resolve()
    if p.exists():
        return p
    return None


@contextmanager
def _prepend_sys_path(path: Path):
    s = str(path)
    old = list(sys.path)
    if s not in sys.path:
        sys.path.insert(0, s)
    try:
        yield
    finally:
        sys.path[:] = old


@contextmanager
def _numpy_seed(seed: int):
    """Contain estimators that draw from NumPy's global RNG."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _require_repo(env_var: str, default_rel: str, label: str) -> Path:
    p = _candidate_repo(env_var, default_rel)
    if p is None:
        raise ImportError(
            f"{label} public code is not installed. Set {env_var} to a clone "
            f"path or clone it under {default_rel}."
        )
    return p


class QuantileBinarizer:
    """Small numeric-only binarizer for rule-list baselines.

    KEEL data reaches benchmark2 as numeric arrays.  We create threshold
    predicates ``x_j <= q``; negated thresholds are available to models through
    negative rule weights or rule-list branches.
    """

    def __init__(self, n_bins: int = 9):
        self.n_bins = n_bins
        self.thresholds_: list[np.ndarray] = []

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        probs = np.linspace(1.0 / (self.n_bins + 1), self.n_bins / (self.n_bins + 1), self.n_bins)
        self.thresholds_ = []
        for j in range(X.shape[1]):
            col = X[:, j]
            vals = np.unique(col[~np.isnan(col)])
            if vals.size <= 1:
                cuts = np.array([], dtype=float)
            elif vals.size <= self.n_bins + 1:
                cuts = vals[:-1].astype(float)
            else:
                cuts = np.unique(np.nanquantile(col, probs)).astype(float)
            self.thresholds_.append(cuts)
        self.n_features_out_ = int(sum(len(t) for t in self.thresholds_))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        cols = []
        for j, cuts in enumerate(self.thresholds_):
            if len(cuts):
                cols.append((X[:, [j]] <= cuts[None, :]).astype(np.float32))
        if not cols:
            return np.zeros((X.shape[0], 1), dtype=np.float32)
        return np.concatenate(cols, axis=1)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class SamRuLeBinarizer:
    """Paper preprocessing: four quantile thresholds and both directions."""

    def __init__(self, n_thresholds: int = 4):
        self.n_thresholds = n_thresholds

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        probs = np.arange(1, self.n_thresholds + 1) / (self.n_thresholds + 1)
        self.thresholds_ = []
        self.predicate_names_ = []
        for j in range(X.shape[1]):
            cuts = np.unique(np.quantile(X[:, j], probs)).astype(float)
            col_min, col_max = np.min(X[:, j]), np.max(X[:, j])
            cuts = cuts[(cuts > col_min) & (cuts < col_max)]
            self.thresholds_.append(cuts)
            for cut in cuts:
                self.predicate_names_.extend((f"x{j}>={cut:.12g}", f"x{j}<{cut:.12g}"))
        self.n_features_out_ = len(self.predicate_names_)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        columns = []
        for j, cuts in enumerate(self.thresholds_):
            for cut in cuts:
                columns.append(X[:, j] >= cut)
                columns.append(X[:, j] < cut)
        if not columns:
            return np.zeros((len(X), 1), dtype=np.uint8)
        return np.column_stack(columns).astype(np.uint8)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


@dataclass
class _CorelsRuleList:
    conditions: list[int]
    predictions: list[int]
    branch_positive_rates: np.ndarray | None = None

    def branch_index(self, X):
        X = np.asarray(X)
        branch = np.full(len(X), len(self.conditions), dtype=int)
        remaining = np.ones(len(X), dtype=bool)
        for i, condition in enumerate(self.conditions):
            matched = remaining & (X[:, condition] == 1)
            branch[matched] = i
            remaining[matched] = False
        return branch

    def hard_predict(self, X):
        branch = self.branch_index(X)
        return np.asarray(self.predictions, dtype=int)[branch]

    def positive_score(self, X):
        if self.branch_positive_rates is None:
            return self.hard_predict(X).astype(float)
        return self.branch_positive_rates[self.branch_index(X)]


class SamRuLeExternal(BaseEstimator, ClassifierMixin):
    """Adapter for VandinLab/SamRuLe and its bundled exact CORELS backend.

    SamRuLe is binary in the paper. ``allow_multiclass=True`` fits one binary
    SamRuLe model per class and is exposed separately as ``SamRuLe-OVR``.
    """

    def __init__(
        self,
        max_rules: int = 5,
        max_terms: int = 1,
        min_frequency: float = 0.0,
        regularization: float = 1e-4,
        epsilon: float = 1.0,
        theta: float = 0.01,
        delta: float = 0.05,
        n_thresholds: int = 4,
        allow_multiclass: bool = False,
        random_state: int = 0,
        timeout: float | None = None,
    ):
        self.max_rules = max_rules
        self.max_terms = max_terms
        self.min_frequency = min_frequency
        self.regularization = regularization
        self.epsilon = epsilon
        self.theta = theta
        self.delta = delta
        self.n_thresholds = n_thresholds
        self.allow_multiclass = allow_multiclass
        self.random_state = random_state
        self.timeout = timeout

    @staticmethod
    def _sample_size(d, k, z, epsilon, theta, delta):
        if min(d, k, z) <= 0 or min(epsilon, theta, delta) <= 0 or delta >= 1:
            raise ValueError("SamRuLe dimensions and approximation parameters must be positive")
        omega = k * z * math.log(2 * math.e * d / z) + 2

        def sufficient(m):
            log_delta = math.log(2.0 / delta) / m
            log_complexity = (omega + math.log(2.0 / delta)) / m
            bound = (
                math.sqrt(3 * theta * log_delta)
                + math.sqrt(2 * (theta + math.sqrt(3 * theta * log_delta)) * log_complexity)
                + 2 * log_complexity
            )
            return bound <= epsilon * theta

        lower = 3 * math.log(2.0 / delta) / theta
        upper = lower
        while not sufficient(upper):
            upper *= 2
        while upper - lower > 1:
            middle = (lower + upper) / 2
            if sufficient(middle):
                upper = middle
            else:
                lower = middle
        return int(math.ceil(upper))

    @staticmethod
    def _write_corels(path, names, values):
        values = np.asarray(values, dtype=np.uint8)
        with open(path, "w", encoding="ascii") as handle:
            for name, row in zip(names, values.T):
                handle.write(name)
                handle.write(" ")
                handle.write(" ".join(map(str, row.tolist())))
                handle.write("\n")

    @staticmethod
    def _parse_corels(stdout):
        marker = "OPTIMAL RULE LIST"
        if marker not in stdout:
            raise RuntimeError("CORELS output did not contain an optimal rule list")
        lines = stdout.split(marker, 1)[1].splitlines()
        conditions, predictions = [], []
        rule_re = re.compile(r"^if \(\{p(\d+)\}\) then \(\{T=([01])\}\)$")
        else_re = re.compile(r"^else \(\{T=([01])\}\)$")
        for raw in lines:
            line = raw.strip()
            match = rule_re.match(line)
            if match:
                conditions.append(int(match.group(1)))
                predictions.append(int(match.group(2)))
                continue
            match = else_re.match(line)
            if match:
                predictions.append(int(match.group(1)))
                break
        if len(predictions) != len(conditions) + 1:
            raise RuntimeError(f"Could not parse CORELS rule list from output: {lines[:8]}")
        return _CorelsRuleList(conditions, predictions)

    def _fit_binary(self, X_binary, y_binary, seed, original_dimension):
        repo = _require_repo("FERL_SAMRULE_REPO", "external/SamRuLe", "SamRuLe")
        executable = repo / "src" / "corels"
        if not executable.is_file():
            raise ImportError(
                f"SamRuLe CORELS executable is missing at {executable}. "
                "Build it with `make -C external/SamRuLe/src corels NGMP=1`."
            )

        sample_size = self._sample_size(
            original_dimension, self.max_rules, self.max_terms,
            self.epsilon, self.theta, self.delta,
        )
        rng = np.random.RandomState(seed)
        sampled = rng.randint(0, len(y_binary), size=sample_size)
        X_sample, y_sample = X_binary[sampled], y_binary[sampled]

        with tempfile.TemporaryDirectory(prefix="ferl-samrule-") as temp_dir:
            temp = Path(temp_dir)
            data_path, label_path = temp / "sample.db", temp / "sample.labels"
            self._write_corels(data_path, [f"{{p{i}}}" for i in range(X_binary.shape[1])], X_sample)
            labels = np.column_stack((y_sample, 1 - y_sample))
            self._write_corels(label_path, ["{T=1}", "{T=0}"], labels)
            command = [
                str(executable), "-n", os.environ.get("FERL_SAMRULE_MAX_NODES", "1000000000"),
                "-r", str(self.regularization), "-c", "1", "-p", "1",
                str(data_path), str(label_path), "-d", str(self.max_rules),
            ]
            # FERL_SAMRULE_TIMEOUT=0 disables the CORELS wall-clock limit
            timeout = self.timeout or _env_float("FERL_SAMRULE_TIMEOUT", 600.0)
            completed = subprocess.run(
                command, cwd=repo, capture_output=True, text=True,
                timeout=timeout or None, check=False,
            )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout)[-1000:]
            raise RuntimeError(f"SamRuLe CORELS failed ({completed.returncode}): {detail}")

        model = self._parse_corels(completed.stdout)
        branch = model.branch_index(X_binary)
        rates = np.empty(len(model.predictions), dtype=float)
        global_rate = float(np.mean(y_binary))
        for i in range(len(rates)):
            in_branch = branch == i
            rates[i] = float(np.mean(y_binary[in_branch])) if np.any(in_branch) else global_rate
        model.branch_positive_rates = rates
        return model, sample_size

    def _fit_conjunctions(self, X_base):
        if self.max_terms < 1:
            raise ValueError("SamRuLe max_terms must be at least one")
        conjunctions = [(i,) for i in range(X_base.shape[1])]
        base_frequencies = X_base.mean(axis=0)
        for size in range(2, self.max_terms + 1):
            for indices in combinations(range(X_base.shape[1]), size):
                values = np.prod(X_base[:, indices], axis=1)
                frequency = float(values.mean())
                individual_frequencies = base_frequencies[list(indices)]
                if frequency <= self.min_frequency or any(np.isclose(frequency, f) for f in individual_frequencies):
                    continue
                conjunctions.append(indices)
        return conjunctions

    @staticmethod
    def _apply_conjunctions(X_base, conjunctions):
        return np.column_stack([
            X_base[:, indices[0]] if len(indices) == 1 else np.prod(X_base[:, indices], axis=1)
            for indices in conjunctions
        ]).astype(np.uint8)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, yi = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("SamRuLe requires at least two classes")
        if len(self.classes_) > 2 and not self.allow_multiclass:
            raise ValueError("SamRuLe is binary; use SamRuLe-OVR for multiclass data")

        self.imputer_ = SimpleImputer(strategy="median").fit(X)
        Xi = self.imputer_.transform(X)
        self.binarizer_ = SamRuLeBinarizer(self.n_thresholds).fit(Xi)
        X_base = self.binarizer_.transform(Xi)
        self.original_dimension_ = X_base.shape[1]
        self.conjunctions_ = self._fit_conjunctions(X_base)
        Xb = self._apply_conjunctions(X_base, self.conjunctions_)
        if Xb.shape[1] == 0:
            raise ValueError("SamRuLe preprocessing produced no binary predicates")

        targets = [1] if len(self.classes_) == 2 else list(range(len(self.classes_)))
        self.models_, self.sample_sizes_ = [], []
        for offset, target in enumerate(targets):
            model, sample_size = self._fit_binary(
                Xb,
                (yi == target).astype(np.uint8),
                self.random_state + offset,
                self.original_dimension_,
            )
            self.models_.append(model)
            self.sample_sizes_.append(sample_size)
        self.n_rules_ = int(sum(len(model.conditions) for model in self.models_))
        self.condition_complexity_ = float(sum(
            len(self.conjunctions_[condition])
            for model in self.models_
            for condition in model.conditions
        ))
        self.complexity_ = float(self.n_rules_)
        return self

    def _transform(self, X):
        base = self.binarizer_.transform(self.imputer_.transform(np.asarray(X, dtype=float)))
        return self._apply_conjunctions(base, self.conjunctions_)

    def predict_proba(self, X):
        Xb = self._transform(X)
        if len(self.classes_) == 2:
            model = self.models_[0]
            rate = model.positive_score(Xb)
            hard = model.hard_predict(Xb).astype(bool)
            positive = np.where(hard, 0.500001 + 0.499999 * rate, 0.499999 * rate)
            proba = np.column_stack((1 - positive, positive))
        else:
            proba = np.column_stack([
                model.hard_predict(Xb) + 0.49 * model.positive_score(Xb)
                for model in self.models_
            ])
        proba = np.clip(proba, EPS, None)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class SampledGreedyRuleList(BaseEstimator, ClassifierMixin):
    """Cheap sampled greedy rule-list ablation.

    This is intentionally distinct from ``SamRuLeExternal``, which runs the
    public KDD'24 implementation and exact CORELS backend.
    """

    def __init__(
        self,
        max_samples: int | None = None,
        n_bins: int = 9,
        max_depth: int = 5,
        random_state: int = 0,
    ):
        self.max_samples = max_samples
        self.n_bins = n_bins
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        from imodels import GreedyRuleListClassifier

        X = np.asarray(X, dtype=float)
        self.classes_, yi = np.unique(y, return_inverse=True)
        max_samples = self.max_samples or _env_int("FERL_SAMPLED_RULELIST_MAX_SAMPLES", 2048)
        rng = np.random.RandomState(self.random_state)

        if len(yi) > max_samples and len(self.classes_) > 1:
            idx = []
            per_class = max(1, max_samples // len(self.classes_))
            for c in range(len(self.classes_)):
                where = np.flatnonzero(yi == c)
                take = min(len(where), per_class)
                idx.extend(rng.choice(where, size=take, replace=False))
            if len(idx) < max_samples:
                rest = np.setdiff1d(np.arange(len(yi)), np.array(idx, dtype=int), assume_unique=False)
                extra = rng.choice(rest, size=min(len(rest), max_samples - len(idx)), replace=False)
                idx.extend(extra.tolist())
            idx = np.array(sorted(idx), dtype=int)
            X_fit, y_fit = X[idx], yi[idx]
        else:
            X_fit, y_fit = X, yi

        self.imputer_ = SimpleImputer(strategy="median").fit(X_fit)
        X_fit = self.imputer_.transform(X_fit)
        self.binarizer_ = QuantileBinarizer(self.n_bins).fit(X_fit)
        X_bin = self.binarizer_.transform(X_fit)
        base = GreedyRuleListClassifier(max_depth=self.max_depth)
        # imodels does not expose the random_state of its internal CART stumps.
        with _numpy_seed(self.random_state):
            self.model_ = OneVsRestClassifier(base).fit(X_bin, y_fit)
        self.complexity_ = float(np.nansum([getattr(e, "complexity_", np.nan) for e in self.model_.estimators_]))
        self.n_rules_ = int(self.complexity_)
        self.condition_complexity_ = float(self.complexity_)
        self.max_samples_ = int(max_samples)
        return self

    def _transform(self, X):
        X = self.imputer_.transform(np.asarray(X, dtype=float))
        return self.binarizer_.transform(X)

    def predict_proba(self, X):
        p = np.asarray(self.model_.predict_proba(self._transform(X)), dtype=float)
        if p.ndim == 1:
            p = np.vstack([1.0 - p, p]).T
        p = np.clip(p, EPS, None)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class RRLExternal(BaseEstimator, ClassifierMixin):
    """CPU sklearn-style adapter for Wang et al.'s public RRL PyTorch code."""

    def __init__(
        self,
        epochs: int | None = None,
        batch_size: int | None = None,
        binarization_nodes: int | None = None,
        hidden_width: int | None = None,
        learning_rate: float | None = None,
        weight_decay: float | None = None,
        random_state: int = 0,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.binarization_nodes = binarization_nodes
        self.hidden_width = hidden_width
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state

    def fit(self, X, y):
        repo = _require_repo("FERL_RRL_REPO", "external/rrl", "RRL")
        with _prepend_sys_path(repo):
            import torch
            from rrl import components as rrl_components
            from rrl.models import Net

        self._patch_rrl_for_modern_torch(rrl_components, torch)
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=float)
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.imputer_ = SimpleImputer(strategy="median").fit(X)
        self.scaler_ = StandardScaler().fit(self.imputer_.transform(X))
        Xp = self.scaler_.transform(self.imputer_.transform(X)).astype(np.float32)
        left = torch.tensor(np.nanmin(Xp, axis=0), dtype=torch.float32)
        right = torch.tensor(np.nanmax(Xp, axis=0), dtype=torch.float32)

        epochs = self.epochs or _env_int("FERL_RRL_EPOCHS", 80)
        batch_size = self.batch_size or _env_int("FERL_RRL_BATCH_SIZE", 128)
        bin_nodes = self.binarization_nodes or _env_int("FERL_RRL_BIN_NODES", 5)
        width = self.hidden_width or _env_int("FERL_RRL_WIDTH", 64)
        lr = self.learning_rate or _env_float("FERL_RRL_LR", 0.002)
        wd = self.weight_decay if self.weight_decay is not None else _env_float("FERL_RRL_WD", 1e-4)
        self.epochs_ = int(epochs)
        self.batch_size_ = int(batch_size)
        self.binarization_nodes_ = int(bin_nodes)
        self.hidden_width_ = int(width)
        self.learning_rate_ = float(lr)
        self.weight_decay_ = float(wd)

        self.net_ = Net(
            [(0, Xp.shape[1]), bin_nodes, width, len(self.classes_)],
            left=left,
            right=right,
            use_nlaf=True,
            use_skip=False,
            temperature=0.01,
        )
        # Upstream stores connection metadata on a local lambda, which makes a
        # trained network impossible to pickle. Preserve the same attributes on
        # a standard, serializable namespace.
        for layer in self.net_.layer_list:
            connection = layer.conn
            layer.conn = SimpleNamespace(
                prev_layer=connection.prev_layer,
                is_skip_to_layer=connection.is_skip_to_layer,
                skip_from_layer=connection.skip_from_layer,
            )
        opt = torch.optim.Adam(self.net_.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        ds = torch.utils.data.TensorDataset(
            torch.tensor(Xp, dtype=torch.float32),
            torch.tensor(yi, dtype=torch.long),
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

        self.net_.train()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                logits = self.net_(xb) / torch.exp(self.net_.t)
                loss = loss_fn(logits, yb)
                if wd:
                    penalty = torch.zeros((), dtype=logits.dtype)
                    for layer in self.net_.layer_list[1:]:
                        if hasattr(layer, "l2_norm"):
                            penalty = penalty + layer.l2_norm()
                    loss = loss + wd * penalty
                loss.backward()
                opt.step()
                with torch.no_grad():
                    for layer in self.net_.layer_list:
                        if hasattr(layer, "clip"):
                            layer.clip()

        self.net_.eval()
        self.complexity_ = self._complexity()
        self.n_rules_ = int(width)
        self.condition_complexity_ = float(self.complexity_)
        return self

    @staticmethod
    def _patch_rrl_for_modern_torch(rrl_components, torch):
        if getattr(rrl_components.BinarizeLayer, "_ferl_reshape_patch", False):
            return

        def forward(layer, x):
            if layer.input_dim[1] > 0:
                x_disc, x_cont = x[:, 0: layer.input_dim[0]], x[:, layer.input_dim[0]:]
                x_cont = x_cont.unsqueeze(-1)
                if layer.use_not:
                    x_disc = torch.cat((x_disc, 1 - x_disc), dim=1)
                binarize_res = rrl_components.Binarize.apply(x_cont - layer.cl.t()).reshape(x_cont.shape[0], -1)
                return torch.cat((x_disc, binarize_res, 1.0 - binarize_res), dim=1)
            if layer.use_not:
                x = torch.cat((x, 1 - x), dim=1)
            return x

        rrl_components.BinarizeLayer.forward = forward
        rrl_components.BinarizeLayer._ferl_reshape_patch = True

    def _transform(self, X):
        X = self.imputer_.transform(np.asarray(X, dtype=float))
        return self.scaler_.transform(X).astype(np.float32)

    def _complexity(self):
        total = 0.0
        for layer in self.net_.layer_list[1:-1]:
            if hasattr(layer, "con_layer") and hasattr(layer, "dis_layer"):
                total += float((layer.con_layer.W.detach().cpu().numpy() > 0.5).sum())
                total += float((layer.dis_layer.W.detach().cpu().numpy() > 0.5).sum())
        return total

    def predict_proba(self, X):
        import torch
        from rrl import components as rrl_components

        self._patch_rrl_for_modern_torch(rrl_components, torch)
        Xp = torch.tensor(self._transform(X), dtype=torch.float32)
        with torch.no_grad():
            logits = self.net_(Xp) / torch.exp(self.net_.t)
            p = torch.softmax(logits, dim=1).cpu().numpy()
        p = np.clip(p, EPS, None)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class RLNetExternal(BaseEstimator, ClassifierMixin):
    """CPU sklearn-style adapter for Dierckx et al.'s public RL-Net code."""

    def __init__(
        self,
        n_rules: int | None = None,
        n_bins: int | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        lambda_and: float | None = None,
        validation_fraction: float = 0.2,
        l2_lambda: float | None = None,
        random_state: int = 0,
    ):
        self.n_rules = n_rules
        self.n_bins = n_bins
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_and = lambda_and
        self.validation_fraction = validation_fraction
        self.l2_lambda = l2_lambda
        self.random_state = random_state

    def fit(self, X, y):
        repo = _require_repo("FERL_RLNET_REPO", "external/RLNet", "RL-Net")
        with _prepend_sys_path(repo):
            import torch
            from networkTorch_multiClass import Network

        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=float)
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.imputer_ = SimpleImputer(strategy="median").fit(X)
        Xi = self.imputer_.transform(X)
        self.binarizer_ = QuantileBinarizer(self.n_bins or _env_int("FERL_RLNET_BINS", 9)).fit(Xi)
        Xb = self.binarizer_.transform(Xi)

        n_rules = self.n_rules or _env_int("FERL_RLNET_RULES", 20)
        epochs = self.epochs or _env_int("FERL_RLNET_EPOCHS", 3000)
        lr = self.learning_rate or _env_float("FERL_RLNET_LR", 0.01)
        lam = self.lambda_and if self.lambda_and is not None else _env_float("FERL_RLNET_LAMBDA_AND", 1e-3)
        l2_lam = self.l2_lambda if self.l2_lambda is not None else _env_float("FERL_RLNET_L2", 0.0)

        indices = np.arange(len(yi))
        if 0.0 < self.validation_fraction < 1.0 and len(yi) > len(self.classes_):
            try:
                train_idx, val_idx = train_test_split(
                    indices,
                    test_size=self.validation_fraction,
                    random_state=self.random_state,
                    stratify=yi,
                )
            except ValueError:
                train_idx, val_idx = train_test_split(
                    indices,
                    test_size=self.validation_fraction,
                    random_state=self.random_state,
                )
        else:
            train_idx = val_idx = indices
        default_batch_size = max(1, int(len(train_idx) * 0.05))
        batch_size = self.batch_size or _env_int("FERL_RLNET_BATCH_SIZE", default_batch_size)
        self.n_rules_ = int(n_rules)
        self.n_bins_ = int(self.n_bins or _env_int("FERL_RLNET_BINS", 9))
        self.epochs_ = int(epochs)
        self.batch_size_ = int(batch_size)
        self.learning_rate_ = float(lr)
        self.lambda_and_ = float(lam)
        self.l2_lambda_ = float(l2_lam)

        self.model_ = Network(Xb.shape[1], n_rules, len(self.classes_))
        opt = torch.optim.Adam(self.model_.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        ds = torch.utils.data.TensorDataset(
            torch.tensor(Xb[train_idx], dtype=torch.float32),
            torch.tensor(yi[train_idx], dtype=torch.long),
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)
        Xval = torch.tensor(Xb[val_idx], dtype=torch.float32)
        yval = torch.tensor(yi[val_idx], dtype=torch.long)

        best_loss = np.inf
        best_state = None
        for _ in range(epochs):
            self.model_.train()
            for xb, yb in loader:
                opt.zero_grad()
                proba, _ = self.model_(xb)
                l2 = sum(torch.linalg.norm(p, 2) for p in self.model_.output.parameters())
                loss = loss_fn(proba, yb) + lam * self.model_.regularization() + l2_lam * l2
                loss.backward()
                opt.step()

            self.model_.eval()
            with torch.no_grad():
                val_proba, _ = self.model_(Xval)
                val_l2 = sum(torch.linalg.norm(p, 2) for p in self.model_.output.parameters())
                val_loss = loss_fn(val_proba, yval) + lam * self.model_.regularization() + l2_lam * val_l2
                if val_loss.item() < best_loss:
                    best_loss = val_loss.item()
                    best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.best_validation_loss_ = float(best_loss)
        self.complexity_ = float((self.model_.and_layer.masked_weight().detach().cpu().numpy() != 0).sum())
        self.condition_complexity_ = float(self.complexity_)
        return self

    def _transform(self, X):
        X = self.imputer_.transform(np.asarray(X, dtype=float))
        return self.binarizer_.transform(X)

    def predict_proba(self, X):
        import torch

        xb = torch.tensor(self._transform(X), dtype=torch.float32)
        with torch.no_grad():
            p, _ = self.model_(xb)
            p = p.cpu().numpy()
        p = np.clip(p, EPS, None)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class MissingPaperCode(BaseEstimator, ClassifierMixin):
    def __init__(self, label: str):
        self.label = label

    def fit(self, X, y):
        raise ImportError(f"{self.label} has no public estimator code wired into this benchmark yet.")
