"""NeuRules (Neural Rule Lists) classifier.

This is a self-contained sklearn adapter derived from the architecture and
training procedure released in the official NeurIPS 2025 supplemental archive.
It intentionally does not depend on the supplemental repository or its bundled
datasets/baselines.

Reference: R. Guidotti et al., "Neural Rule Lists", NeurIPS 2025.
Official source archive:
https://papers.nips.cc/paper_files/paper/2025/file/
da813258fb4db548b0a6c34d79da8594-Supplemental-Conference.zip
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _SmallCategoricalExpander:
    """Match the public preprocessing: one-hot encode 3/4-valued columns."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.categories_ = []
        for column in X.T:
            values = np.unique(column)
            self.categories_.append(values if 2 < len(values) < 5 else None)
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = sum(
            len(values) if values is not None else 1 for values in self.categories_
        )
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} input features")
        columns = []
        for index, values in enumerate(self.categories_):
            column = X[:, index]
            if values is None:
                columns.append(column[:, None])
            else:
                columns.append((column[:, None] == values[None, :]).astype(float))
        return np.concatenate(columns, axis=1)


class _DiscretizingLayer(nn.Module):
    """Learn one soft interval predicate per feature and rule."""

    def __init__(self, n_features, n_rules, limits, temperature=0.2):
        super().__init__()
        self.temperature = float(temperature)
        self.n_rules = int(n_rules)
        self.register_buffer("limits", limits.detach().clone())

        # Consume the same random tensors as the released implementation before
        # replacing random intervals with full-range intervals.
        interval_size = torch.rand(n_features, n_rules) * 0.2 + 0.2
        start = torch.rand(n_features, n_rules) * (1.0 - interval_size)
        init = torch.stack((start, start + interval_size), dim=1)
        span = limits[:, 1] - limits[:, 0]
        init[:, 0, :] = init[:, 0, :] * span[:, None] + limits[:, 0, None]
        init[:, 1, :] = init[:, 1, :] * span[:, None] + limits[:, 0, None]
        self.cut_points = nn.Parameter(init)
        with torch.no_grad():
            self.cut_points.copy_(limits.unsqueeze(2).repeat(1, 1, n_rules))

    def forward(self, X):
        X = X.unsqueeze(2)
        lower = self.cut_points[:, 0, :]
        upper = self.cut_points[:, 1, :]

        inside = (2.0 * X - lower) / self.temperature
        below = X / self.temperature
        above = (3.0 * X - lower - upper) / self.temperature
        maximum = torch.maximum(torch.maximum(inside, below), above).detach()
        numerator = torch.exp(inside - maximum)
        denominator = (
            numerator + torch.exp(below - maximum) + torch.exp(above - maximum)
        )
        return numerator / denominator

    def fix_parameters(self):
        with torch.no_grad():
            self.cut_points.copy_(torch.sort(self.cut_points, dim=1).values)


class _AndLayer(nn.Module):
    """Relaxed weighted harmonic conjunction from the NeuRules paper."""

    def __init__(self, n_features, n_rules, epsilon=0.001):
        super().__init__()
        self.epsilon = float(epsilon)
        # Preserve the released initialization's RNG consumption before its
        # weights are overwritten with the constant 0.5.
        self.and_weights = nn.Parameter(torch.rand(n_rules, n_features))
        with torch.no_grad():
            self.and_weights.fill_(0.5)

    def forward(self, predicates):
        weights = torch.relu(self.and_weights)
        predicates = predicates.permute(0, 2, 1)
        weight_sum = weights.sum(dim=1)

        # The released expression is undefined only if an entire rule has
        # non-positive weights. Clamp there so a transient optimizer state does
        # not poison the complete fit with NaNs.
        safe_sum = weight_sum.clamp_min(torch.finfo(predicates.dtype).eps)
        eta = (self.epsilon / safe_sum).detach()[None, :, None]
        inverse = (1.0 + eta) / (predicates + eta)
        denominator = (inverse * weights[None, :, :]).sum(dim=2)
        result = safe_sum[None, :] / denominator.clamp_min(
            torch.finfo(predicates.dtype).eps
        )
        return torch.where(weight_sum[None, :] > 0, result, torch.zeros_like(result))


class _NeuRulesNetwork(nn.Module):
    def __init__(
        self,
        n_features,
        n_classes,
        n_rules,
        limits,
        predicate_temperature=0.2,
        selector_temperature=1.0,
        epsilon=0.001,
    ):
        super().__init__()
        self.n_rules = int(n_rules)
        self.n_classes = int(n_classes)
        self.discretizer = _DiscretizingLayer(
            n_features, n_rules, limits, predicate_temperature
        )
        self.rules = _AndLayer(n_features, n_rules, epsilon)
        self.rule_order = nn.Parameter(torch.rand(n_rules))
        self.rule_weights = nn.Linear(n_rules, n_classes, bias=False)
        self.selector_temperature = float(selector_temperature)

        with torch.no_grad():
            self.rule_weights.weight.fill_(0.5)
            part = n_rules // n_classes
            for class_index in range(n_classes):
                end = (class_index + 1) * part
                if class_index == n_classes - 1:
                    end = n_rules
                self.rule_weights.weight[class_index, class_index * part : end] = 10.0

    def forward(self, X, return_rules=False):
        predicates = self.discretizer(X)
        rule_activations = self.rules(predicates)
        priority = self.rule_order - torch.min(self.rule_order).item() + 1.0
        weighted_priority = rule_activations * priority
        selection = nn.functional.gumbel_softmax(
            weighted_priority, tau=self.selector_temperature, hard=False
        )
        logits = self.rule_weights(selection)
        if return_rules:
            return logits, rule_activations, selection
        return logits

    def fix_parameters(self):
        self.discretizer.fix_parameters()


@dataclass(frozen=True)
class _TemperatureSchedule:
    start: float
    end: float
    steps: int

    def value(self, step):
        fraction = min(max(step, 0), self.steps) / max(self.steps, 1)
        return self.start + fraction * (self.end - self.start)


class NeuRules(BaseEstimator, ClassifierMixin):
    """Neural Rule Lists classifier with hard rule-list predictions.

    ``source_compatible=True`` reproduces two noteworthy behaviors of the
    released trainer: two Adam updates per batch and a minimum-support-only
    selection penalty. Setting it to ``False`` uses one update and the squared
    min/max support penalty described by Equation 15 of the paper.
    """

    def __init__(
        self,
        n_rules=None,
        epochs=None,
        batch_size=None,
        learning_rate=None,
        min_support=0.2,
        max_support=0.9,
        coverage_weight=0.4,
        predicate_temperature=0.2,
        selector_temperature=1.0,
        predicate_temperature_end=0.04,
        selector_temperature_end=0.2,
        epsilon=0.001,
        source_compatible=True,
        random_state=0,
        device="cpu",
    ):
        self.n_rules = n_rules
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.min_support = min_support
        self.max_support = max_support
        self.coverage_weight = coverage_weight
        self.predicate_temperature = predicate_temperature
        self.selector_temperature = selector_temperature
        self.predicate_temperature_end = predicate_temperature_end
        self.selector_temperature_end = selector_temperature_end
        self.epsilon = epsilon
        self.source_compatible = source_compatible
        self.random_state = random_state
        self.device = device

    @staticmethod
    def _env_or_value(value, environment, default, cast):
        if value is not None:
            return cast(value)
        return cast(os.environ.get(environment, default))

    @staticmethod
    def _as_float_matrix(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
            raise ValueError("X must be a non-empty two-dimensional array")
        if np.isinf(X).any():
            raise ValueError("X contains infinite values")
        return X

    def _validate_parameters(self):
        if not 0 <= self.min_support <= self.max_support <= 1:
            raise ValueError("support bounds must satisfy 0 <= min <= max <= 1")
        if self.coverage_weight < 0 or self.epsilon <= 0:
            raise ValueError("coverage_weight must be non-negative and epsilon positive")
        if min(
            self.predicate_temperature,
            self.selector_temperature,
            self.predicate_temperature_end,
            self.selector_temperature_end,
        ) <= 0:
            raise ValueError("temperatures must be positive")
        if self.device != "cpu" and not torch.cuda.is_available():
            raise ValueError(f"requested unavailable torch device: {self.device}")

    def _fit_preprocessor(self, X):
        self.imputer_ = SimpleImputer(strategy="median").fit(X)
        X = self.imputer_.transform(X)
        self.expander_ = _SmallCategoricalExpander().fit(X)
        X = self.expander_.transform(X)
        self.scaler_ = StandardScaler().fit(X)
        return self.scaler_.transform(X).astype(np.float32, copy=False)

    def _transform(self, X):
        X = self._as_float_matrix(X)
        X = self.imputer_.transform(X)
        X = self.expander_.transform(X)
        return self.scaler_.transform(X).astype(np.float32, copy=False)

    def fit(self, X, y):
        self._validate_parameters()
        X = self._as_float_matrix(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be one-dimensional and aligned with X")
        self.classes_, encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("NeuRules requires at least two classes")

        X_train = self._fit_preprocessor(X)
        n_classes = len(self.classes_)
        default_rules = 10 if n_classes == 2 else 5 * n_classes
        self.n_rules_ = self._env_or_value(
            self.n_rules, "FERL_NEURULES_RULES", default_rules, int
        )
        self.epochs_ = self._env_or_value(
            self.epochs, "FERL_NEURULES_EPOCHS", 500, int
        )
        self.batch_size_ = self._env_or_value(
            self.batch_size, "FERL_NEURULES_BATCH_SIZE", 2048, int
        )
        self.learning_rate_ = self._env_or_value(
            self.learning_rate, "FERL_NEURULES_LR", 0.01, float
        )
        if min(self.n_rules_, self.epochs_, self.batch_size_) <= 0:
            raise ValueError("n_rules, epochs, and batch_size must be positive")
        if self.learning_rate_ <= 0:
            raise ValueError("learning_rate must be positive")

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        torch.use_deterministic_algorithms(True, warn_only=True)

        tensor_X = torch.from_numpy(X_train)
        tensor_y = torch.from_numpy(encoded.astype(np.int64, copy=False))
        limits = torch.stack((tensor_X.min(dim=0).values, tensor_X.max(dim=0).values), dim=1)
        self.network_ = _NeuRulesNetwork(
            X_train.shape[1],
            n_classes,
            self.n_rules_,
            limits,
            self.predicate_temperature,
            self.selector_temperature,
            self.epsilon,
        ).to(self.device)

        loader = DataLoader(
            TensorDataset(tensor_X, tensor_y),
            batch_size=min(self.batch_size_, len(tensor_X)),
            shuffle=True,
        )
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.learning_rate_)
        criterion = nn.CrossEntropyLoss()
        half = self.epochs_ // 2
        schedule_steps = max(self.epochs_ - half, 1)
        selector_schedule = _TemperatureSchedule(
            self.selector_temperature, self.selector_temperature_end, schedule_steps
        )
        predicate_schedule = _TemperatureSchedule(
            self.predicate_temperature, self.predicate_temperature_end, schedule_steps
        )
        schedule_step = 0
        self.loss_curve_ = []
        self.network_.train()

        for epoch in range(self.epochs_):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits, _, selection = self.network_(batch_X, return_rules=True)
                fit_loss = criterion(logits, batch_y)
                support = selection.mean(dim=0)
                if self.source_compatible:
                    coverage_loss = torch.relu(self.min_support - support).mean()
                else:
                    under = torch.relu(self.min_support - support).square()
                    over = torch.relu(support - self.max_support).square()
                    coverage_loss = (under + over).mean()
                loss = fit_loss + self.coverage_weight * coverage_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("NeuRules produced a non-finite training loss")
                loss.backward()
                if epoch < half:
                    self.network_.rule_weights.weight.grad = None
                optimizer.step()
                if self.source_compatible:
                    optimizer.step()
                self.network_.fix_parameters()
                self.loss_curve_.append(float(loss.detach().cpu()))

                if epoch >= half:
                    schedule_step += 1
                    self.network_.selector_temperature = selector_schedule.value(schedule_step)
                    self.network_.discretizer.temperature = predicate_schedule.value(schedule_step)

        self.network_.eval()
        self._extract_hard_rules()
        return self

    def _extract_hard_rules(self):
        network = self.network_
        self.cut_points_ = network.discretizer.cut_points.detach().cpu().numpy().copy()
        self.and_weights_ = network.rules.and_weights.detach().cpu().numpy().copy()
        self.rule_order_ = network.rule_order.detach().cpu().numpy().copy()
        self.rule_logits_ = network.rule_weights.weight.detach().cpu().numpy().T.copy()
        self.rule_indices_ = np.argsort(self.rule_order_)[::-1]
        self.active_features_ = self.and_weights_ > 0
        nonempty = self.active_features_.any(axis=1)
        self.complexity_ = float(nonempty.sum())
        self.condition_complexity_ = float(self.active_features_[nonempty].sum())
        self.rules_ = []
        for index in self.rule_indices_:
            active = np.flatnonzero(self.active_features_[index])
            if not len(active):
                continue
            self.rules_.append(
                {
                    "index": int(index),
                    "priority": float(self.rule_order_[index]),
                    "features": active.copy(),
                    "lower": self.cut_points_[active, 0, index].copy(),
                    "upper": self.cut_points_[active, 1, index].copy(),
                    "class": self.classes_[int(np.argmax(self.rule_logits_[index]))],
                }
            )

    def _hard_logits(self, X):
        X = self._transform(X)
        logits = np.zeros((len(X), len(self.classes_)), dtype=np.float32)
        uncovered = np.ones(len(X), dtype=bool)
        for index in self.rule_indices_:
            active = self.active_features_[index]
            if not active.any():
                continue
            match = (
                (X[:, active] > self.cut_points_[active, 0, index])
                & (X[:, active] < self.cut_points_[active, 1, index])
            ).all(axis=1)
            selected = uncovered & match
            logits[selected] = self.rule_logits_[index]
            uncovered[selected] = False
            if not uncovered.any():
                break
        return logits

    def predict_proba(self, X):
        check_is_fitted(self, "rule_logits_")
        logits = self._hard_logits(X).astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
