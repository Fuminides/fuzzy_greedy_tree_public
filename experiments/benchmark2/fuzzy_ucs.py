"""Python port of jUCS Fuzzy-UCS with Dempster-Shafer inference.

The algorithm follows YNU-NakataLab/jUCS Fuzzy-UCS core at revision
81f8bb6673436fef45fd31929836ce4f443de15b. Dataset splitting/reporting are
intentionally left to benchmark2; this exposes a deterministic sklearn estimator.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer


EPS = 1e-12


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class FuzzyUCSParameters:
    population_size: int = 2000
    fitness_threshold: float = 0.99
    fitness_exponent: float = 1.0
    ga_threshold: int = 50
    crossover_probability: float = 0.8
    mutation_probability: float = 0.04
    deletion_threshold: int = 50
    deletion_fraction: float = 0.1
    subsumption_threshold: int = 50
    tournament_probability: float = 0.4
    ga_subsumption: bool = True
    correct_set_subsumption: bool = True
    wildcard_probability: float = 0.33
    exploitation_threshold: float = 10.0


@dataclass
class FuzzyClassifier:
    identifier: int
    condition: np.ndarray
    weights: np.ndarray
    fitness: float = 1.0
    correct_matching: np.ndarray = field(default_factory=lambda: np.empty(0))
    experience: float = 0.0
    time_stamp: int = 0
    correct_set_size: float = 1.0
    correct_set_size_sum: float = 0.0
    correct_set_size_count: int = 0
    numerosity: int = 1
    matching_degree: float = 1.0


def cnf_membership(condition, value):
    """Membership of a disjunction of the five jUCS triangular terms."""
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Fuzzy-UCS input must be in [0, 1], got {value}")
    segment = min(int(np.floor(value * 4)), 3)
    lower, upper = bool(condition[segment]), bool(condition[segment + 1])
    if lower and upper:
        return 1.0
    fraction = (value - segment * 0.25) / 0.25
    if lower:
        return 1.0 - fraction
    if upper:
        return fraction
    return 0.0


def combine_dempster(left, right):
    """Combine singleton-plus-frame masses using normalized Dempster's rule."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    combined = np.zeros_like(left)
    combined[:-1] = (
        left[:-1] * right[-1]
        + left[-1] * right[:-1]
        + left[:-1] * right[:-1]
    )
    combined[-1] = left[-1] * right[-1]
    total = float(combined.sum())
    conflict = max(0.0, 1.0 - total)
    if total <= EPS:
        return np.zeros_like(combined), 1.0
    return combined / total, conflict


class FuzzyUCSDS(BaseEstimator, ClassifierMixin):
    """Michigan-style fuzzy classifier system with DS class inference."""

    def __init__(
        self,
        epochs: int | None = None,
        population_size: int | None = None,
        fitness_threshold: float = 0.99,
        fitness_exponent: float = 1.0,
        ga_threshold: int = 50,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.04,
        deletion_threshold: int = 50,
        deletion_fraction: float = 0.1,
        subsumption_threshold: int = 50,
        tournament_probability: float = 0.4,
        ga_subsumption: bool = True,
        correct_set_subsumption: bool = True,
        wildcard_probability: float = 0.33,
        exploitation_threshold: float = 10.0,
        random_state: int = 0,
    ):
        self.epochs = epochs
        self.population_size = population_size
        self.fitness_threshold = fitness_threshold
        self.fitness_exponent = fitness_exponent
        self.ga_threshold = ga_threshold
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.deletion_threshold = deletion_threshold
        self.deletion_fraction = deletion_fraction
        self.subsumption_threshold = subsumption_threshold
        self.tournament_probability = tournament_probability
        self.ga_subsumption = ga_subsumption
        self.correct_set_subsumption = correct_set_subsumption
        self.wildcard_probability = wildcard_probability
        self.exploitation_threshold = exploitation_threshold
        self.random_state = random_state

    def _parameters(self):
        return FuzzyUCSParameters(
            population_size=self.population_size or _env_int("FERL_FUZZYUCS_POPULATION", 2000),
            fitness_threshold=self.fitness_threshold,
            fitness_exponent=self.fitness_exponent,
            ga_threshold=self.ga_threshold,
            crossover_probability=self.crossover_probability,
            mutation_probability=self.mutation_probability,
            deletion_threshold=self.deletion_threshold,
            deletion_fraction=self.deletion_fraction,
            subsumption_threshold=self.subsumption_threshold,
            tournament_probability=self.tournament_probability,
            ga_subsumption=self.ga_subsumption,
            correct_set_subsumption=self.correct_set_subsumption,
            wildcard_probability=self.wildcard_probability,
            exploitation_threshold=self.exploitation_threshold,
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, yi = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Fuzzy-UCS requires at least two classes")
        self.n_classes_ = len(self.classes_)
        self.imputer_ = SimpleImputer(strategy="median").fit(X)
        Xi = self.imputer_.transform(X)
        self.feature_min_ = Xi.min(axis=0)
        self.feature_max_ = Xi.max(axis=0)
        Xn = self._normalize_imputed(Xi)

        self.parameters_ = self._parameters()
        self.rng_ = np.random.RandomState(self.random_state)
        self.population_: list[FuzzyClassifier] = []
        self.time_stamp_ = 0
        self.global_id_ = 0
        self.covering_count_ = 0
        self.subsumption_count_ = 0
        self._condition_cache = None

        epochs = self.epochs or _env_int("FERL_FUZZYUCS_EPOCHS", 50)
        self.epochs_ = int(epochs)
        self.population_size_ = int(self.parameters_.population_size)
        # jUCS recreates MersenneTwister(seed) for each epoch's ordering.
        fixed_order = np.random.RandomState(self.random_state).permutation(len(yi))
        for _ in range(epochs):
            for index in fixed_order:
                self._train_one(Xn[index], int(yi[index]))

        self.complexity_ = float(len(self.population_))
        self.n_rules_ = int(len(self.population_))
        self.condition_complexity_ = float(sum((~rule.condition.all(axis=1)).sum() for rule in self.population_))
        return self

    def _normalize_imputed(self, X):
        span = self.feature_max_ - self.feature_min_
        normalized = np.empty_like(X, dtype=float)
        varying = span > 0
        normalized[:, varying] = (X[:, varying] - self.feature_min_[varying]) / span[varying]
        normalized[:, ~varying] = 0.5
        return np.clip(normalized, 0.0, 1.0)

    def _transform(self, X):
        Xi = self.imputer_.transform(np.asarray(X, dtype=float))
        return self._normalize_imputed(Xi)

    def _new_condition(self, state):
        condition = np.ones((len(state), 5), dtype=bool)
        for i, value in enumerate(state):
            if self.rng_.rand() < self.parameters_.wildcard_probability:
                continue
            condition[i] = False
            if value == 0.0:
                condition[i, 0] = True
            elif value < 0.25:
                condition[i, :2] = True
            elif value < 0.5:
                condition[i, 1] = True
                condition[i, 2] = value != 0.25
            elif value < 0.75:
                condition[i, 2] = True
                condition[i, 3] = value != 0.5
            elif value < 1.0:
                condition[i, 3] = True
                condition[i, 4] = value != 0.75
            else:
                condition[i, 4] = True
        return condition

    def _covering_classifier(self, state, answer):
        weights = np.zeros(self.n_classes_, dtype=float)
        weights[answer] = 1.0
        rule = FuzzyClassifier(
            identifier=self.global_id_,
            condition=self._new_condition(state),
            weights=weights,
            correct_matching=np.zeros(self.n_classes_, dtype=float),
            time_stamp=self.time_stamp_,
        )
        self.global_id_ += 1
        self.covering_count_ += 1
        return rule

    def _invalidate_conditions(self):
        self._condition_cache = None

    def _matching_degrees(self, state):
        if not self.population_:
            return np.empty(0, dtype=float)
        if self._condition_cache is None:
            self._condition_cache = np.stack([rule.condition for rule in self.population_])
        conditions = self._condition_cache
        segments = np.minimum(np.floor(state * 4).astype(int), 3)
        feature_indices = np.arange(len(state))
        lower = conditions[:, feature_indices, segments]
        upper = conditions[:, feature_indices, segments + 1]
        fraction = (state - segments * 0.25) / 0.25
        membership = np.where(
            lower & upper,
            1.0,
            np.where(lower, 1.0 - fraction, np.where(upper, fraction, 0.0)),
        )
        return membership.prod(axis=1)

    def _match_set(self, state):
        degrees = self._matching_degrees(state)
        match = []
        for rule, degree in zip(self.population_, degrees):
            rule.matching_degree = float(degree)
            if degree > 0:
                match.append(rule)
        return match

    def _covering_sufficient(self, match_set, answer):
        totals = np.zeros(self.n_classes_, dtype=float)
        for rule in match_set:
            totals[int(np.argmax(rule.weights))] += rule.matching_degree
            if totals[answer] >= 1.0:
                return True
        return False

    def _train_one(self, state, answer):
        match_set = self._match_set(state)
        if not self._covering_sufficient(match_set, answer):
            rule = self._covering_classifier(state, answer)
            self.population_.append(rule)
            self._invalidate_conditions()
            self._delete_from_population()
            match_set.append(rule)
            # deletion may have removed a rule still referenced by match_set;
            # keeping it would divide by its zero numerosity in _run_ga
            match_set = [r for r in match_set if r.numerosity > 0]

        correct_set = [rule for rule in match_set if int(np.argmax(rule.weights)) == answer]
        self._update_set(match_set, correct_set, answer)
        self._run_ga(correct_set)
        self.time_stamp_ += 1

    def _update_set(self, match_set, correct_set, answer):
        if not correct_set:
            return
        set_numerosity = sum(rule.numerosity for rule in correct_set)
        for rule in match_set:
            rule.experience += rule.matching_degree
            rule.correct_matching[answer] += rule.matching_degree
            if rule.experience > 0:
                rule.weights = rule.correct_matching / rule.experience
            rule.fitness = 2 * float(np.max(rule.weights)) - float(np.sum(rule.weights))
        for rule in correct_set:
            rule.correct_set_size_sum += set_numerosity
            rule.correct_set_size_count += 1
            rule.correct_set_size = rule.correct_set_size_sum / rule.correct_set_size_count
        if self.parameters_.correct_set_subsumption:
            self._correct_set_subsumption(correct_set)

    def _could_subsume(self, rule):
        return (
            rule.experience > self.parameters_.subsumption_threshold
            and rule.fitness > self.parameters_.fitness_threshold
        )

    @staticmethod
    def _more_general(general, specific):
        contains = np.all(general.condition | ~specific.condition)
        return bool(contains and not np.array_equal(general.condition, specific.condition))

    def _correct_set_subsumption(self, correct_set):
        subsumer = None
        for rule in list(correct_set):
            if self._could_subsume(rule) and (subsumer is None or self._more_general(rule, subsumer)):
                subsumer = rule
        if subsumer is None:
            return
        changed = False
        for rule in list(correct_set):
            if rule is not subsumer and self._more_general(subsumer, rule):
                subsumer.numerosity += rule.numerosity
                if rule in self.population_:
                    self.population_.remove(rule)
                correct_set.remove(rule)
                self.subsumption_count_ += 1
                changed = True
        if changed:
            self._invalidate_conditions()

    def _select_offspring(self, positive_set):
        p = self.parameters_
        if p.tournament_probability == 0:
            weights = np.array([
                max(rule.fitness, 0.0) ** p.fitness_exponent * rule.matching_degree
                for rule in positive_set
            ])
            if weights.sum() <= 0:
                return positive_set[self.rng_.randint(len(positive_set))]
            point = self.rng_.rand() * weights.sum()
            return positive_set[min(np.searchsorted(np.cumsum(weights), point, side="right"), len(positive_set) - 1)]

        parent = None
        parent_score = -np.inf
        for rule in positive_set:
            score = (
                max(rule.fitness, 0.0) ** p.fitness_exponent
                * rule.matching_degree
                / rule.numerosity
            )
            if parent is None or parent_score < score:
                for _ in range(rule.numerosity):
                    if self.rng_.rand() < p.tournament_probability:
                        parent, parent_score = rule, score
                        break
        if parent is None:
            parent = positive_set[self.rng_.randint(len(positive_set))]
        return parent

    def _expand(self, allele):
        candidates = np.flatnonzero(~allele)
        if len(candidates):
            allele[self.rng_.choice(candidates)] = True

    def _contract(self, allele):
        candidates = np.flatnonzero(allele)
        if len(candidates):
            allele[self.rng_.choice(candidates)] = False

    def _shift(self, allele):
        candidates = np.flatnonzero(allele)
        if not len(candidates):
            return
        selected = int(self.rng_.choice(candidates))
        allele[selected] = False
        adjacent = selected - 1 if selected > 0 else selected + 1
        if adjacent < len(allele):
            allele[adjacent] = True

    def _mutate(self, rule):
        for allele in rule.condition:
            if self.rng_.rand() >= self.parameters_.mutation_probability:
                continue
            count = int(allele.sum())
            if count == len(allele):
                self._contract(allele)
            elif count == 1:
                self._expand(allele) if self.rng_.rand() < 0.5 else self._shift(allele)
            else:
                operation = self.rng_.randint(3)
                (self._expand, self._contract, self._shift)[operation](allele)

    def _crossover(self, first, second):
        swap = self.rng_.rand(*first.condition.shape) < 0.5
        values = first.condition[swap].copy()
        first.condition[swap] = second.condition[swap]
        second.condition[swap] = values
        for child in (first, second):
            for allele in child.condition:
                if not allele.any():
                    self._expand(allele)

    def _run_ga(self, correct_set):
        if not correct_set:
            return
        positive = [rule for rule in correct_set if rule.fitness >= 0]
        if not positive:
            return
        weighted_timestamp = sum(rule.time_stamp * rule.numerosity for rule in correct_set)
        numerosity = sum(rule.numerosity for rule in correct_set)
        if self.time_stamp_ - weighted_timestamp / numerosity <= self.parameters_.ga_threshold:
            return
        for rule in correct_set:
            rule.time_stamp = self.time_stamp_

        parent_1 = self._select_offspring(positive)
        parent_2 = self._select_offspring(positive)
        child_1, child_2 = copy.deepcopy(parent_1), copy.deepcopy(parent_2)
        for child in (child_1, child_2):
            child.identifier = self.global_id_
            self.global_id_ += 1
            child.numerosity = 1
            child.experience = 0.0
            child.correct_matching = np.zeros(self.n_classes_, dtype=float)
            child.correct_set_size_sum = 0.0
            child.correct_set_size_count = 0

        if self.rng_.rand() < self.parameters_.crossover_probability:
            self._crossover(child_1, child_2)
        for child in (child_1, child_2):
            self._mutate(child)
            if self.parameters_.ga_subsumption and self._could_subsume(parent_1) and self._more_general(parent_1, child):
                parent_1.numerosity += 1
                self.subsumption_count_ += 1
            elif self.parameters_.ga_subsumption and self._could_subsume(parent_2) and self._more_general(parent_2, child):
                parent_2.numerosity += 1
                self.subsumption_count_ += 1
            else:
                self._insert_in_population(child)
            self._delete_from_population()
        self._invalidate_conditions()

    def _insert_in_population(self, candidate):
        for rule in self.population_:
            if np.array_equal(rule.condition, candidate.condition):
                rule.numerosity += 1
                return
        self.population_.append(candidate)

    def _deletion_vote(self, rule, average_fitness):
        vote = rule.correct_set_size * rule.numerosity
        powered = max(rule.fitness, 0.0) ** self.parameters_.fitness_exponent
        if rule.experience > self.parameters_.deletion_threshold and powered < self.parameters_.deletion_fraction * average_fitness:
            vote *= average_fitness / max(powered, EPS)
        return vote

    def _delete_from_population(self):
        total = sum(rule.numerosity for rule in self.population_)
        if total <= self.parameters_.population_size:
            return
        average = sum(max(rule.fitness, 0.0) ** self.parameters_.fitness_exponent for rule in self.population_) / total
        votes = np.array([self._deletion_vote(rule, average) for rule in self.population_])
        if votes.sum() <= 0:
            selected = self.rng_.randint(len(self.population_))
        else:
            selected = min(np.searchsorted(np.cumsum(votes), self.rng_.rand() * votes.sum(), side="right"), len(votes) - 1)
        rule = self.population_[selected]
        rule.numerosity -= 1
        if rule.numerosity == 0:
            self.population_.pop(selected)
            self._invalidate_conditions()

    def _mass(self, state):
        match_set = self._match_set(state)
        experienced = [
            rule for rule in match_set
            if rule.experience > self.parameters_.exploitation_threshold
        ]
        if not experienced:
            mass = np.zeros(self.n_classes_ + 1, dtype=float)
            mass[-1] = 1.0
            return mass, 0.0

        combined = None
        last_conflict = 0.0
        for rule in experienced:
            mass = np.empty(self.n_classes_ + 1, dtype=float)
            mass[:-1] = rule.weights * rule.matching_degree
            mass[-1] = max(1.0 - mass[:-1].sum(), 0.0)
            mass /= max(mass.sum(), EPS)
            for _ in range(rule.numerosity):
                if combined is None:
                    combined = mass.copy()
                else:
                    combined, last_conflict = combine_dempster(combined, mass)
                    if last_conflict >= 1.0 - EPS:
                        zero = np.zeros(self.n_classes_ + 1, dtype=float)
                        zero[-1] = 1.0
                        return zero, 1.0
        return combined, last_conflict

    def predict_mass(self, X):
        Xn = self._transform(X)
        masses = np.vstack([self._mass(row)[0] for row in Xn])
        return masses

    def predict_proba(self, X):
        masses = self.predict_mass(X)
        proba = masses[:, :-1] + masses[:, [-1]] / self.n_classes_
        proba = np.clip(proba, EPS, None)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_set(self, X):
        masses = self.predict_mass(X)
        belief = masses[:, :-1]
        plausibility = belief + masses[:, [-1]]
        return plausibility >= belief.max(axis=1, keepdims=True) - EPS
