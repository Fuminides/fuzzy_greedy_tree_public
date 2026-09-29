"""The FERL variants described in the paper (Table 1) must
match the configurations the experiments use. Run: pytest tests/test_variant_matrix.py
"""
import numpy as np
import pytest

from ferl.core.learned_tree import LearnedFuzzyTree
from ferl.pipeline.ferl_pipeline import make


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int) + (X[:, 2] > 1).astype(int)
    return X, y


def test_compact_configuration():
    m = make("ferl-compact", random_state=0)
    assert (m.partition, m.split_mode, m.prediction_mode) == ("quantile", "fixed", "soft")
    assert (m.max_rules, m.max_depth, m.patience, m.min_improvement) == (20, 5, 3, 0.01)
    assert m.calibration is None


def test_medium_configuration():
    m = make("ferl-medium", random_state=0)
    assert (m.split_mode, m.learned_width, m.prediction_mode) == ("learned", "bootstrap", "soft")
    assert (m.max_rules, m.max_depth, m.patience, m.min_improvement) == (150, 12, 16, 0.0)
    assert m.calibration is None


def test_compact_and_medium_split_on_cci(data):
    X, y = data
    for config in ("ferl-compact", "ferl-medium"):
        tree = make(config, random_state=0).fit(X, y).tree_
        assert tree.target_metric == "cci"
    assert make("ferl-medium").fit(X, y).tree_.learned_n_boot == 25


def test_deep_configuration():
    m = make("ferl-deep")
    assert isinstance(m, LearnedFuzzyTree)
    p = m.get_params()
    assert (p["max_depth"], p["min_leaf_w"], p["n_boot"]) == (12, 2.0, 25)
    assert (p["width"], p["criterion"], p["bounded_support"], p["oob_margin"]) == (
        "bootstrap", "gini", True, 1.0)
    assert not hasattr(m, "max_rules")          # no rule budget


def test_deep_point_label_is_leaf_soft_vote(data):
    X, y = data
    m = make("ferl-deep").fit(X, y)
    M, cons, names, _ = m.node_activation_matrix(X)
    leaves = np.flatnonzero(m.leaf_mask(names))
    vote = M[:, leaves] @ cons[leaves]
    np.testing.assert_allclose(m.predict_proba(X), vote / vote.sum(1, keepdims=True), atol=1e-12)


def test_deep_sets_use_leaves_only_dempster(data):
    X, y = data
    m = make("ferl-deep").fit(X, y)
    _, bel, pl, _ = m.predict_ds(X, rule="dempster", leaves_only=True)
    np.testing.assert_array_equal(m.predict_set(X), pl >= bel.max(1, keepdims=True) - 1e-12)


def test_deep_width_floor(data):
    X, y = data
    m = make("ferl-deep").fit(X, y)

    def internal(node):
        if not node["leaf"]:
            yield node
            yield from internal(node["L"])
            yield from internal(node["R"])

    nodes = list(internal(m.root_))
    assert nodes and all(n["h"] > 0 for n in nodes)


def test_tuned_width_selects_from_grid(data):
    from ferl.core.learned_tree import LearnedFuzzyTreeCV
    X, y = data
    m = LearnedFuzzyTreeCV(random_state=0).fit(X, y)
    assert m.selected_width_ in ("bootstrap", 0.25, 0.5, 1.0)
    assert len(m.width_scores_) == 4 and m.width == m.selected_width_
    assert m.selected_width_ == m.width_grid[int(np.argmin(np.round(m.width_scores_, 12)))]


def test_nominal_encoding_default_is_ordinal():
    from ferl.pipeline import run_configs
    assert run_configs.NOMINAL == "ordinal" or __import__("os").environ.get("FERL_NOMINAL")
