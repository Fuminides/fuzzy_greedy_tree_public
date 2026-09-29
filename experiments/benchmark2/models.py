"""
Estimator registry for the SOTA benchmark. Each model is built fresh by name and
exposes the common protocol via sklearn-style fit / predict_proba / classes_.
Complexity (interpretability proxy) is extracted best-effort per model.

M1 set: FERL (ours) + interpretable (CART, C4.5, RuleFit, FIGS) + ceiling
(RF, GBDT, LogReg). GOSDT is a wired-but-skipped slot (needs the `gosdt` build).
Credal / fuzzy / FURIA models slot in behind the same protocol in later milestones.
Comparison-paper methods from ``Comparison papers/`` are split into runnable
local wrappers and optional public-code adapters.
"""
import warnings
import numpy as np

warnings.filterwarnings("ignore")

INTERPRETABLE = ["CART", "C45", "RuleFit", "FIGS"]
CEILING = ["RF", "GBDT", "LogReg", "MLP", "NGBoost"]
OURS = ["FERL-compact", "FERL-enhanced", "FERL-credal",
        "FERL-medium", "FERL-deep", "FERL-deep-tuned"]
CREDAL = ["CredalC45", "ICDT", "NCC"]                  # M2
ENSEMBLE = ["FERL-forest", "FERL-forest-ev"]          # M5: fuzzy "random forest"
NEURAL = ["EDL"]                                       # neural evidential competitor
FUZZY = ["FURIA"]                                      # M4: fuzzy-rule lineage (ex_fuzzy later)
LEARNED = ["FERL-medium", "FERL-deep"]         # learned-threshold fuzzification family
PAPER_PROXY = ["SampledRuleList"]                      # sampled greedy ablation, not SamRuLe
PAPER_LOCAL = ["FuzzyUCS-DS", "NeuRules"]             # self-contained public-source ports
PAPER_PUBLIC = ["SamRuLe", "SamRuLe-OVR", "RRL", "RL-Net"]  # require public repos
M1_MODELS = ["FERL-compact"] + INTERPRETABLE + CEILING         # GOSDT added once installed
M2_MODELS = ["FERL-credal"] + CREDAL
ALL_MODELS = M1_MODELS + LEARNED + M2_MODELS + ENSEMBLE + NEURAL + FUZZY + PAPER_PROXY + PAPER_LOCAL
SET_VALUED = ["FERL-credal", "ICDT", "NCC", "FERL-forest-ev", "FERL-deep",
              "FERL-deep-tuned", "FuzzyUCS-DS"]


class _FERLCredal:
    """FERL with top-p routing -> native credal set (interval dominance on Bel/Pl)
    and pignistic point estimate. tau calibrated in-sample (smallest preserving acc)."""
    TAUS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)

    def fit(self, X, y):
        from ferl.pipeline.ferl_pipeline import make
        self.tree = make("ferl-compact", random_state=0).fit(X, y).tree_
        self.classes_ = self.tree.classes_
        idx = np.array([list(self.classes_).index(v) for v in y])
        base = (self.tree.predict_ds(X)[0].argmax(1) == idx).mean()
        self.tau = 1.0
        for t in self.TAUS:
            if (self.tree.predict_ds(X, top_p=t)[0].argmax(1) == idx).mean() >= base - 0.01:
                self.tau = t; break
        self.complexity_ = float(self.tree.tree_rules)
        return self

    def predict_proba(self, X):
        return self.tree.predict_ds(X, top_p=self.tau)[0]

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def predict_set(self, X):
        _, bel, pl, _ = self.tree.predict_ds(X, top_p=self.tau)
        return pl >= bel.max(1, keepdims=True) - 1e-12


class _NGBoost:
    """NGBoost natural-gradient probabilistic boosting (k_categorical head set at
    fit time, since the registry builds models before seeing the class count)."""

    def fit(self, X, y):
        from ngboost import NGBClassifier
        from ngboost.distns import k_categorical
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.model = NGBClassifier(Dist=k_categorical(len(self.classes_)),
                                   n_estimators=100, learning_rate=0.01, verbose=False)
        self.model.fit(np.asarray(X, float), yi.astype(int))
        try:
            self.complexity_ = float(sum(t.get_n_leaves() for stage in self.model.base_models
                                         for t in stage))
        except Exception:
            self.complexity_ = np.nan
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X, float))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def build(name):
    if name in ("FERL-compact",):
        from ferl.pipeline.ferl_pipeline import make
        return make("ferl-compact", random_state=0)
    if name == "FERL-enhanced":
        from ferl.pipeline.ferl_pipeline import make
        return make("ferl-enhanced", random_state=0)
    if name == "FERL-medium":                         # compact learned-threshold, native credal
        from ferl.pipeline.ferl_pipeline import make
        return make("ferl-medium", random_state=0)
    if name == "FERL-deep":                             # deep learned-threshold, max accuracy
        from ferl.core.learned_tree import LearnedFuzzyTree
        return LearnedFuzzyTree(random_state=0)
    if name == "FERL-deep-tuned":                       # FERL-deep, band width by inner CV
        from ferl.core.learned_tree import LearnedFuzzyTreeCV
        return LearnedFuzzyTreeCV(random_state=0)
    if name == "CART":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=0)
    if name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1)
    if name == "GBDT":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(random_state=0)
    if name == "LogReg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000)
    if name == "MLP":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64,), max_iter=500,
                          early_stopping=True, random_state=0),
        )
    if name == "NGBoost":
        return _NGBoost()
    if name == "RuleFit":
        from imodels import RuleFitClassifier
        from sklearn.multiclass import OneVsRestClassifier      # imodels RuleFit is binary-only
        return OneVsRestClassifier(RuleFitClassifier(random_state=0, max_rules=30))
    if name == "C45":
        from imodels import C45TreeClassifier
        from sklearn.multiclass import OneVsRestClassifier      # imodels C45 proba is binary-only
        return OneVsRestClassifier(C45TreeClassifier())
    if name == "FIGS":
        from imodels import FIGSClassifier
        return FIGSClassifier(max_rules=30)
    if name == "GOSDT":
        from imodels import HSOptimalTreeClassifier      # needs `gosdt` backend
        return HSOptimalTreeClassifier()
    if name == "FERL-credal":
        return _FERLCredal()
    if name in ("FERL-forest", "FERL-forest-ev"):
        import ferl_forest as FF
        return FF.FERLForest(combine="evidence" if name.endswith("-ev") else "avg")
    if name == "EDL":
        import edl
        return edl.EDL()
    if name == "FURIA":
        import furia
        return furia.FURIA(random_state=0)
    if name == "SampledRuleList":
        import comparison_methods as CMP
        return CMP.SampledGreedyRuleList(random_state=0)
    if name in ("SamRuLe", "SamRuLe-OVR"):
        import comparison_methods as CMP
        return CMP.SamRuLeExternal(allow_multiclass=name.endswith("-OVR"), random_state=0)
    if name == "RRL":
        import comparison_methods as CMP
        return CMP.RRLExternal(random_state=0)
    if name == "RL-Net":
        import comparison_methods as CMP
        return CMP.RLNetExternal(random_state=0)
    if name == "FuzzyUCS-DS":
        import fuzzy_ucs
        return fuzzy_ucs.FuzzyUCSDS(random_state=0)
    if name == "NeuRules":
        import neurules
        return neurules.NeuRules(random_state=0)
    if name in ("CredalC45", "ICDT", "NCC"):
        import credal_models as CM
        if name == "NCC":
            return CM.NCC()
        return CM.CredalTree(imprecise=(name == "ICDT"))
    raise KeyError(name)


def complexity(name, est):
    """#rules / #leaves / #weights — interpretability proxy (best-effort)."""
    try:
        if name in ("FERL-compact", "FERL-enhanced", "FERL-medium"):
            return float(est.tree_.tree_rules)
        if name == "CART":
            return float(est.get_n_leaves())
        if name in ("RF", "GBDT"):
            ests = est.estimators_.ravel() if name == "GBDT" else est.estimators_
            return float(sum(e.get_n_leaves() for e in ests))
        if name == "LogReg":
            return float(est.coef_.size)
        if name == "MLP":
            mlp = est.named_steps.get("mlpclassifier", est)
            return float(sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_))
        if name in ("RuleFit", "C45"):                         # OneVsRest -> sum sub-models
            subs = getattr(est, "estimators_", [])
            vals = [getattr(s, "complexity_", np.nan) for s in subs]
            return float(np.nansum(vals)) if vals else np.nan
        if name in ("FIGS", "GOSDT", "CredalC45", "ICDT", "NCC", "FERL-credal", "NGBoost",
                    "FERL-forest", "FERL-forest-ev", "EDL", "FURIA", "FERL-deep",
                    "FERL-deep-tuned"):
            return float(getattr(est, "complexity_", np.nan))
        if name in ("SampledRuleList", "SamRuLe", "SamRuLe-OVR", "RRL", "RL-Net", "FuzzyUCS-DS", "NeuRules"):
            return float(getattr(est, "complexity_", np.nan))
    except Exception:
        return np.nan
    return np.nan


def classes_of(est):
    return est.classes_
