"""
Post-hoc fine-tuning of FERL node consequents (the user's "fix the confidences").

The tree structure (rules / partition) is frozen. Each non-root node's class
consequent is treated as a learnable parameter (in logit space, initialized from
the MLE frequencies) and fine-tuned on a held-out calibration split by gradient
descent on NLL, with an anchor regularizer toward the MLE to prevent overfitting.

Soft inference is linear in the consequents:
    P(c|x) = (M @ softmax(theta)) / M.sum(1)
so this is "temperature scaling's expressive big brother" -- one calibrated
consequent per rule, still interpretable.

Decisive test: raw FERL vs temperature scaling vs fine-tuned consequents, on
ECE / NLL / accuracy. Fine-tuning must beat temperature scaling to be worth it.
"""
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from ferl.core.tree_learning import FuzzyCART
from ferl.fuzzification.fuzzification_mdlp import learn_partitions_mdlp
from ferl.uncertainty.conformal_conditional import load_filtered, DATASETS
import ferl.eval._eval_accuracy as e

warnings.filterwarnings("ignore")
N_SEEDS = 10
EPS = 1e-8


def ece(probs, y, n_bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e_ = 0.0
    for b in range(n_bins):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.sum() > 0:
            e_ += (m.mean()) * abs(acc[m].mean() - conf[m].mean())
    return float(e_)


def nll(probs, y):
    p = np.clip(probs[np.arange(len(y)), y], EPS, 1.0)
    return float(-np.log(p).mean())


def soft_predict(M, cons):
    den = M.sum(axis=1)
    P = (M @ cons) / (den[:, None] + EPS)
    P[den <= EPS] = 1.0 / cons.shape[1]   # uniform fallback for zero-firing rows
    return P


def temperature_scale(M, cons, y_cal, M_cal):
    """One global temperature on the aggregated log-probabilities (fit on cal)."""
    P_cal = np.clip(soft_predict(M_cal, cons), EPS, 1.0)
    logit = torch.tensor(np.log(P_cal), dtype=torch.float32)
    yt = torch.tensor(y_cal, dtype=torch.long)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logit / torch.exp(logT), yt)
        loss.backward()
        return loss
    opt.step(closure)
    T = float(torch.exp(logT).item())
    P = np.clip(soft_predict(M, cons), EPS, 1.0)
    sl = np.log(P) / T
    sl -= sl.max(axis=1, keepdims=True)
    ex = np.exp(sl)
    return ex / ex.sum(axis=1, keepdims=True)


def finetune_consequents(M_cal, y_cal, cons0, n_classes, lam=1.0, epochs=300, lr=0.05):
    """Fine-tune per-node consequent logits on the calibration split."""
    theta0 = torch.tensor(np.log(np.clip(cons0, EPS, 1.0)), dtype=torch.float32)
    theta = theta0.clone().requires_grad_(True)
    # Only firing rows carry consequent gradients; zero-firing rows always fall
    # back to the prior and cannot be improved here, so drop them from the loss.
    fires = M_cal.sum(axis=1) > EPS
    Mt = torch.tensor(M_cal[fires], dtype=torch.float32)
    yt = torch.tensor(y_cal[fires], dtype=torch.long)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        q = F.softmax(theta, dim=1)                 # (K, C) per-node consequents
        P = (Mt @ q) / (Mt.sum(1, keepdim=True) + EPS)
        loss = F.nll_loss(torch.log(P + EPS), yt) + lam * ((theta - theta0) ** 2).mean()
        loss.backward()
        opt.step()
    return F.softmax(theta.detach(), dim=1).numpy()


def _ds_dempster_torch(Me, cons, C):
    """Dempster combine discounted firing Me (N,K) with consequents cons (K,C).
    Returns (m_c, m_theta) for the singleton+Theta mass structure."""
    term = Me[:, :, None] * cons[None, :, :] + (1.0 - Me)[:, :, None]   # (N,K,C)
    Qc = term.prod(1)                                                   # (N,C)
    Qt = (1.0 - Me).prod(1)                                             # (N,)
    mc = torch.clamp(Qc - Qt[:, None], min=0.0)
    tot = (mc.sum(1) + Qt).clamp_min(EPS)
    return mc / tot[:, None], Qt / tot


def _pinball(resid, tau):
    return torch.maximum(tau * resid, (tau - 1.0) * resid)


def finetune_reliability(M_cal, y_cal, node_support, node_depth, cons, n_classes,
                         loss="evidential", lam=0.1, alpha=0.1, beta=0.5,
                         epochs=400, lr=0.05):
    """Learn a per-node reliability r_n = sigmoid(w . [logN, depth] + b), tree
    frozen, on the calibration split -- the supervised, ensemble-free fuzzy->mass
    map. Discounted firing r_n*mu_n is Dempster-combined; we optimise either:

      loss='evidential' : Brier(betp,y) + lam * KL(Dir(alpha_wrong) || Dir(1))
                          -- routes mass to Theta where the model errs (the
                          "Laplacian"/uniform-Dirichlet pull). Calibrates BetP.
      loss='interval'   : pinball quantile loss making [Bel,Pl] a central
                          (1-alpha) interval for the one-vs-rest class indicator,
                          plus beta * mean width -- directly calibrates imprecise
                          coverage without the trivial widen-to-[0,1] solution.

    Returns r (K,) -- pass to predict_ds(reliability_vec=r). Low-dim (3 params),
    so it cannot overfit; it learns the discount the Dirichlet a0 only guessed.
    """
    f = np.stack([np.log(np.asarray(node_support) + 1.0),
                  np.asarray(node_depth, dtype=float)], axis=1)         # (K,2)
    f = (f - f.mean(0)) / (f.std(0) + EPS)
    ft = torch.tensor(f, dtype=torch.float32)
    Mt = torch.tensor(M_cal, dtype=torch.float32)
    const = torch.tensor(cons, dtype=torch.float32)
    yt = torch.tensor(y_cal, dtype=torch.long)
    onehot = F.one_hot(yt, n_classes).float()
    w = torch.zeros(ft.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    tau_lo, tau_hi = alpha / 2.0, 1.0 - alpha / 2.0
    for _ in range(epochs):
        opt.zero_grad()
        r = torch.sigmoid(ft @ w + b)                                  # (K,)
        m_c, m_th = _ds_dempster_torch(Mt * r[None, :], const, n_classes)
        if loss == "interval":
            bel = m_c                                                  # (N,C)
            pl = m_c + m_th[:, None]
            pinball = (_pinball(onehot - bel, tau_lo) + _pinball(onehot - pl, tau_hi)).mean()
            l = pinball + beta * m_th.mean()                           # width penalty (Pl-Bel = m_theta)
        else:                                                          # evidential
            betp = m_c + m_th[:, None] / n_classes
            brier = ((betp - onehot) ** 2).sum(1).mean()
            a_wrong = 1.0 + n_classes * m_c * (1.0 - onehot)           # remove true-class evidence
            S = a_wrong.sum(1)
            kl = (torch.lgamma(S) - torch.lgamma(torch.tensor(float(n_classes)))
                  - torch.lgamma(a_wrong).sum(1)
                  + ((a_wrong - 1.0) * (torch.digamma(a_wrong) - torch.digamma(S)[:, None])).sum(1))
            l = brier + lam * kl.mean()
        l.backward()
        opt.step()
    return torch.sigmoid(ft @ w + b).detach().numpy()


def main():
    print(f"Post-hoc consequent recalibration | {N_SEEDS} seeds")
    print("cells = ECE / NLL / acc  (ECE,NLL lower=better)\n")
    methods = ["raw", "temp-scale", "finetune"]
    header = "dataset".ljust(12) + "".join(m.ljust(24) for m in methods)
    print(header)
    agg = {m: {"ece": [], "nll": [], "acc": []} for m in methods}
    for name in DATASETS:
        try:
            X, y = load_filtered(name)
        except Exception as ex:
            print(f"{name.ljust(12)} SKIP ({ex})"); continue
        per = {m: {"ece": [], "nll": [], "acc": []} for m in methods}
        for seed in range(N_SEEDS):
            Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
            Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=seed, stratify=ytmp)
            clf = FuzzyCART(learn_partitions_mdlp(Xtr, ytr), max_rules=20)
            clf.fit(Xtr, ytr)
            C = len(clf.classes_)
            M_te, cons, _ = clf.node_activation_matrix(Xte)
            M_cal, _, _ = clf.node_activation_matrix(Xcal)

            P_raw = soft_predict(M_te, cons)
            P_temp = temperature_scale(M_te, cons, ycal, M_cal)
            cons_ft = finetune_consequents(M_cal, ycal, cons, C)
            P_ft = soft_predict(M_te, cons_ft)

            for m, P in zip(methods, [P_raw, P_temp, P_ft]):
                per[m]["ece"].append(ece(P, yte))
                per[m]["nll"].append(nll(P, yte))
                per[m]["acc"].append((P.argmax(1) == yte).mean())
        row = name.ljust(12)
        for m in methods:
            for k in per[m]:
                agg[m][k].append(np.mean(per[m][k]))
            row += (f"{np.mean(per[m]['ece']):.3f}/{np.mean(per[m]['nll']):.3f}/"
                    f"{np.mean(per[m]['acc']):.3f}").ljust(24)
        print(row)
    print("-" * len(header))
    mrow = "MEAN".ljust(12)
    for m in methods:
        mrow += (f"{np.mean(agg[m]['ece']):.3f}/{np.mean(agg[m]['nll']):.3f}/"
                 f"{np.mean(agg[m]['acc']):.3f}").ljust(24)
    print(mrow)


if __name__ == "__main__":
    main()
