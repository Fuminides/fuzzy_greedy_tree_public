"""
Evidential Deep Learning (Sensoy et al., NeurIPS 2018) -- the neural evidential
classifier whose name FERL's "evidential fuzzy rule tree" framing invites
comparison against. An MLP emits non-negative per-class evidence e_k; the Dirichlet
alpha_k = e_k + 1 gives the predictive mean p_k = alpha_k / S (S = sum alpha) and a
vacuity / ignorance u = K / S directly analogous to FERL's DS ignorance.

Trained with the Bayes-risk cross-entropy (digamma) loss plus an annealed
KL-to-uniform regulariser that drives evidence to zero on misleading inputs.

Exposes the benchmark2 protocol: fit / predict / predict_proba / classes_ /
complexity_ (parameter count). Scored as a probabilistic model (conformal-APS
sets), so predict_proba is the Dirichlet mean -- a fair head-to-head with the
other point predictors.
"""
import numpy as np


class EDL:
    def __init__(self, hidden=64, epochs=200, lr=1e-3, weight_decay=1e-4,
                 anneal_epochs=20, random_state=0):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.anneal_epochs = anneal_epochs
        self.random_state = random_state

    def _build(self, D, K):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(D, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, K),
        )

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.random_state)
        X = np.asarray(X, float)
        self.classes_, yi = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        self.mu_ = X.mean(0); self.sd_ = X.std(0) + 1e-8
        Xs = (X - self.mu_) / self.sd_

        Xt = torch.tensor(Xs, dtype=torch.float32)
        Y = torch.zeros(len(yi), K); Y[np.arange(len(yi)), yi] = 1.0
        self.net_ = self._build(X.shape[1], K)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        def kl_uniform(alpha):                          # KL(Dir(alpha) || Dir(1))
            S = alpha.sum(1)
            t1 = torch.lgamma(S) - torch.lgamma(alpha).sum(1) - torch.lgamma(
                torch.tensor(float(K)))
            t2 = ((alpha - 1.0) * (torch.digamma(alpha)
                                   - torch.digamma(S).unsqueeze(1))).sum(1)
            return t1 + t2

        self.net_.train()
        for ep in range(self.epochs):
            opt.zero_grad()
            evidence = torch.nn.functional.softplus(self.net_(Xt))
            alpha = evidence + 1.0
            S = alpha.sum(1, keepdim=True)
            # Bayes-risk cross-entropy (digamma form)
            ce = (Y * (torch.digamma(S) - torch.digamma(alpha))).sum(1)
            lam = min(1.0, (ep + 1) / self.anneal_epochs)
            alpha_tilde = Y + (1.0 - Y) * alpha         # strip true-class evidence
            loss = (ce + lam * kl_uniform(alpha_tilde)).mean()
            loss.backward(); opt.step()

        self.complexity_ = float(sum(p.numel() for p in self.net_.parameters()))
        return self

    def _alpha(self, X):
        import torch
        Xs = (np.asarray(X, float) - self.mu_) / self.sd_
        self.net_.eval()
        with torch.no_grad():
            ev = torch.nn.functional.softplus(self.net_(torch.tensor(Xs, dtype=torch.float32)))
        return ev.numpy() + 1.0

    def predict_proba(self, X):
        a = self._alpha(X)
        return a / a.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
