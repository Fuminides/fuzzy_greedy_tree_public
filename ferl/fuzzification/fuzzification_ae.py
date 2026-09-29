"""
Supervised fuzzification via a fuzzy autoencoder (PyTorch).

The encoder *is* the fuzzification layer: per feature, K Gaussian membership
functions with learnable centers/widths, normalized to a partition of unity.
The decoder is a centroid defuzzifier that reuses the encoder's centers, so the
reconstruction error directly measures fuzzification fidelity and cannot be
"cheated" by a high-capacity decoder. An optional linear classifier head makes
the partition discriminative.

Objective (an Information-Bottleneck-style trade-off):

    L = L_recon(centroid defuzz)  +  lam * L_CE(head)  +  reg

With ``lam = 0`` this reduces to the pure-reconstruction ("most bijective")
fuzzification; ``lam > 0`` places set boundaries where classes separate.

The learned per-feature Gaussian sets are exported as ex_fuzzy
``fuzzyVariable`` objects, a drop-in replacement for
``ex_fuzzy.utils.construct_partitions`` that feeds straight into ``FuzzyCART``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ex_fuzzy import fuzzy_sets as fs

_TERM_NAMES = ["low", "lowmed", "med", "medhigh", "high"]


def _term_name(k: int, K: int) -> str:
    if K <= len(_TERM_NAMES):
        # spread evenly across the canonical names
        idx = int(round(k * (len(_TERM_NAMES) - 1) / max(K - 1, 1)))
        return f"{_TERM_NAMES[idx]}_{k}"
    return f"term_{k}"


class FuzzyAutoencoder(nn.Module):
    """Per-feature Gaussian fuzzifier + centroid defuzzifier + linear head.

    All parameters live in a per-feature normalized [0, 1] space; export to
    ex_fuzzy converts them back to the original feature scale.

    Parameters
    ----------
    n_features : int
        Number of input features ``d``.
    n_partitions : int
        Number of linguistic terms ``K`` per feature.
    n_classes : int
        Number of target classes (for the supervised head).
    init_centers : np.ndarray, optional
        ``(d, K)`` initial centers in [0, 1] (e.g. quantiles). Defaults to
        evenly spaced centers.
    min_sigma : float
        Lower bound on the (normalized) Gaussian width, to avoid spikes.
    """

    def __init__(self, n_features: int, n_partitions: int, n_classes: int,
                 init_centers: np.ndarray | None = None, min_sigma: float = 0.03):
        super().__init__()
        self.d = n_features
        self.K = n_partitions
        self.min_sigma = min_sigma

        if init_centers is None:
            grid = np.linspace(0.0, 1.0, n_partitions)[None, :].repeat(n_features, 0)
            init_centers = grid
        self.centers = nn.Parameter(torch.tensor(init_centers, dtype=torch.float32))

        # Width initialized to roughly the inter-center spacing.
        init_spacing = 1.0 / max(n_partitions - 1, 1)
        init_log_sigma = np.log(np.full((n_features, n_partitions), init_spacing * 0.6))
        self._log_sigma = nn.Parameter(torch.tensor(init_log_sigma, dtype=torch.float32))

        # Discriminative head over the flattened membership vector (d*K).
        self.head = nn.Linear(n_features * n_partitions, n_classes)

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self._log_sigma) + self.min_sigma

    def memberships(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, d) -> mu: (N, d, K), normalized per feature (partition of unity)."""
        # (N, d, 1) - (1, d, K)
        diff = x.unsqueeze(-1) - self.centers.unsqueeze(0)
        g = torch.exp(-(diff ** 2) / (2.0 * self.sigma.unsqueeze(0) ** 2))
        mu = g / (g.sum(dim=-1, keepdim=True) + 1e-8)
        return mu

    def decode(self, mu: torch.Tensor) -> torch.Tensor:
        """Centroid defuzzification: x_hat[:, j] = sum_k mu[:, j, k] * c[j, k]."""
        return (mu * self.centers.unsqueeze(0)).sum(dim=-1)

    def forward(self, x: torch.Tensor):
        mu = self.memberships(x)
        x_hat = self.decode(mu)
        logits = self.head(mu.reshape(x.shape[0], -1))
        return mu, x_hat, logits


def _spread_reg(centers: torch.Tensor, min_gap: float) -> torch.Tensor:
    """Penalize adjacent centers (sorted per feature) closer than min_gap."""
    sorted_c, _ = torch.sort(centers, dim=-1)
    gaps = sorted_c[:, 1:] - sorted_c[:, :-1]
    return F.relu(min_gap - gaps).pow(2).mean()


def train_autoencoder(X: np.ndarray, y: np.ndarray, n_partitions: int = 3,
                      lam: float = 1.0, epochs: int = 400, lr: float = 0.05,
                      lam_spread: float = 1.0, lam_anchor: float = 1.0,
                      seed: int = 0, verbose: bool = False):
    """Fit the fuzzy autoencoder on (X, y) and return (model, scaler).

    ``scaler`` is ``(mins, ranges)`` for per-feature [0, 1] normalization.

    ``lam_anchor`` keeps centers near their quantile initialization so the
    supervised head refines boundaries locally instead of collapsing the
    terms together (which destroys the range coverage the tree needs).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = np.asarray(X, dtype=np.float64)
    mins = X.min(axis=0)
    ranges = X.max(axis=0) - mins
    ranges[ranges == 0] = 1.0  # constant features -> avoid div by zero
    Xn = (X - mins) / ranges

    n_classes = int(np.max(y)) + 1
    init_centers = np.stack([np.quantile(Xn[:, j], np.linspace(0.0, 1.0, n_partitions))
                             for j in range(Xn.shape[1])], axis=0)

    model = FuzzyAutoencoder(Xn.shape[1], n_partitions, n_classes, init_centers=init_centers)
    Xt = torch.tensor(Xn, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    anchor = torch.tensor(init_centers, dtype=torch.float32)  # quantile centers
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    min_gap = 0.5 / max(n_partitions - 1, 1)
    for ep in range(epochs):
        opt.zero_grad()
        mu, x_hat, logits = model(Xt)
        recon = F.mse_loss(x_hat, Xt)
        ce = F.cross_entropy(logits, yt) if lam > 0 else torch.tensor(0.0)
        reg = lam_spread * _spread_reg(model.centers, min_gap)
        anchor_pen = lam_anchor * F.mse_loss(model.centers, anchor)
        loss = recon + lam * ce + reg + anchor_pen
        loss.backward()
        opt.step()
        if verbose and (ep % 100 == 0 or ep == epochs - 1):
            print(f"    ep{ep:4d} loss={loss.item():.4f} recon={recon.item():.4f} ce={float(ce):.4f}")

    return model, (mins, ranges)


def model_to_partitions(model: FuzzyAutoencoder, scaler) -> list:
    """Export the learned per-feature Gaussian sets as ex_fuzzy fuzzyVariables."""
    mins, ranges = scaler
    centers = model.centers.detach().numpy()
    sigmas = model.sigma.detach().numpy()
    d, K = centers.shape

    variables = []
    for j in range(d):
        order = np.argsort(centers[j])
        lo = float(mins[j])
        hi = float(mins[j] + ranges[j])
        domain = [lo, hi]
        sets = []
        for rank, k in enumerate(order):
            mean_raw = float(centers[j, k] * ranges[j] + mins[j])
            std_raw = float(max(sigmas[j, k] * ranges[j], 1e-6))
            sets.append(fs.gaussianFS(name=_term_name(rank, K),
                                      membership_parameters=[mean_raw, std_raw],
                                      domain=domain))
        variables.append(fs.fuzzyVariable(name=f"feature_{j}", fuzzy_sets=sets))
    return variables


def learn_partitions(X: np.ndarray, y: np.ndarray, n_partitions: int = 3,
                     lam: float = 1.0, epochs: int = 400, lr: float = 0.05,
                     seed: int = 0, verbose: bool = False) -> list:
    """Drop-in replacement for ex_fuzzy.utils.construct_partitions.

    Returns a list[fuzzyVariable] with supervised, autoencoder-learned Gaussian
    membership functions. Set ``lam=0`` for the pure-reconstruction variant.
    """
    model, scaler = train_autoencoder(X, y, n_partitions=n_partitions, lam=lam,
                                      epochs=epochs, lr=lr, seed=seed, verbose=verbose)
    return model_to_partitions(model, scaler)


if __name__ == "__main__":
    # Quick self-test on iris.
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    parts = learn_partitions(X, y, n_partitions=3, lam=1.0, epochs=300, verbose=True)
    print("built", len(parts), "fuzzy variables")
    v = parts[0]
    print("feature 0 sets:", [(s.name, [round(p, 3) for p in s.membership_parameters]) for s in v])
    print("membership of feature0 at its median:",
          v[0].membership(np.array([float(np.median(X[:, 0]))])))
