"""5th-order Newton-Schulz polar decomposition for Muon orthogonalization.

Given a momentum matrix M, Muon approximates the polar factor
    polar(M) = U V^T
where M = U Σ V^T is the compact SVD. Singular values are mapped toward 1,
so the update is isometric along singular directions and condition-number
explosion is neutralized — without an O(min(m,n)^2 max(m,n)) SVD.

The quintic polynomial
    p(x) = a x + b x^3 + c x^5
with (a, b, c) = (3.4445, -4.7750, 2.0315) is applied iteratively to the
Frobenius-normalized matrix via pure GEMM operations.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Coefficients from Jordan et al. (2024) / Keller Jordan Muon reference.
# Chosen so that p maps singular values in (0, 1] toward 1 under iteration.
NS5_A: float = 3.4445
NS5_B: float = -4.7750
NS5_C: float = 2.0315

EPS: float = 1e-8


def frobenius_normalize(X: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Return X / (||X||_F + eps). Guarantees spectral radius ≤ 1 for NS start."""
    X = np.asarray(X, dtype=np.float64)
    norm = float(np.linalg.norm(X, ord="fro"))
    return X / (norm + eps)


def aspect_ratio_scale(rows: int, cols: int) -> float:
    """Operator-norm compensation: max(1, sqrt(m/n)) for m×n matrices.

    When m > n the polar factor has operator norm 1 in the tall direction;
    scaling by sqrt(m/n) equalizes effective step size across shapes.
    """
    if cols <= 0:
        return 1.0
    return float(max(1.0, np.sqrt(rows / cols)))


def newton_schulz5(
    G: np.ndarray,
    steps: int = 5,
    eps: float = EPS,
    coefficients: Tuple[float, float, float] = (NS5_A, NS5_B, NS5_C),
    apply_aspect_scale: bool = True,
) -> np.ndarray:
    """Approximate the orthogonal polar factor of G via 5th-order Newton-Schulz.

    Algorithm
    ---------
    1. If m < n, work on G^T so the Gram matrix is the smaller square factor.
    2. X_0 = G / (||G||_F + eps)
    3. For k = 0 .. steps-1:
           A = X X^T
           B = b A + c A^2
           X <- a X + B X
    4. Restore transpose if needed.
    5. Optionally multiply by aspect-ratio scale max(1, sqrt(m/n)).

    Parameters
    ----------
    G :
        Momentum / gradient matrix of shape (m, n), m,n ≥ 1.
    steps :
        Number of Newton-Schulz iterations (default 5; 3–10 typical).
    eps :
        Floor for Frobenius normalization.
    coefficients :
        (a, b, c) for p(x) = a x + b x^3 + c x^5.
    apply_aspect_scale :
        Whether to apply the Muon aspect-ratio operator scale.

    Returns
    -------
    O : np.ndarray
        Approximate polar factor, same shape as G. Singular values ≈ 1.
    """
    if steps < 1:
        raise ValueError(f"steps must be ≥ 1, got {steps}")

    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError(f"G must be 2D, got shape {G.shape}")
    m, n = G.shape
    if m == 0 or n == 0:
        return G.copy()

    a, b, c = coefficients
    # Transpose tall matrices so A = X X^T is the smaller Gram (Jordan et al.).
    transposed = m > n
    X = G.T.copy() if transposed else G.copy()
    X = frobenius_normalize(X, eps=eps)

    for _ in range(steps):
        # A = X X^T  (square on the smaller side after possible transpose)
        A = X @ X.T
        # B = b A + c A^2
        A2 = A @ A
        B = b * A + c * A2
        # X <- a X + B X = (a I + B) X
        X = a * X + B @ X

    if transposed:
        X = X.T

    if apply_aspect_scale:
        X = X * aspect_ratio_scale(m, n)

    return X


def polar_factor_svd(G: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Exact polar factor U V^T via compact SVD (reference oracle).

    For G = U Σ V^T, polar(G) = U V^T (with zero singular directions dropped
    via the standard compact SVD convention).
    """
    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError(f"G must be 2D, got shape {G.shape}")
    if G.size == 0:
        return G.copy()

    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    # Zero out numerically vanished singular directions to keep polarity stable.
    mask = S > eps
    if not np.any(mask):
        # Degenerate: return zeros of correct shape via thin factors.
        return U @ Vt * 0.0
    U = U[:, mask]
    Vt = Vt[mask, :]
    return U @ Vt


def orthogonal_error(O: np.ndarray, eps: float = EPS) -> float:
    """Measure deviation of O from partial-isometry: ||O O^T - I||_F / sqrt(r)
    (or ||O^T O - I||_F when wide). Lower is more orthogonal.
    """
    O = np.asarray(O, dtype=np.float64)
    m, n = O.shape
    if m <= n:
        Gram = O @ O.T
        r = m
    else:
        Gram = O.T @ O
        r = n
    I = np.eye(r, dtype=np.float64)
    return float(np.linalg.norm(Gram - I, ord="fro") / (np.sqrt(r) + eps))


def singular_value_spread(G: np.ndarray, eps: float = EPS) -> Tuple[float, float, float]:
    """Return (σ_min, σ_max, condition κ = σ_max / σ_min) of G.

    For a perfect polar factor, σ_min ≈ σ_max ≈ 1 and κ ≈ 1.
    """
    G = np.asarray(G, dtype=np.float64)
    if G.size == 0:
        return 0.0, 0.0, 1.0
    S = np.linalg.svd(G, compute_uv=False)
    smax = float(S[0]) if S.size else 0.0
    smin = float(S[-1]) if S.size else 0.0
    kappa = smax / (smin + eps)
    return smin, smax, kappa


def ns5_vs_svd_error(
    G: np.ndarray,
    steps: int = 5,
    apply_aspect_scale: bool = False,
) -> float:
    """Relative Frobenius error between NS5 polar approx and exact SVD polar factor.

    Aspect scaling is disabled by default so both targets match UV^T.
    """
    O_ns = newton_schulz5(G, steps=steps, apply_aspect_scale=apply_aspect_scale)
    O_svd = polar_factor_svd(G)
    denom = float(np.linalg.norm(O_svd, ord="fro")) + EPS
    return float(np.linalg.norm(O_ns - O_svd, ord="fro") / denom)
