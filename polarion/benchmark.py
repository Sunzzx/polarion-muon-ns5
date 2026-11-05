"""Toy MLP regression benchmark: Muon vs AdamW vs SGD convergence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .optimizer import AdamWState, MuonOptimizer, sgd_step
from .parameter import Parameter


@dataclass
class BenchmarkResult:
    name: str
    losses: List[float]
    final_loss: float
    best_loss: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "losses": self.losses,
        }


def _make_mlp_params(
    d_in: int,
    d_hidden: int,
    d_out: int,
    rng: np.random.Generator,
    scale: float = 0.05,
) -> List[Parameter]:
    """2-layer MLP: W1 (d_hidden, d_in), b1, W2 (d_out, d_hidden), b2."""
    W1 = Parameter("W1", rng.normal(0, scale, size=(d_hidden, d_in)))
    b1 = Parameter("b1", np.zeros(d_hidden))
    W2 = Parameter("W2", rng.normal(0, scale, size=(d_out, d_hidden)))
    b2 = Parameter("b2", np.zeros(d_out))
    return [W1, b1, W2, b2]


def _forward(params: Dict[str, Parameter], X: np.ndarray) -> np.ndarray:
    """ReLU MLP forward. X: (batch, d_in) → (batch, d_out)."""
    h = X @ params["W1"].data.T + params["b1"].data
    h = np.maximum(h, 0.0)
    y = h @ params["W2"].data.T + params["b2"].data
    return y


def _backward(
    params: Dict[str, Parameter],
    X: np.ndarray,
    Y: np.ndarray,
    Y_hat: np.ndarray,
) -> float:
    """MSE loss + grads. Returns scalar loss."""
    batch = X.shape[0]
    residual = (Y_hat - Y) / batch  # dL/dY_hat
    loss = float(0.5 * np.mean((Y_hat - Y) ** 2))

    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    h_pre = X @ W1.data.T + b1.data
    h = np.maximum(h_pre, 0.0)

    # Output layer
    W2.set_grad(residual.T @ h)
    b2.set_grad(residual.sum(axis=0))

    # Hidden
    dh = residual @ W2.data
    dh_pre = dh * (h_pre > 0.0)
    W1.set_grad(dh_pre.T @ X)
    b1.set_grad(dh_pre.sum(axis=0))
    return loss


def _clone_params(params: List[Parameter]) -> List[Parameter]:
    return [p.clone() for p in params]


def _adamw_only_step(
    params: List[Parameter],
    state: Dict[str, AdamWState],
    lr: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> None:
    b1, b2 = betas
    for p in params:
        if p.grad is None:
            continue
        if p.name not in state:
            state[p.name] = AdamWState(
                m=np.zeros_like(p.data),
                v=np.zeros_like(p.data),
                step=0,
            )
        st = state[p.name]
        st.step += 1
        st.m = b1 * st.m + (1.0 - b1) * p.grad
        st.v = b2 * st.v + (1.0 - b2) * (p.grad * p.grad)
        m_hat = st.m / (1.0 - b1 ** st.step)
        v_hat = st.v / (1.0 - b2 ** st.step)
        if weight_decay != 0.0:
            p.data *= 1.0 - lr * weight_decay
        p.data -= lr * m_hat / (np.sqrt(v_hat) + eps)


def run_mlp_benchmark(
    steps: int = 60,
    batch: int = 64,
    d_in: int = 16,
    d_hidden: int = 32,
    d_out: int = 4,
    seed: int = 42,
    muon_lr: float = 0.05,
    adamw_lr: float = 0.01,
    sgd_lr: float = 0.05,
) -> Dict[str, BenchmarkResult]:
    """Compare Muon (hybrid), pure AdamW, and SGD on a fixed synthetic regression.

    Target: y = tanh(X @ W_true.T) with teacher matrix. Same init / data for all.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1.0, size=(batch, d_in))
    W_true = rng.normal(0, 0.4, size=(d_out, d_in))
    Y = np.tanh(X @ W_true.T)

    base = _make_mlp_params(d_in, d_hidden, d_out, rng)

    results: Dict[str, BenchmarkResult] = {}

    # ---- Muon hybrid ----
    params_m = _clone_params(base)
    opt = MuonOptimizer(
        parameters=params_m,
        lr=muon_lr,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.0,
        ns_steps=5,
        adamw_lr=adamw_lr,
    )
    losses_m: List[float] = []
    pmap_m = {p.name: p for p in params_m}
    for _ in range(steps):
        yhat = _forward(pmap_m, X)
        loss = _backward(pmap_m, X, Y, yhat)
        losses_m.append(loss)
        opt.step()
        opt.zero_grad()
    results["muon"] = BenchmarkResult(
        name="muon",
        losses=losses_m,
        final_loss=losses_m[-1],
        best_loss=min(losses_m),
    )

    # ---- AdamW ----
    params_a = _clone_params(base)
    aw_state: Dict[str, AdamWState] = {}
    losses_a: List[float] = []
    pmap_a = {p.name: p for p in params_a}
    for _ in range(steps):
        yhat = _forward(pmap_a, X)
        loss = _backward(pmap_a, X, Y, yhat)
        losses_a.append(loss)
        _adamw_only_step(params_a, aw_state, lr=adamw_lr, weight_decay=0.0)
        for p in params_a:
            p.zero_grad()
    results["adamw"] = BenchmarkResult(
        name="adamw",
        losses=losses_a,
        final_loss=losses_a[-1],
        best_loss=min(losses_a),
    )

    # ---- SGD ----
    params_s = _clone_params(base)
    losses_s: List[float] = []
    pmap_s = {p.name: p for p in params_s}
    for _ in range(steps):
        yhat = _forward(pmap_s, X)
        loss = _backward(pmap_s, X, Y, yhat)
        losses_s.append(loss)
        sgd_step(params_s, lr=sgd_lr, weight_decay=0.0)
        for p in params_s:
            p.zero_grad()
    results["sgd"] = BenchmarkResult(
        name="sgd",
        losses=losses_s,
        final_loss=losses_s[-1],
        best_loss=min(losses_s),
    )

    return results


def format_benchmark_table(results: Dict[str, BenchmarkResult]) -> str:
    lines = [
        f"{'optimizer':<10} {'final_loss':>14} {'best_loss':>14} {'start_loss':>14}",
        "-" * 56,
    ]
    for name in ("muon", "adamw", "sgd"):
        r = results[name]
        lines.append(
            f"{r.name:<10} {r.final_loss:14.6e} {r.best_loss:14.6e} {r.losses[0]:14.6e}"
        )
    return "\n".join(lines)
