"""Hybrid Muon + AdamW optimizer.

Muon path (2D hidden weights)
-----------------------------
    M_t = μ M_{t-1} + G_t                  # (optional Nesterov look-ahead)
    O_t = NewtonSchulz5(M_t)               # polar factor ≈ U V^T
    W   ← W - η (O_t + λ W)                # decoupled weight decay

AdamW path (biases, norms, embeddings, vectors)
-----------------------------------------------
    Standard AdamW with decoupled weight decay (Loshchilov & Hutter).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np

from .newton_schulz import newton_schulz5
from .parameter import Parameter


@dataclass
class AdamWState:
    """First/second moment buffers for a single AdamW-routed parameter."""

    m: np.ndarray
    v: np.ndarray
    step: int = 0


@dataclass
class MuonOptimizer:
    """Hybrid spectral Muon optimizer with automatic AdamW fallback.

    Parameters
    ----------
    parameters :
        Iterable of Parameter objects.
    lr :
        Muon learning rate for matrix parameters (typical 0.02).
    momentum :
        Nesterov/heavy-ball momentum coefficient μ (typical 0.95).
    nesterov :
        If True, orthogonalize μ M + G rather than M after the momentum update.
    weight_decay :
        Decoupled L2 coefficient λ applied on both Muon and AdamW paths.
    ns_steps :
        Newton-Schulz iterations (default 5).
    adamw_lr :
        Learning rate for non-matrix parameters (typical 0.003).
    betas :
        AdamW (β1, β2).
    eps :
        AdamW numerical floor.
    """

    parameters: List[Parameter]
    lr: float = 0.02
    momentum: float = 0.95
    nesterov: bool = True
    weight_decay: float = 0.01
    ns_steps: int = 5
    adamw_lr: float = 0.003
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    _muon_buf: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _adamw: Dict[str, AdamWState] = field(default_factory=dict, init=False, repr=False)
    _global_step: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.parameters = list(self.parameters)
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("Parameter names must be unique")
        for p in self.parameters:
            if p.is_muon_matrix():
                self._muon_buf[p.name] = np.zeros_like(p.data)
            else:
                self._adamw[p.name] = AdamWState(
                    m=np.zeros_like(p.data),
                    v=np.zeros_like(p.data),
                    step=0,
                )

    def zero_grad(self) -> None:
        for p in self.parameters:
            p.zero_grad()

    def step(self) -> None:
        """Apply one optimizer step to all parameters that have gradients."""
        self._global_step += 1
        for p in self.parameters:
            if p.grad is None or not p.requires_grad:
                continue
            if p.is_muon_matrix():
                self._step_muon(p)
            else:
                self._step_adamw(p)

    def _step_muon(self, p: Parameter) -> None:
        g = p.grad
        assert g is not None
        buf = self._muon_buf[p.name]
        # Heavy-ball momentum accumulation
        buf *= self.momentum
        buf += g

        if self.nesterov:
            # Look-ahead momentum used only for the polar factor
            update_src = self.momentum * buf + g
        else:
            update_src = buf

        O = newton_schulz5(update_src, steps=self.ns_steps, apply_aspect_scale=True)

        # Decoupled weight decay + orthogonalized update
        if self.weight_decay != 0.0:
            p.data *= 1.0 - self.lr * self.weight_decay
        p.data -= self.lr * O

    def _step_adamw(self, p: Parameter) -> None:
        g = p.grad
        assert g is not None
        st = self._adamw[p.name]
        st.step += 1
        b1, b2 = self.betas

        st.m = b1 * st.m + (1.0 - b1) * g
        st.v = b2 * st.v + (1.0 - b2) * (g * g)

        # Bias correction
        m_hat = st.m / (1.0 - b1 ** st.step)
        v_hat = st.v / (1.0 - b2 ** st.step)

        if self.weight_decay != 0.0:
            p.data *= 1.0 - self.adamw_lr * self.weight_decay
        p.data -= self.adamw_lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state_dict(self) -> dict:
        return {
            "global_step": self._global_step,
            "lr": self.lr,
            "momentum": self.momentum,
            "nesterov": self.nesterov,
            "weight_decay": self.weight_decay,
            "ns_steps": self.ns_steps,
            "adamw_lr": self.adamw_lr,
            "betas": self.betas,
            "eps": self.eps,
            "muon_buf": {k: v.copy() for k, v in self._muon_buf.items()},
            "adamw": {
                k: {"m": s.m.copy(), "v": s.v.copy(), "step": s.step}
                for k, s in self._adamw.items()
            },
            "params": {p.name: p.data.copy() for p in self.parameters},
        }

    def load_state_dict(self, state: dict) -> None:
        self._global_step = int(state["global_step"])
        self.lr = float(state["lr"])
        self.momentum = float(state["momentum"])
        self.nesterov = bool(state["nesterov"])
        self.weight_decay = float(state["weight_decay"])
        self.ns_steps = int(state["ns_steps"])
        self.adamw_lr = float(state["adamw_lr"])
        self.betas = tuple(state["betas"])
        self.eps = float(state["eps"])
        self._muon_buf = {k: np.asarray(v, dtype=np.float64).copy() for k, v in state["muon_buf"].items()}
        self._adamw = {
            k: AdamWState(
                m=np.asarray(v["m"], dtype=np.float64).copy(),
                v=np.asarray(v["v"], dtype=np.float64).copy(),
                step=int(v["step"]),
            )
            for k, v in state["adamw"].items()
        }
        pdata = state.get("params", {})
        for p in self.parameters:
            if p.name in pdata:
                p.data = np.asarray(pdata[p.name], dtype=np.float64).copy()

    def routing_summary(self) -> Dict[str, str]:
        """Map parameter name → 'muon' | 'adamw'."""
        out: Dict[str, str] = {}
        for p in self.parameters:
            out[p.name] = "muon" if p.is_muon_matrix() else "adamw"
        return out


def sgd_step(params: Iterable[Parameter], lr: float = 0.01, weight_decay: float = 0.0) -> None:
    """Plain SGD baseline for benchmarks."""
    for p in params:
        if p.grad is None:
            continue
        if weight_decay != 0.0:
            p.data *= 1.0 - lr * weight_decay
        p.data -= lr * p.grad
