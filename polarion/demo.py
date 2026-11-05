"""Interactive terminal demo for Polarion Muon NS5."""

from __future__ import annotations

import sys
import time

import numpy as np

from .benchmark import format_benchmark_table, run_mlp_benchmark
from .newton_schulz import (
    newton_schulz5,
    ns5_vs_svd_error,
    orthogonal_error,
    polar_factor_svd,
    singular_value_spread,
)
from .optimizer import MuonOptimizer
from .parameter import Parameter


def _bar(value: float, width: int = 28, lo: float = 0.0, hi: float = 1.0) -> str:
    if hi <= lo:
        return " " * width
    t = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    n = int(round(t * width))
    return "█" * n + "·" * (width - n)


def _header(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def demo_orthogonalization(rows: int = 64, cols: int = 128, steps: int = 5) -> None:
    _header(f"Newton-Schulz Polar Factor  ({rows}×{cols}, steps={steps})")
    rng = np.random.default_rng(0)
    # Ill-conditioned momentum: log-spaced singular values
    U, _ = np.linalg.qr(rng.normal(size=(rows, min(rows, cols))))
    V, _ = np.linalg.qr(rng.normal(size=(cols, min(rows, cols))))
    s = np.logspace(0, -4, min(rows, cols))
    G = (U * s) @ V.T

    t0 = time.perf_counter()
    O_ns = newton_schulz5(G, steps=steps, apply_aspect_scale=False)
    t_ns = time.perf_counter() - t0

    t0 = time.perf_counter()
    O_svd = polar_factor_svd(G)
    t_svd = time.perf_counter() - t0

    err = float(np.linalg.norm(O_ns - O_svd, ord="fro") / (np.linalg.norm(O_svd, ord="fro") + 1e-12))
    smin_g, smax_g, kappa_g = singular_value_spread(G)
    smin_o, smax_o, kappa_o = singular_value_spread(O_ns)
    ortho = orthogonal_error(O_ns)

    print(f"  Input κ(G)           : {kappa_g:.4e}   σ∈[{smin_g:.3e}, {smax_g:.3e}]")
    print(f"  Output κ(O_ns)       : {kappa_o:.4e}   σ∈[{smin_o:.3e}, {smax_o:.3e}]")
    print(f"  Orthogonality error  : {ortho:.4e}")
    print(f"  NS5 vs SVD rel-error : {err:.4e}")
    print(f"  Timing NS5 / SVD     : {t_ns*1e3:.2f} ms  /  {t_svd*1e3:.2f} ms")
    print(f"  Speedup              : {t_svd / max(t_ns, 1e-12):.1f}×")

    # Convergence of NS error vs steps
    print()
    print("  steps | rel-err to SVD polar | ortho-err | bar")
    print("  " + "-" * 58)
    for k in range(1, 9):
        e = ns5_vs_svd_error(G, steps=k, apply_aspect_scale=False)
        o = orthogonal_error(newton_schulz5(G, steps=k, apply_aspect_scale=False))
        print(f"    {k:2d}  | {e:18.4e} | {o:9.2e} | {_bar(np.log10(e + 1e-16), hi=0, lo=-8)}")


def demo_routing() -> None:
    _header("Hybrid Routing: Muon matrices vs AdamW vectors")
    rng = np.random.default_rng(1)
    params = [
        Parameter("attn_q", rng.normal(scale=0.02, size=(64, 64))),
        Parameter("attn_k", rng.normal(scale=0.02, size=(64, 64))),
        Parameter("mlp_up", rng.normal(scale=0.02, size=(128, 64))),
        Parameter("bias", np.zeros(64)),
        Parameter("ln_scale", np.ones(64)),
        Parameter("embed", rng.normal(scale=0.02, size=(256,))),  # 1D vocab slice
    ]
    opt = MuonOptimizer(parameters=params, lr=0.02, adamw_lr=0.003)
    summary = opt.routing_summary()
    print(f"  {'parameter':<12} {'shape':<14} {'route':<8}")
    print("  " + "-" * 36)
    for p in params:
        print(f"  {p.name:<12} {str(p.shape):<14} {summary[p.name]:<8}")

    # One fake step
    for p in params:
        p.set_grad(rng.normal(scale=0.01, size=p.shape))
    before = {p.name: p.data.copy() for p in params}
    opt.step()
    print()
    print("  After 1 step — mean |Δ|:")
    for p in params:
        delta = float(np.mean(np.abs(p.data - before[p.name])))
        print(f"    {p.name:<12}  Δ={delta:.6e}  via {summary[p.name]}")


def demo_benchmark(steps: int = 50) -> None:
    _header(f"MLP Convergence Benchmark  (steps={steps})")
    results = run_mlp_benchmark(steps=steps, seed=7)
    print(format_benchmark_table(results))
    print()
    # ASCII loss curves (downsample)
    max_loss = max(r.losses[0] for r in results.values())
    print("  Loss trajectories (normalized to start):")
    colors = {"muon": "M", "adamw": "A", "sgd": "S"}
    width = 40
    for name, r in results.items():
        pts = [r.losses[i] for i in np.linspace(0, len(r.losses) - 1, width, dtype=int)]
        chars = []
        for v in pts:
            h = int(round(8 * (1.0 - min(1.0, v / (max_loss + 1e-12)))))
            chars.append(" ▁▂▃▄▅▆▇█"[h])
        print(f"  {colors[name]} {name:<6} {''.join(chars)}  → {r.final_loss:.3e}")

    # Declare winner by final loss
    winner = min(results.values(), key=lambda r: r.final_loss)
    print()
    print(f"  Winner by final loss: {winner.name.upper()} ({winner.final_loss:.6e})")


def demo_train_step() -> None:
    _header("Single Hybrid Train Step on 2-layer MLP")
    rng = np.random.default_rng(3)
    W1 = Parameter("W1", rng.normal(scale=0.05, size=(32, 16)))
    b1 = Parameter("b1", np.zeros(32))
    W2 = Parameter("W2", rng.normal(scale=0.05, size=(4, 32)))
    b2 = Parameter("b2", np.zeros(4))
    opt = MuonOptimizer([W1, b1, W2, b2], lr=0.05, adamw_lr=0.01, weight_decay=0.0)

    X = rng.normal(size=(24, 16))
    Y = rng.normal(size=(24, 4))
    h = np.maximum(X @ W1.data.T + b1.data, 0.0)
    Yhat = h @ W2.data.T + b2.data
    residual = (Yhat - Y) / 24.0
    loss = float(0.5 * np.mean((Yhat - Y) ** 2))
    W2.set_grad(residual.T @ h)
    b2.set_grad(residual.sum(0))
    dh = residual @ W2.data
    dh_pre = dh * ((X @ W1.data.T + b1.data) > 0)
    W1.set_grad(dh_pre.T @ X)
    b1.set_grad(dh_pre.sum(0))

    print(f"  pre-step MSE loss : {loss:.6e}")
    print(f"  routing           : {opt.routing_summary()}")
    opt.step()
    h2 = np.maximum(X @ W1.data.T + b1.data, 0.0)
    Yhat2 = h2 @ W2.data.T + b2.data
    loss2 = float(0.5 * np.mean((Yhat2 - Y) ** 2))
    print(f"  post-step MSE loss: {loss2:.6e}")
    print(f"  Δ loss            : {loss2 - loss:+.6e}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    print("Polarion · Muon NS5 Spectral Optimizer")
    print("Momentum Orthogonalized by 5th-order Newton-Schulz")
    demo_orthogonalization()
    demo_routing()
    demo_train_step()
    demo_benchmark(steps=50)
    _header("Done")
    print("  python -m polarion.demo")
    print("  polarion benchmark | orthogonalize | verify | status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
