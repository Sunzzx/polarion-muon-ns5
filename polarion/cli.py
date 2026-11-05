"""CLI entrypoint: polarion / polarion-muon-ns5."""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import __version__
from .benchmark import format_benchmark_table, run_mlp_benchmark
from .newton_schulz import (
    NS5_A,
    NS5_B,
    NS5_C,
    newton_schulz5,
    ns5_vs_svd_error,
    orthogonal_error,
    polar_factor_svd,
    singular_value_spread,
)
from .optimizer import MuonOptimizer
from .parameter import Parameter


def cmd_status(_: argparse.Namespace) -> int:
    print(f"polarion-muon-ns5 v{__version__}")
    print()
    print("Muon — Momentum Orthogonalized by Newton-Schulz (Jordan et al., 2024)")
    print()
    print("Quintic polynomial:")
    print(f"  p(x) = {NS5_A} x  +  {NS5_B} x^3  +  {NS5_C} x^5")
    print()
    print("Hybrid routing:")
    print("  2D weights (m,n ≥ 2)  →  Muon  (η≈0.02, μ≈0.95, NS steps=5)")
    print("  biases / norms / 1D   →  AdamW (η≈0.003, β=(0.9, 0.999))")
    print()
    print("Update:")
    print("  M_t = μ M_{t-1} + G_t")
    print("  O_t = NewtonSchulz5(μ M_t + G_t)   # Nesterov look-ahead")
    print("  W   ← (1 - ηλ) W - η O_t")
    print()
    print("Commands: status | orthogonalize | train-step | benchmark | verify | demo")
    return 0


def cmd_orthogonalize(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    G = rng.normal(size=(args.rows, args.cols))
    # Stretch spectrum
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    S = np.logspace(0, -args.condition_decades, S.size)
    G = (U * S) @ Vt

    t0 = time.perf_counter()
    O = newton_schulz5(G, steps=args.steps, apply_aspect_scale=not args.no_scale)
    dt = time.perf_counter() - t0

    O_ref = polar_factor_svd(G)
    # Compare without aspect scale for fair SVD match
    O_noscale = newton_schulz5(G, steps=args.steps, apply_aspect_scale=False)
    rel = float(
        np.linalg.norm(O_noscale - O_ref, ord="fro")
        / (np.linalg.norm(O_ref, ord="fro") + 1e-12)
    )
    smin, smax, kappa = singular_value_spread(O_noscale)
    print(json.dumps({
        "shape": [args.rows, args.cols],
        "steps": args.steps,
        "ns5_ms": round(dt * 1000, 4),
        "rel_error_vs_svd": rel,
        "orthogonal_error": orthogonal_error(O_noscale),
        "sigma_min": smin,
        "sigma_max": smax,
        "condition": kappa,
        "frobenius_norm_O": float(np.linalg.norm(O, ord="fro")),
    }, indent=2))
    return 0


def cmd_train_step(_: argparse.Namespace) -> int:
    rng = np.random.default_rng(0)
    params = [
        Parameter("W", rng.normal(scale=0.05, size=(32, 16))),
        Parameter("b", np.zeros(32)),
    ]
    opt = MuonOptimizer(params, lr=0.02, adamw_lr=0.003)
    for p in params:
        p.set_grad(rng.normal(scale=0.1, size=p.shape))
    before = {p.name: float(np.linalg.norm(p.data)) for p in params}
    opt.step()
    after = {p.name: float(np.linalg.norm(p.data)) for p in params}
    print(json.dumps({
        "routing": opt.routing_summary(),
        "norm_before": before,
        "norm_after": after,
        "global_step": opt._global_step,
    }, indent=2))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    results = run_mlp_benchmark(steps=args.steps, seed=args.seed)
    print(format_benchmark_table(results))
    winner = min(results.values(), key=lambda r: r.final_loss)
    print(f"\nwinner: {winner.name}  final_loss={winner.final_loss:.6e}")
    if args.json:
        print(json.dumps({k: v.as_dict() for k, v in results.items()}, indent=2))
    return 0


def cmd_verify(_: argparse.Namespace) -> int:
    """Self-hosted mathematical verification (mirrors pytest invariants)."""
    rng = np.random.default_rng(123)
    failures = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
        if not cond:
            failures += 1

    # 1. NS5 close to SVD polar on well-scaled matrices
    for shape in [(32, 32), (64, 32), (32, 64), (16, 48)]:
        G = rng.normal(size=shape)
        err = ns5_vs_svd_error(G, steps=5, apply_aspect_scale=False)
        check(f"NS5≈SVD polar {shape}", err < 0.55, f"rel_err={err:.4e}")

    # 2. Polar factor has singular values in Muon band ~[0.15, 2.5]
    G = rng.normal(size=(40, 40))
    O = newton_schulz5(G, steps=5, apply_aspect_scale=False)
    smin, smax, kappa = singular_value_spread(O)
    check("σ cluster in Muon band", smin > 0.05 and smax < 2.5, f"σ∈[{smin:.3f},{smax:.3f}]")
    check("κ(O) modest", kappa < 25.0, f"κ={kappa:.3f}")

    # 3. Exact SVD polar is a partial isometry
    O_svd = polar_factor_svd(G)
    check("SVD polar ortho", orthogonal_error(O_svd) < 1e-8, f"err={orthogonal_error(O_svd):.2e}")

    # 4. Optimizer reduces loss on toy problem
    from .benchmark import run_mlp_benchmark
    res = run_mlp_benchmark(steps=40, seed=0)
    check(
        "Muon loss decreases",
        res["muon"].final_loss < res["muon"].losses[0] * 0.9,
        f"{res['muon'].losses[0]:.3e} → {res['muon'].final_loss:.3e}",
    )
    check(
        "Muon competitive vs SGD",
        res["muon"].final_loss <= res["sgd"].final_loss * 1.25,
        f"muon={res['muon'].final_loss:.3e} sgd={res['sgd'].final_loss:.3e}",
    )

    # 5. State dict round-trip
    p = Parameter("W", rng.normal(size=(8, 8)))
    opt = MuonOptimizer([p], lr=0.01)
    p.set_grad(rng.normal(size=(8, 8)))
    opt.step()
    sd = opt.state_dict()
    p2 = Parameter("W", np.zeros((8, 8)))
    opt2 = MuonOptimizer([p2], lr=0.01)
    opt2.load_state_dict(sd)
    check("state_dict round-trip", np.allclose(p.data, p2.data))

    print()
    if failures:
        print(f"VERIFY FAILED: {failures} check(s)")
        return 1
    print("VERIFY OK: all checks passed")
    return 0


def cmd_demo(_: argparse.Namespace) -> int:
    from .demo import main as demo_main
    return demo_main([])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="polarion",
        description="Polarion Muon NS5 — spectral matrix optimizer via Newton-Schulz polar decomposition",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Architecture, formulas, hyperparameters").set_defaults(func=cmd_status)

    o = sub.add_parser("orthogonalize", help="Run NS5 polar factor on a random matrix")
    o.add_argument("--rows", type=int, default=64)
    o.add_argument("--cols", type=int, default=128)
    o.add_argument("--steps", type=int, default=5)
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--condition-decades", type=float, default=3.0)
    o.add_argument("--no-scale", action="store_true", help="Disable aspect-ratio scale")
    o.set_defaults(func=cmd_orthogonalize)

    sub.add_parser("train-step", help="One hybrid optimizer step on toy params").set_defaults(func=cmd_train_step)

    b = sub.add_parser("benchmark", help="Muon vs AdamW vs SGD MLP convergence")
    b.add_argument("--steps", type=int, default=60)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--json", action="store_true")
    b.set_defaults(func=cmd_benchmark)

    sub.add_parser("verify", help="Run mathematical self-checks").set_defaults(func=cmd_verify)
    sub.add_parser("demo", help="Full interactive terminal demo").set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
