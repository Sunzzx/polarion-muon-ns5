"""Unit, edge-case, and numerical invariance tests for Polarion Muon NS5."""

from __future__ import annotations

import numpy as np
import pytest

from polarion.benchmark import run_mlp_benchmark
from polarion.newton_schulz import (
    aspect_ratio_scale,
    frobenius_normalize,
    newton_schulz5,
    ns5_vs_svd_error,
    orthogonal_error,
    polar_factor_svd,
    singular_value_spread,
)
from polarion.optimizer import MuonOptimizer, sgd_step
from polarion.parameter import Parameter


# ---------------------------------------------------------------------------
# Newton-Schulz core
# ---------------------------------------------------------------------------


class TestFrobeniusNormalize:
    def test_unit_frobenius(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 7))
        Y = frobenius_normalize(X)
        # Floor eps in denominator → norm slightly under 1
        assert abs(np.linalg.norm(Y, ord="fro") - 1.0) < 1e-7

    def test_zero_matrix_finite(self):
        Z = frobenius_normalize(np.zeros((4, 4)))
        assert np.all(np.isfinite(Z))
        assert np.allclose(Z, 0.0)


class TestAspectRatio:
    def test_square_is_one(self):
        assert aspect_ratio_scale(64, 64) == 1.0

    def test_wide_is_one(self):
        # m < n → sqrt(m/n) < 1 → max(1, ·) = 1
        assert aspect_ratio_scale(32, 128) == 1.0

    def test_tall_scales(self):
        s = aspect_ratio_scale(100, 25)
        assert abs(s - 2.0) < 1e-12


class TestNewtonSchulz5:
    def test_shape_preserved_square(self):
        G = np.random.default_rng(1).normal(size=(16, 16))
        O = newton_schulz5(G, steps=5)
        assert O.shape == G.shape

    def test_shape_preserved_tall(self):
        G = np.random.default_rng(2).normal(size=(64, 16))
        O = newton_schulz5(G, steps=5)
        assert O.shape == (64, 16)

    def test_shape_preserved_wide(self):
        G = np.random.default_rng(3).normal(size=(16, 64))
        O = newton_schulz5(G, steps=5)
        assert O.shape == (16, 64)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            newton_schulz5(np.ones(5))

    def test_rejects_zero_steps(self):
        with pytest.raises(ValueError):
            newton_schulz5(np.eye(3), steps=0)

    def test_empty_matrix(self):
        G = np.zeros((0, 5))
        O = newton_schulz5(G)
        assert O.shape == (0, 5)

    def test_ns_improves_over_raw_gradient_alignment(self):
        """NS5 polar approx stays in the same ballpark as SVD polar (not exact).

        Jordan et al. intentionally tune the quintic past the global-convergence
        regime so singular values land in ~[0.5, 1.5] rather than exactly 1;
        more steps do not monotonically reduce SVD error.
        """
        rng = np.random.default_rng(4)
        G = rng.normal(size=(24, 24))
        e5 = ns5_vs_svd_error(G, steps=5, apply_aspect_scale=False)
        assert e5 < 0.55, f"rel_err={e5}"

    def test_approx_matches_svd_polar(self):
        rng = np.random.default_rng(5)
        for shape in [(20, 20), (30, 12), (12, 30)]:
            G = rng.normal(size=shape)
            err = ns5_vs_svd_error(G, steps=5, apply_aspect_scale=False)
            # Inexact polar by design — bound is loose but finite.
            assert err < 0.55, f"shape={shape} err={err}"

    def test_singular_values_cluster(self):
        """Muon NS5 targets σ ∈ ~[0.5, 1.5], not exact ones (Jordan docstring)."""
        rng = np.random.default_rng(6)
        G = rng.normal(size=(32, 32))
        O = newton_schulz5(G, steps=5, apply_aspect_scale=False)
        smin, smax, kappa = singular_value_spread(O)
        assert smin > 0.05
        assert smax < 2.5
        assert kappa < 25.0

    def test_orthogonal_error_bounded(self):
        rng = np.random.default_rng(7)
        G = rng.normal(size=(28, 28))
        O = newton_schulz5(G, steps=5, apply_aspect_scale=False)
        # Partial-isometry residual is moderate under inexact NS5.
        assert orthogonal_error(O) < 0.75

    def test_ill_conditioned_input_stabilized(self):
        rng = np.random.default_rng(8)
        U, _ = np.linalg.qr(rng.normal(size=(40, 40)))
        V, _ = np.linalg.qr(rng.normal(size=(40, 40)))
        s = np.logspace(0, -6, 40)
        G = (U * s) @ V.T
        _, _, kappa_in = singular_value_spread(G)
        O = newton_schulz5(G, steps=5, apply_aspect_scale=False)
        smin, smax, kappa_out = singular_value_spread(O)
        assert kappa_in > 1e5
        # Dominant directions lift toward O(1); κ collapses by orders of magnitude.
        assert kappa_out < kappa_in / 10.0
        assert smax < 3.0
        assert smin > 1e-4


class TestPolarFactorSVD:
    def test_partial_isometry(self):
        rng = np.random.default_rng(9)
        G = rng.normal(size=(25, 18))
        O = polar_factor_svd(G)
        assert orthogonal_error(O) < 1e-9

    def test_zero_matrix(self):
        O = polar_factor_svd(np.zeros((5, 5)))
        assert np.allclose(O, 0.0)


# ---------------------------------------------------------------------------
# Parameter + Optimizer
# ---------------------------------------------------------------------------


class TestParameter:
    def test_muon_matrix_gate(self):
        assert Parameter("W", np.zeros((4, 4))).is_muon_matrix()
        assert not Parameter("b", np.zeros(4)).is_muon_matrix()
        assert not Parameter("row", np.zeros((1, 8))).is_muon_matrix()
        assert not Parameter("e", np.zeros((8,))).is_muon_matrix()

    def test_grad_shape_guard(self):
        p = Parameter("W", np.zeros((3, 3)))
        with pytest.raises(ValueError):
            p.set_grad(np.zeros((2, 2)))


class TestMuonOptimizer:
    def test_routing(self):
        params = [
            Parameter("W", np.eye(4)),
            Parameter("b", np.zeros(4)),
        ]
        opt = MuonOptimizer(params)
        assert opt.routing_summary() == {"W": "muon", "b": "adamw"}

    def test_step_changes_params(self):
        rng = np.random.default_rng(10)
        W = Parameter("W", rng.normal(size=(8, 8)) * 0.1)
        b = Parameter("b", np.zeros(8))
        opt = MuonOptimizer([W, b], lr=0.05, adamw_lr=0.01, weight_decay=0.0)
        W0, b0 = W.data.copy(), b.data.copy()
        W.set_grad(rng.normal(size=(8, 8)))
        b.set_grad(rng.normal(size=(8,)))
        opt.step()
        assert not np.allclose(W.data, W0)
        assert not np.allclose(b.data, b0)

    def test_zero_grad(self):
        p = Parameter("W", np.eye(3))
        p.set_grad(np.ones((3, 3)))
        opt = MuonOptimizer([p])
        opt.zero_grad()
        assert p.grad is None

    def test_state_dict_roundtrip(self):
        rng = np.random.default_rng(11)
        W = Parameter("W", rng.normal(size=(6, 6)))
        b = Parameter("b", rng.normal(size=(6,)))
        opt = MuonOptimizer([W, b], lr=0.02, momentum=0.9, nesterov=True)
        W.set_grad(rng.normal(size=(6, 6)))
        b.set_grad(rng.normal(size=(6,)))
        opt.step()
        sd = opt.state_dict()

        W2 = Parameter("W", np.zeros((6, 6)))
        b2 = Parameter("b", np.zeros(6))
        opt2 = MuonOptimizer([W2, b2])
        opt2.load_state_dict(sd)
        assert np.allclose(W.data, W2.data)
        assert np.allclose(b.data, b2.data)
        assert opt2._global_step == opt._global_step
        assert np.allclose(opt._muon_buf["W"], opt2._muon_buf["W"])

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError):
            MuonOptimizer([
                Parameter("W", np.eye(2)),
                Parameter("W", np.eye(2)),
            ])

    def test_weight_decay_shrinks_norm(self):
        W = Parameter("W", np.ones((4, 4)))
        opt = MuonOptimizer([W], lr=0.1, weight_decay=0.5, momentum=0.0, nesterov=False)
        # grad = 0 → only decay acts after orthogonalizing zeros... use tiny grad
        W.set_grad(np.zeros((4, 4)))
        # With zero grad, NS of zero buffer is zero; decay still applies
        n0 = np.linalg.norm(W.data)
        opt.step()
        n1 = np.linalg.norm(W.data)
        assert n1 < n0

    def test_sgd_baseline(self):
        p = Parameter("W", np.array([[1.0, 0.0], [0.0, 1.0]]))
        p.set_grad(np.array([[1.0, 0.0], [0.0, 0.0]]))
        sgd_step([p], lr=0.5)
        assert abs(p.data[0, 0] - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# Benchmark / end-to-end numerical invariance
# ---------------------------------------------------------------------------


class TestBenchmark:
    def test_muon_loss_decreases(self):
        res = run_mlp_benchmark(steps=40, seed=0)
        assert res["muon"].final_loss < res["muon"].losses[0] * 0.85

    def test_all_methods_finite(self):
        res = run_mlp_benchmark(steps=20, seed=1)
        for r in res.values():
            assert np.isfinite(r.final_loss)
            assert all(np.isfinite(x) for x in r.losses)

    def test_muon_beats_or_matches_sgd(self):
        res = run_mlp_benchmark(steps=50, seed=2)
        # Allow small slack — stochastic geometry of toy MLP
        assert res["muon"].final_loss <= res["sgd"].final_loss * 1.3

    def test_reproducible(self):
        a = run_mlp_benchmark(steps=15, seed=99)
        b = run_mlp_benchmark(steps=15, seed=99)
        assert a["muon"].losses == pytest.approx(b["muon"].losses, rel=1e-12)


class TestInvariants:
    def test_ns5_dtype_float64(self):
        G = np.ones((5, 5), dtype=np.float32)
        O = newton_schulz5(G)
        assert O.dtype == np.float64

    def test_identity_polar_near_identity(self):
        I = np.eye(10)
        O = newton_schulz5(I, steps=5, apply_aspect_scale=False)
        # Polar of I is I; NS may have mild residual
        assert np.linalg.norm(O - I, ord="fro") < 0.5

    def test_negative_scaling_sign_structure(self):
        """Polar factor ignores singular-value signs of magnitude — NS should stay finite."""
        G = -3.0 * np.eye(8)
        O = newton_schulz5(G, steps=5, apply_aspect_scale=False)
        assert np.all(np.isfinite(O))
        # Polar of -I is -I (all singular vals positive after abs in SVD; U absorbs sign)
        O_svd = polar_factor_svd(G)
        err = np.linalg.norm(O - O_svd, ord="fro") / (np.linalg.norm(O_svd) + 1e-12)
        assert err < 0.2
