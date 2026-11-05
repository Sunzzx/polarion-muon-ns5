"""Polarion: Muon spectral optimizer via 5th-order Newton-Schulz polar decomposition.

Implements Momentum Orthogonalized by Newton-Schulz (Muon; Jordan et al., 2024)
with hybrid AdamW routing for 1D parameters, aspect-ratio operator scaling,
optional Nesterov momentum, and numerical verification utilities.
"""

from .newton_schulz import (
    newton_schulz5,
    polar_factor_svd,
    frobenius_normalize,
    orthogonal_error,
    singular_value_spread,
)
from .parameter import Parameter
from .optimizer import MuonOptimizer, AdamWState
from .benchmark import run_mlp_benchmark, BenchmarkResult

__version__ = "0.1.0"
__all__ = [
    "newton_schulz5",
    "polar_factor_svd",
    "frobenius_normalize",
    "orthogonal_error",
    "singular_value_spread",
    "Parameter",
    "MuonOptimizer",
    "AdamWState",
    "run_mlp_benchmark",
    "BenchmarkResult",
    "__version__",
]
