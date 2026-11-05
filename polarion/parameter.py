"""Trainable parameter container with optional gradient buffer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Parameter:
    """Named tensor with mutable data and gradient slots.

    Muon routes 2D parameters (ndim == 2 and min(shape) >= 2) through
    Newton-Schulz orthogonalization. All other shapes use AdamW.
    """

    name: str
    data: np.ndarray
    grad: Optional[np.ndarray] = field(default=None, repr=False)
    requires_grad: bool = True

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.grad is not None:
            self.grad = np.asarray(self.grad, dtype=np.float64)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return int(self.data.size)

    def is_muon_matrix(self) -> bool:
        """True iff this tensor should be optimized with Muon (2D, both dims ≥ 2)."""
        if self.ndim != 2:
            return False
        m, n = self.shape
        return m >= 2 and n >= 2

    def zero_grad(self) -> None:
        self.grad = None

    def set_grad(self, g: np.ndarray) -> None:
        g = np.asarray(g, dtype=np.float64)
        if g.shape != self.data.shape:
            raise ValueError(
                f"grad shape {g.shape} != param shape {self.data.shape} ({self.name})"
            )
        self.grad = g

    def clone(self) -> "Parameter":
        return Parameter(
            name=self.name,
            data=self.data.copy(),
            grad=None if self.grad is None else self.grad.copy(),
            requires_grad=self.requires_grad,
        )
