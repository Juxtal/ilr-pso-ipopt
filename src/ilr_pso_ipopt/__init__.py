"""ILR-PSO + IPOPT: a generic optimizer for fixed-sum allocation problems in Pyomo."""

from .core import (
    OptimizationResult,
    build_helmert_basis,
    ilr_inverse,
    ilr_transform,
    optimize_allocation,
)

__all__ = [
    "OptimizationResult",
    "build_helmert_basis",
    "ilr_inverse",
    "ilr_transform",
    "optimize_allocation",
]

__version__ = "0.1.0"
