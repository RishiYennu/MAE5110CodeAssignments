"""Small control algorithms for gridded dynamical systems."""

from .discretization import build_transition_matrix
from .value_iteration import value_iteration

__all__ = ["build_transition_matrix", "value_iteration"]
