"""Levenberg-Marquardt nonlinear least squares implemented in Mojo."""

from ._lib import build
from .solver import (
    LM_DIFF_DELTA,
    LM_INIT_MU,
    LM_STOP_THRESH,
    Result,
    least_squares,
    levmar,
    solve,
)

__all__ = [
    "Result",
    "solve",
    "least_squares",
    "levmar",
    "build",
    "LM_INIT_MU",
    "LM_STOP_THRESH",
    "LM_DIFF_DELTA",
]

__version__ = "0.1.0"
