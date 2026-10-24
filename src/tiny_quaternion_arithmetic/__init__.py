"""tiny_quaternion_arithmetic: Hamilton product, conjugate, norm, and rotation
extraction for real quaternions represented as 4-tuples of floats.

All public functions live in :mod:`tiny_quaternion_arithmetic.core` and are
re-exported here for convenience.
"""

from tiny_quaternion_arithmetic.core import (
    hamilton,
    conjugate,
    norm,
    normalize,
    to_axis_angle,
)

__all__ = [
    "hamilton",
    "conjugate",
    "norm",
    "normalize",
    "to_axis_angle",
]
