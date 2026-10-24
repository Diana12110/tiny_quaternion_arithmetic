from __future__ import annotations

import math
from typing import Tuple

# A quaternion is (w, x, y, z): scalar first. This ordering is used throughout
# because it keeps the rotation formula q = (cos(theta/2), sin(theta/2)*axis)
# readable and matches the convention in Hamilton's original formulation.
Quaternion = Tuple[float, float, float, float]


def hamilton(a: Quaternion, b: Quaternion) -> Quaternion:
    """Return the Hamilton product ``a * b``.

    Non-commutative in general; ``hamilton(a, b) != hamilton(b, a)`` except for
    special cases. Inputs are 4-tuples of real numbers in (w, x, y, z) order.
    """
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def conjugate(q: Quaternion) -> Quaternion:
    """Return the conjugate of ``q`` (negate the vector part)."""
    w, x, y, z = q
    return (w, -x, -y, -z)


def norm(q: Quaternion) -> float:
    """Return the Euclidean norm sqrt(w^2 + x^2 + y^2 + z^2)."""
    w, x, y, z = q
    return math.sqrt(w * w + x * x + y * y + z * z)


def normalize(q: Quaternion, eps: float = 1e-12) -> Quaternion:
    """Return ``q`` scaled to unit length.

    Quaternions with norm below ``eps`` are treated as zero and returned
    unchanged: dividing by a near-zero norm would produce inf/nan and is
    never the caller's intent. "eps`` is a scale relative to the input, not
    an absolute tolerance on the result.
    """
    n = norm(q)
    if n < eps:
        return q
    w, x, y, z = q
    return (w / n, x / n, y / n, z / n)


def to_axis_angle(q: Quaternion, eps: float = 1e-12) -> Tuple[Tuple[float, float, float], float]:
    """Extract a (unit_axis, angle) pair from ``q``.

    The quaternion is normalized first, so the caller does not need to.

    Returns ``((0.0, 0.0, 1.0), 0.0)`` when the rotation is identity, i.e.
    when the normalized vector part is shorter than ``eps`` or the scalar part
    is within ``eps`` of +/-1. The default axis ``(0, 0, 1)`` is an arbitrary
    but valid choice: a zero rotation about any axis is still a zero rotation.

    The returned angle is in radians, in the range ``[0, 2*pi]``. Using
    ``2 * acos(w_norm)`` (rather than ``atan2``) gives the shortest positive
    angle for the given quaternion; if you want the signed angle, negate the
    axis for angles past pi.
    """
    qn = normalize(q)
    w, x, y, z = qn

    # Guard against acos domain errors from floating-point drift.
    w = -1.0 if w < -1.0 else (1.0 if w > 1.0 else w)

    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))

    if s < eps:
        # Vector part vanished: no rotation. The axis is undefined; pick a
        # canonical one so callers always get a valid unit vector.
        return (0.0, 0.0, 1.0), 0.0

    return (x / s, y / s, z / s), angle
