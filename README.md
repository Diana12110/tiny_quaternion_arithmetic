# tiny_quaternion_arithmetic

Hamilton product, conjugate, norm, and rotation extraction for real quaternions represented as 4-tuples of floats. Pure standard library, no dependencies.

## Usage

```python
from tiny_quaternion_arithmetic import hamilton, conjugate, norm, normalize, to_axis_angle

q1 = (1.0, 2.0, 3.0, 4.0)
q2 = (5.0, 6.0, 7.0, 8.0)

prod = hamilton(q1, q2)          # (-60.0, 12.0, 30.0, 24.0)
conj = conjugate(q1)             # (1.0, -2.0, -3.0, -4.0)
n   = norm(q1)                  # ~5.477
u   = normalize(q1)             # unit-length version of q1
axis, angle = to_axis_angle(u)  # ((x, y, z), radians in [0, 2*pi])
```

A quaternion is the tuple `(w, x, y, z)` — scalar first. The Hamilton product is non-commutative; `hamilton(a, b)` is not `hamilton(b, a)` in general.

## Why

Most quaternion libraries on PyPI bundle a vector class, rotation matrices, interpolation (slerp), and integration helpers. When the only need is the four core operations on plain tuples — say, for a tiny embedded script or a build tool that must stay dependency-free — that surface area is overhead and a maintenance risk. This library implements exactly those four operations on tuples, nothing more. The trade-off: you give up ergonomics (no methods on a `Quaternion` object, no pretty repr) for a module small enough to audit by eye and safe to vendor.

## Edge cases

- `normalize` returns a zero quaternion unchanged when the norm is below `1e-12`, rather than dividing by near-zero and producing `inf`/`nan`.
- `to_axis_angle` returns the default axis `(0.0, 0.0, 1.0)` and angle `0.0` for an identity quaternion, because the axis is mathematically undefined there. The caller should not read meaning into which axis is returned for a zero rotation.
- The rotation angle is always in `[0, 2*pi]` via `2 * acos(w)`. If you need a signed angle, negate the axis for angles past `pi`.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

