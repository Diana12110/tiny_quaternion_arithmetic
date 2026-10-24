import math
import unittest

from tiny_quaternion_arithmetic import (
    hamilton,
    conjugate,
    norm,
    normalize,
    to_axis_angle,
)


class TestHamilton(unittest.TestCase):
    def test_basis_multiplication(self):
        i = (0.0, 1.0, 0.0, 0.0)
        j = (0.0, 0.0, 1.0, 0.0)
        k = (0.0, 0.0, 0.0, 1.0)
        self.assertEqual(hamilton(i, i), (-1.0, 0.0, 0.0, 0.0))
        self.assertEqual(hamilton(j, j), (-1.0, 0.0, 0.0, 0.0))
        self.assertEqual(hamilton(k, k), (-1.0, 0.0, 0.0, 0.0))
        self.assertEqual(hamilton(i, j), k)
        self.assertEqual(hamilton(j, k), i)
        self.assertEqual(hamilton(k, i), j)
        self.assertEqual(hamilton(j, i), (-0.0, 0.0, 0.0, -1.0))

    def test_identity(self):
        ident = (1.0, 0.0, 0.0, 0.0)
        q = (2.0, -3.0, 4.5, -0.5)
        self.assertEqual(hamilton(ident, q), q)
        self.assertEqual(hamilton(q, ident), q)

    def test_non_commutative(self):
        a = (1.0, 2.0, 3.0, 4.0)
        b = (5.0, 6.0, 7.0, 8.0)
        self.assertNotEqual(hamilton(a, b), hamilton(b, a))

    def test_known_product(self):
        a = (1.0, 2.0, 3.0, 4.0)
        b = (5.0, 6.0, 7.0, 8.0)
        # w = 1*5 - 2*6 - 3*7 - 4*8 = -60
        # x = 1*6 + 2*5 + 3*8 - 4*7 = 12
        # y = 1*7 - 2*8 + 3*5 + 4*6 = 30
        # z = 1*8 + 2*7 - 3*6 + 4*5 = 24
        self.assertEqual(hamilton(a, b), (-60.0, 12.0, 30.0, 24.0))


class TestConjugate(unittest.TestCase):
    def test_negates_vector_part(self):
        self.assertEqual(conjugate((1.0, 2.0, 3.0, 4.0)), (1.0, -2.0, -3.0, -4.0))

    def test_pure_scalar_unchanged(self):
        self.assertEqual(conjugate((5.0, 0.0, 0.0, 0.0)), (5.0, 0.0, 0.0, 0.0))


class TestNorm(unittest.TestCase):
    def test_known(self):
        self.assertAlmostEqual(norm((1.0, 2.0, 2.0, 0.0)), 3.0)

    def test_zero(self):
        self.assertEqual(norm((0.0, 0.0, 0.0, 0.0)), 0.0)

    def test_unit(self):
        self.assertAlmostEqual(norm((1.0, 0.0, 0.0, 0.0)), 1.0)


class TestNormalize(unittest.TestCase):
    def _near_unit(self, q):
        n = norm(q)
        self.assertTrue(abs(n - 1.0) < 1e-12)

    def test_scales_to_unit(self):
        q = (0.0, 2.0, 0.0, 0.0)
        nq = normalize(q)
        self._near_unit(nq)
        self.assertAlmostEqual(nq[0], 0.0)
        self.assertAlmostEqual(nq[1], 1.0)

    def test_already_unit(self):
        q = (1.0, 0.0, 0.0, 0.0)
        self._near_unit(normalize(q))

    def test_zero_quaternion_returned_unchanged(self):
        z = (0.0, 0.0, 0.0, 0.0)
        self.assertEqual(normalize(z), z)


class TestToAxisAngle(unittest.TestCase):
    def test_identity_quaternion(self):
        axis, angle = to_axis_angle((1.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(angle, 0.0)
        self.assertEqual(axis, (0.0, 0.0, 1.0))

    def test_half_turn_about_x(self):
        # q = (cos(pi/2), sin(pi/2), 0, 0) = (0, 1, 0, 0)
        axis, angle = to_axis_angle((0.0, 1.0, 0.0, 0.0))
        self.assertAlmostEqual(angle, math.pi)
        self.assertAlmostEqual(axis[0], 1.0)
        self.assertAlmostEqual(axis[1], 0.0)
        self.assertAlmostEqual(axis[2], 0.0)

    def test_quarter_turn_about_y(self):
        # angle = pi/2, so w = cos(pi/4), y = sin(pi/4)
        h = math.sqrt(2.0) / 2.0
        axis, angle = to_axis_angle((h, 0.0, h, 0.0))
        self.assertAlmostEqual(angle, math.pi / 2.0)
        self.assertAlmostEqual(axis[0], 0.0)
        self.assertAlmostEqual(axis[1], 1.0)
        self.assertAlmostEqual(axis[2], 0.0)

    def test_unnormalized_input_handled(self):
        # 5 * a unit quaternion about z by pi/2
        h = math.sqrt(2.0) / 2.0
        q = (5.0 * h, 0.0, 0.0, 5.0 * h)
        axis, angle = to_axis_angle(q)
        self.assertAlmostEqual(angle, math.pi / 2.0)
        self.assertAlmostEqual(axis[0], 0.0)
        self.assertAlmostEqual(axis[1], 0.0)
        self.assertAlmostEqual(axis[2], 1.0)

    def test_near_zero_vector_part_uses_default_axis(self):
        axis, angle = to_axis_angle((1.0, 1e-15, 0.0, 0.0))
        self.assertEqual(axis, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(angle, 0.0)

    def test_scalar_clamping_when_drift_exceeds_one(self):
        # w slightly > 1 would make acos raise ValueError without clamping.
        axis, angle = to_axis_angle((1.0 + 1e-10, 0.0, 0.0, 0.0))
        self.assertEqual(axis, (0.0, 0.0, 1.0))
        self.assertTrue(angle >= 0.0)


if __name__ == "__main__":
    unittest.main()
