import unittest
import numpy as np

from exact import check_vehicles, solve


class SingleIntersectionExactTest(unittest.TestCase):

    def test_exact_check_overlap(self):
        releases = [np.array([0, 1, 2]), np.array([0, 2, 5])]
        lengths = [np.array([0, 1, 1]), np.array([1, 3, 1])]

        # zero length is not allowed
        self.assertRaises(Exception, check_vehicles, releases, lengths,
                          msg="Platoon lengths should be positive.")

        lengths = [np.array([1, 1, 1]), np.array([1, 3, 1])]

        # no overlap, should be fine
        check_vehicles(releases, lengths)

        releases = [np.array([0, 1, 2]), np.array([0, 2, 5]), np.array([0, 3])]
        lengths = [np.array([1, 1, 1]), np.array([1, 4, 1]), np.array([1, 1])]

        # overlap of second and third platoon in second lane
        self.assertRaises(Exception, check_vehicles, releases, lengths)


    def test_exact_example_two_platoons(self):
        """Test a specific example of the general rule for two platoons."""
        n = 1
        switch = 1

        n_A, n_B = 5, 7

        def order(r_B):
            releases = [np.array([0]), np.array([r_B])]
            lengths = [np.array([n_A]), np.array([n_B])]

            y, o, obj = solve(switch, releases, lengths, consolelog=False)

            return o[0, 1, 0, 0]

        threshold = (n_B - n_A) * switch / (n_A + n_B)
        epsilon = 0.01

        r_B = threshold + epsilon
        self.assertEqual(order(r_B), 0)

        r_B = threshold - epsilon
        self.assertEqual(order(r_B), 1)


    def test_exact_example(self):
        """Test some small example."""
        n = 2
        switch = 2

        releases = [np.array([0, 3]), np.array([1, 6])]
        lengths = [np.array([2, 1]), np.array([3, 1])]

        y, o, obj = solve(switch, releases, lengths, consolelog=False)

        # optimal lane sequence is 0, 1, 1, 0
        self.assertEqual(o[0, 1, 0, 0], 0)
        self.assertEqual(o[0, 1, 0, 1], 0)
        self.assertEqual(o[0, 1, 1, 0], 1)
        self.assertEqual(o[0, 1, 1, 1], 1)

        # total delay
        self.assertEqual(obj, -17)
