import unittest
import numpy as np

from scheduling.single_intersection import read_instance, check_platoons, solve


class SingleIntersectionExactTest(unittest.TestCase):

    def test_exact_read_from_file(self):
        file = "./scheduling/instances/single1.txt"

        n, switch, release, length = read_instance(file)

        self.assertEqual(n, 3)
        self.assertEqual(switch, 2)

        self.assertTrue(np.array_equal(release, np.array([[0, 2, 4], [0, 2, 7]])))
        self.assertTrue(np.array_equal(length, np.array([[1, 1, 2], [1, 2, 3]])))


    def test_exact_check_overlap(self):
        n = 3

        release = np.array([[0, 1, 2], [0, 2, 5]])
        length = np.array([[0, 1, 1], [1, 3, 1]])

        # zero length is not allowed
        self.assertRaises(Exception, check_platoons, release, length,
                          msg="Platoon lengths should be positive.")

        length = np.array([[1, 1, 1], [1, 3, 1]])

        # no overlap, should be fine
        check_platoons(release, length)

        release = np.array([[0, 1, 2], [0, 2, 5]])
        length = np.array([[1, 1, 1], [1, 4, 1]])

        # overlap of second and third platoon in second lane
        self.assertRaises(Exception, check_platoons, release, length)


    def test_exact_example_two_platoons(self):
        """Test a specific example of the general rule for two platoons."""
        n = 1
        switch = 1

        n_A, n_B = 5, 7

        def order(r_B):
            release = np.array([[0], [r_B]])
            length = np.array([[n_A], [n_B]])

            y, o, obj = solve(n, switch, release, length, log=False)

            return o[0, 0]

        threshold = (n_B - n_A) * switch / (n_A + n_B)
        epsilon = 0.01

        r_B = threshold + epsilon
        self.assertEqual(order(r_B), 0)

        r_B = threshold - epsilon
        self.assertEqual(order(r_B), 1)


    def test_exact_example(self):
        n = 2
        switch = 2

        release = np.array([[0, 3], [1, 6]])
        length = np.array([[2, 1], [3, 1]])

        y, o, obj = solve(n, switch, release, length, log=False)

        # optimal lane sequence is 0, 1, 1, 0
        self.assertEqual(o[0, 0], 0)
        self.assertEqual(o[0, 1], 0)
        self.assertEqual(o[1, 0], 1)
        self.assertEqual(o[1, 1], 1)

        # total delay
        self.assertEqual(obj, 17)
