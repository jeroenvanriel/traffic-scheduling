import unittest
import numpy as np

from simple import solve, common_substring


def symmetric_adj(distance):
    """Compute full symmetric adjacency matrix from lower triangular."""
    # number of nodes
    m = len(distance)

    # If entries in the upper triangular part are missing, we use the lower
    # triangular part to fill the upper part.
    for i in range(m):
        for k in range(i+1, m):
            if len(distance[i]) > k:
                distance[i][k] = distance[k][i]
            else:
                distance[i].append(distance[k][i])

    return distance


class SimpleFiniteBuffersTest(unittest.TestCase):

    def test_common_substring(self):
        a = [1, 2, 3, 4]
        b = [2, 3]

        self.assertEqual(common_substring(a, b), [2, 3])


    def test_basis(self):

        ptime = 1
        switch = 2

        # network:
        #      1    2
        # 0 -- 6 -- 7 -- 5
        #      3    4
        distance = np.array(symmetric_adj([
            [ 0                     ],
            [ 0, 0                  ],
            [ 0, 0, 0               ],
            [ 0, 0, 0, 0            ],
            [ 0, 0, 0, 0, 0         ],
            [ 0, 0, 0, 0, 0, 0      ],
            [ 1, 1, 0, 1, 0, 0, 0   ],
            [ 0, 0, 1, 0, 1, 1, 1, 0],
        ]))

        # each lane has space for 3 vehicles
        buffer = 3 * distance

        route = [
            [0, 6, 7, 5],       # main corridor
            [1, 6, 3],
            [2, 7, 4],
        ]

        # arrivals for each route
        release = [
            [1, 4, 8],
            [2, 3, 5],
            [2, 3, 5, 8, 10],
        ]

        solve(ptime, switch, distance, buffer, route, release)
