import unittest
import numpy as np
from src import group


class ArrangeGroupTestCase(unittest.TestCase):
    def test_one_center_group(self):
        image = np.zeros((10, 10))
        image[3:7, 3:7] = 1
        nonzero = np.dstack(np.nonzero(image))[0]
        group_range = group._arrange_groups(nonzero, 1, 1, 0)
        self.assertEqual(group_range, [[3, 6, 3, 6]])

    def test_four_corner_groups(self):
        image = np.zeros((10, 10))
        image[1:4, 1:4] = 1
        image[1:4, 6:9] = 1
        image[6:9, 1:4] = 1
        image[6:9, 6:9] = 1
        nonzero = np.dstack(np.nonzero(image))[0]
        group_range = group._arrange_groups(nonzero, 1, 1, 0)
        self.assertEqual(group_range, [[1, 3, 1, 3], [1, 3, 6, 8],
                                       [6, 8, 1, 3], [6, 8, 6, 8]])

    def test_four_subgroups(self):
        image = np.zeros((10, 10))
        image[1:4, 1:4] = 1
        image[1:4, 6:9] = 1
        image[6:9, 1:4] = 1
        image[6:9, 6:9] = 1
        nonzero = np.dstack(np.nonzero(image))[0]
        group_range = group._arrange_groups(nonzero, 3, 1, 0)
        self.assertEqual(group_range, [[1, 8, 1, 8]])

    def test_nested_groups(self):
        image = np.zeros((10, 10))
        image[[0, 9], :] = 1
        image[:, [0, 9]] = 1
        image[[3, 6], 3:7] = 1
        image[3:7, [3, 6]] = 1
        nonzero = np.dstack(np.nonzero(image))[0]
        group_range = group._arrange_groups(nonzero, 3, 1, 0)
        self.assertEqual(group_range, [[0, 9, 0, 9]])

    def test_group_size(self):
        image = np.zeros((10, 10))
        image[1:3, 1:3] = 1
        image[5:9, 5:9] = 1
        nonzero = np.dstack(np.nonzero(image))[0]
        group_range = group._arrange_groups(nonzero, 2, 1, 0)
        self.assertEqual(group_range, [[1, 2, 1, 2], [5, 8, 5, 8]])

    def test_group_factor(self):
        image = np.zeros((10, 10))
        image[1:3, 1:3] = 1
        image[5:9, 5:9] = 1
        nonzero = np.dstack(np.nonzero(image))[0]
        group_range = group._arrange_groups(nonzero, 2, 1, 0.5)
        self.assertEqual(group_range, [[5, 8, 5, 8]])


if __name__ == '__main__':
    unittest.main()
