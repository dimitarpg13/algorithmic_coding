import unittest
from parameterized import parameterized
from one_dimensional_arrays.binary_search.iterative_solution import _binary_search



class IterativeSolutionTestCase(unittest.TestCase):
    @parameterized.expand([
            ([-2, -1, 0, 1, 7, 11, 115, 223], -1, 1),  # (array, x, expected)
            ([1, 5, 9, 11, 17, 21, 29, 33, 59], 29, 6),  # (array, x, expected)
            ([1, 5, 9, 11, 17, 21, 29, 33, 59], 23, -1), # (array, x, expected)
            ([1, 5, 9, 11, 17, 21, 29, 33, 59], 61, -1) # (array, x, expected)

        ])
    def test_iterative_solution_1(self, arr, x, expected):
        result = _binary_search(arr, 0, len(arr) - 1, x)
        if result != -1:
            print(f"Element {x} is present at index {result}")
            self.assertEqual(result, expected, f"For element {x} expected index {expected}")
        else:
            print(f"Element {x} is not present in array")
            self.assertEqual(result, -1, f"Element {x} should not be present in array {arr}")


if __name__ == '__main__':
    unittest.main()
