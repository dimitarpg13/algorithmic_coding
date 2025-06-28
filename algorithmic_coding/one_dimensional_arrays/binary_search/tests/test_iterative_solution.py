import unittest
from one_dimensional_arrays.binary_search.iterative_solution import _binary_search


class IterativeSolutionTestCase(unittest.TestCase):
    def test_iterative(self):
        arr = [-2, -1, 0, 1, 7, 11, 115, 223]
        x = -1
        expected = 1
        result = _binary_search(arr, 0, len(arr) - 1, x)
        if result != -1:
            print(f"Element is present at index {result}")
            self.assertEqual(result, expected, f"Expected index {expected}")
        else:
            print("Element is not present in array")
            self.assertEqual(result, -1, "Element should not be present in array")


if __name__ == '__main__':
    unittest.main()
