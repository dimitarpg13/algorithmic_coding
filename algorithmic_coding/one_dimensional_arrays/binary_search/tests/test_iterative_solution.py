import unittest
from parameterized import parameterized
from one_dimensional_arrays.binary_search.iterative_solution import _binary_search1, _binary_search2



class IterativeSolutionTestCase(unittest.TestCase):
    def setUp(self):
        print("========================================")

    def tearDown(self):
        pass
    
    def print_header(self, arr=None, x=None, expected=None):
        print(f"Testing binary search for element {x} in array {arr}")
        print(f"Expected index: {expected}")
        print(f"Running {self.id()}...")
        
    @parameterized.expand([
            ([-2, -1, 0, 1, 7, 11, 115, 223], -1, 1),  # (array, x, expected)
            ([1, 5, 9, 11, 17, 21, 29, 33, 59], 29, 6),  # (array, x, expected)
            ([1, 5, 9, 11, 17, 21, 29, 33, 59], 23, -1), # (array, x, expected)
            ([1, 5, 9, 11, 17, 21, 29, 33, 59], 61, -1) # (array, x, expected)

        ])
    def test_iterative_solution_binary_search1(self, arr, x, expected):
        self.print_header(arr, x, expected)

        # Call the binary search function
        result = _binary_search1(arr, 0, len(arr) - 1, x)
        if result != -1:
            print(f"Element {x} is present at index {result}")
            self.assertEqual(result, expected, f"For element {x} expected index {expected}")
        else:
            print(f"Element {x} is not present in array")
            self.assertEqual(result, -1, f"Element {x} should not be present in array {arr}")

    @parameterized.expand([
            ([-2, -1, 0, 1, 7, 11, 115, 223], -1, 1),  # (array, x, expected)
            ([1, 5, 9, 11, 17, 21, 29, 33, 59], 29, 6),  # (array, x, expected)
            ([1, 5, 9, 11, 17, 21, 29, 33, 59], 23, -1), # (array, x, expected)
            ([1, 5, 9, 11, 17, 21, 29, 33, 59], 61, -1) # (array, x, expected)

        ])
    def test_iterative_solution_binary_search2(self, arr, x, expected):
        self.print_header(arr, x, expected)

        result = _binary_search2(arr, 0, len(arr), x)
        if result != -1:
            print(f"Element {x} is present at index {result}")
            self.assertEqual(result, expected, f"For element {x} expected index {expected}")
        else:
            print(f"Element {x} is not present in array")
            self.assertEqual(result, -1, f"Element {x} should not be present in array {arr}")


if __name__ == '__main__':
    unittest.main()
