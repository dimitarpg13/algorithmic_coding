from one_dimensional_arrays.binary_search.utils import get_mid
from enum import IntFlag
from typing import List, Optional

class SearchType(IntFlag):
    RETURN_INVALID_IF_MISSING = 1
    RETURN_CLOSEST_ON_LEFT_IF_MISSING = 2
    RETURN_CLOSEST_ON_RIGHT_IF_MISSING = 4
    ARR_DEFINED_WITH_INDICES_OF_FIRST_AND_LAST = 8
    ARR_DEFINED_WITH_INDEX_OF_FIRST_AND_LENGTH = 16

def _binary_search(arr: List, left: int, right: int, x: float) -> int:
    """
    Perform binary search on a sorted array.
    :param arr: List of sorted elements
    :param left: Left index of the subarray to search
    :param right: Right index of the subarray to search
    :param x: Element to search for
    :return: Index of x in arr if found, otherwise -1
    """
        
    while left <= right:
        mid = get_mid(left, right)
				
        if arr[mid] == x:
            return mid
		
        elif arr[mid] > x:
            right = mid - 1
        
        else:
            left = mid + 1

    return -1


def _binary_search2(arr: List, left: int, length: int, x: float) -> int:
    """
    Perform binary search on a sorted array.
    :param arr: List of sorted elements
    :param left: Left index of the subarray to search
    :param length: Length of the subarray to search
    :param x: Element to search for
    :return: Index of x in arr if found, otherwise -1
    """
    while left <= length:
       mid = get_mid(left, length)

       if mid == left or mid == length:
           return -1

       if arr[mid] == x:
           return mid
       
       elif arr[mid] > x:
           length = mid

       else:
           left = mid
         
    return -1


if __name__ == '__main__':
    arr = [-2, -1, 0, 1, 7, 11, 115, 223]

    print("testing binary_search")
    x = -1
    expected = 1
    result = _binary_search(arr, 0, len(arr)-1, x)
    if result != -1:
        print(f"Element is present at index {result}")
        assert result == expected, f"Expected index {expected}"
    else:
        print("Element is not present in array")

    arr2 = [1, 5, 9, 11, 17, 21, 29, 33, 59]
 
    x = 29
    expected = 6
    result = _binary_search(arr2, 0, len(arr2)-1, x)
    if result != -1:
        print(f"Element is present at index {result}")
        assert result == expected, f"Expected index {expected}"
    else:
        assert False, "Element is not present in array"
    
    x = 23
    expected = -1
    result = _binary_search(arr2, 0, len(arr2)-1, x)
    if result == expected:
        print(f"Element {x} is not present at the array")
    else:
        assert False, f"Element {x} is found but it should not be present in array"

    x = 61
    expected = -1
    result = _binary_search(arr2, 0, len(arr2)-1, x)
    if result == expected:
        print(f"Element {x} is not present at the array")
    else:
        assert False, f"Element {x} is found but it should not be present in array"
    print()
    print("testing binary_search2")
    x = -1
    expected = 1
    result = _binary_search(arr, 0, len(arr)-1, x)
    if result != -1:
        print(f"Element is present at index {result}")
        assert result == expected, f"Expected index {expected}"
    else:
        print("Element is not present in array")

    x = 29
    expected = 6
    result = _binary_search2(arr2, 0, len(arr2), x)
    if result != -1:
        print(f"Element is present at index {result}")
        assert result == expected, f"Expected index {expected}"
    else:
        print("Element is not present in array")
    x = 23
    expected = -1
    result = _binary_search2(arr2, 0, len(arr2), x)
    if result == expected:
        print(f"Element {x} is not present at the array")
    else:
        assert False, f"Element {x} is found but it should not be present in array"
    x = 61
    expected = -1
    result = _binary_search2(arr2, 0, len(arr2), x)
    if result == expected:
        print(f"Element {x} is not present at the array")
    else:
        assert False, f"Element {x} is found but it should not be present in array"
    