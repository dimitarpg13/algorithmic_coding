from utils import get_mid

def binary_search(arr, left, right, x):
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


def binary_search2(arr, left, right, x):
    """
    Perform binary search on a sorted array.
    :param arr: List of sorted elements
    :param left: Left index of the subarray to search
    :param right: Right index of the subarray to search
    :param x: Element to search for
    :return: Index of x in arr if found, otherwise -1
    """
    while left <= right + 1:
       mid = get_mid(left, right + 1)

       if mid == left or mid == right + 1:
           return -1

       if arr[mid] == x:
           return mid
       
       elif arr[mid] > x:
           right = mid - 1

       else:
           left = mid
         
    return -1


if __name__ == '__main__':
    arr = [-2, -1, 0, 1, 7, 11, 115, 223]

    print("testing binary_search")
    x = -1
    expected = 1
    result = binary_search(arr, 0, len(arr)-1, x)
    if result != -1:
        print(f"Element is present at index {result}")
        assert result == expected, f"Expected index {expected}"
    else:
        print("Element is not present in array")

    arr2 = [1, 5, 9, 11, 17, 21, 29, 33, 59]
 
    x = 29
    expected = 6
    result = binary_search(arr2, 0, len(arr2)-1, x)
    if result != -1:
        print(f"Element is present at index {result}")
        assert result == expected, f"Expected index {expected}"
    else:
        assert False, "Element is not present in array"
    
    x = 23
    expected = -1
    result = binary_search(arr2, 0, len(arr2)-1, x)
    if result == expected:
        print(f"Element {x} is not present at the array")
    else:
        assert False, f"Element {x} is found but it should not be present in array"

    x = 61
    expected = -1
    result = binary_search(arr2, 0, len(arr2)-1, x)
    if result == expected:
        print(f"Element {x} is not present at the array")
    else:
        assert False, f"Element {x} is found but it should not be present in array"
    print()
    print("testing binary_search2")
    x = -1
    expected = 1
    result = binary_search(arr, 0, len(arr)-1, x)
    if result != -1:
        print(f"Element is present at index {result}")
        assert result == expected, f"Expected index {expected}"
    else:
        print("Element is not present in array")

    x = 29
    expected = 6
    result = binary_search2(arr2, 0, len(arr2)-1, x)
    if result != -1:
        print(f"Element is present at index {result}")
        assert result == expected, f"Expected index {expected}"
    else:
        print("Element is not present in array")
    x = 23
    expected = -1
    result = binary_search2(arr2, 0, len(arr2)-1, x)
    if result == expected:
        print(f"Element {x} is not present at the array")
    else:
        assert False, f"Element {x} is found but it should not be present in array"
    x = 61
    expected = -1
    result = binary_search2(arr2, 0, len(arr2)-1, x)
    if result == expected:
        print(f"Element {x} is not present at the array")
    else:
        assert False, f"Element {x} is found but it should not be present in array"
    