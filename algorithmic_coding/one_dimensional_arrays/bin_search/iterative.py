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
        mid = left + (right - left) // 2
        
        if arr[mid] == x:
            return mid 
        
        elif arr[mid] < x:
            left = mid + 1
        
        else:
            right = mid - 1
            
    return -1

if __name__ == '__main__':
    arr = [-2, -1, 0, 1, 7, 11, 115, 223]
    x = -1
    
    result = binary_search(arr, 0, len(arr)-1, x)
    if result != -1:
        print("Element is present at index", result)
    else:
        print("Element is not present in array")
