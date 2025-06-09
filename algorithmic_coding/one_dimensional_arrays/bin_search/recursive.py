def binary_search(arr, left, right, x):
    """
    Perform binary search on a sorted array.
    :param arr: List of sorted elements
    :param left: Left index of the subarray to search
    :param right: Right index of the subarray to search
    :param x: Element to search for
    :return: Index of x in arr if found, otherwise -1
    """
    if left > right:
        return -1  # Not found
    mid = (left + right) // 2
    if arr[mid] == x:
        return mid
    elif arr[mid] < x:
        return binary_search(arr, mid + 1, right, x)
    else:
        return binary_search(arr, left, mid - 1, x)

# Example usage:
if __name__ == "__main__":
    arr = [1, 3, 5, 7, 9, 11]
    target = 7
    result = binary_search(arr, target)
    print(f"Index of {target}: {result}")