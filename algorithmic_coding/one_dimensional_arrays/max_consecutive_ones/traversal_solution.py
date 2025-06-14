def max_consecutive_ones(arr):
    """
    Find the maximum number of consecutive 1s in a binary array.
    :param arr: List of integers (0s and 1s)
    :return: Maximum number of consecutive 1s
    """
    max_count = 0
    current_count = 0

    for num in arr:
        if num == 1:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0

    return max_count

if __name__ == '__main__':
    arr = [1, 1, 0, 1, 1, 1]
    print("Maximum number of consecutive 1s:", max_consecutive_ones(arr))
    
    arr = [0, 0, 0]
    print("Maximum number of consecutive 1s:", max_consecutive_ones(arr))
    
    arr = [1, 1, 1, 1]
    print("Maximum number of consecutive 1s:", max_consecutive_ones(arr))
    
    arr = []
    print("Maximum number of consecutive 1s:", max_consecutive_ones(arr))  # Edge case: empty array 