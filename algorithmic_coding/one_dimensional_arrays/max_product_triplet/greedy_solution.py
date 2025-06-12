def max_product(arr):
    """
    Find the maximum product of any triplet in the array using a greedy approach.
    :param arr: List of integers
    :return: Maximum product of any triplet
    """
    n = len(arr)
    if n < 3:
        return None  # Not enough elements for a triplet

    # Initialize variables to track the three largest and two smallest numbers
    max1 = max2 = max3 = float('-inf')
    min1 = min2 = float('inf')

    for num in arr:
        # Update the three largest numbers
        if num > max1:
            max3 = max2
            max2 = max1
            max1 = num
        elif num > max2:
            max3 = max2
            max2 = num
        elif num > max3:
            max3 = num

        # Update the two smallest numbers
        if num < min1:
            min2 = min1
            min1 = num
        elif num < min2:
            min2 = num

    # The maximum product can be either from the three largest numbers or two smallest and one largest
    return max(max1 * max2 * max3, min1 * min2 * max1)

if __name__ == '__main__':
    arr = [1, 2, 3, 4, 5]
    print("Maximum product of any triplet:", max_product(arr))
    
    arr = [-10, -10, 1, 3, 2]
    print("Maximum product of any triplet:", max_product(arr))
    
    arr = [1, 2]
    print("Maximum product of any triplet:", max_product(arr))  # Not enough elements