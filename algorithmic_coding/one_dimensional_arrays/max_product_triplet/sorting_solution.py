
def max_product(arr):
    """
    Find the maximum product of any triplet in the array using sorting.
    :param arr: List of integers
    :return: Maximum product of any triplet
    """
    n = len(arr)
    if n < 3:
        return None  # Not enough elements for a triplet

    # Sort the array
    arr.sort()

    # The maximum product can be either from the three largest numbers or two smallest and one largest
    max_product = max(arr[-1] * arr[-2] * arr[-3], arr[0] * arr[1] * arr[-1])

    return max_product

if __name__ == '__main__':
    arr = [1, 2, 3, 4, 5]
    print("Maximum product of any triplet:", max_product(arr))
    
    arr = [-10, -10, 1, 3, 2]
    print("Maximum product of any triplet:", max_product(arr))
    
    arr = [1, 2]
    print("Maximum product of any triplet:", max_product(arr))  # Not enough elements