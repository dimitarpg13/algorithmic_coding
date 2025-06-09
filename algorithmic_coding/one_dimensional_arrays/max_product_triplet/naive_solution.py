def max_product(arr):
    """
    Find the maximum product of any triplet in the array.
    :param arr: List of integers
    :return: Maximum product of any triplet
    """
    n = len(arr)
    if n < 3:
        return None  # Not enough elements for a triplet

    max_product = float('-inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                product = arr[i] * arr[j] * arr[k]
                max_product = max(max_product, product)
                
    return max_product
if __name__ == '__main__':
    arr = [1, 2, 3, 4, 5]
    print("Maximum product of any triplet:", max_product(arr))
    
    arr = [-10, -10, 1, 3, 2]
    print("Maximum product of any triplet:", max_product(arr))
    
    arr = [1, 2]
    print("Maximum product of any triplet:", max_product(arr))  # Not enough elements