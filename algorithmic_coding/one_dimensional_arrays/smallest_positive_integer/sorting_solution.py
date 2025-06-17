def smallest_positive_integer(A):
    # Implement your solution here
    if A is None or len(A) == 0:
        return 1
    set_pos = {a for a in A if a > 0}
    sorted_pos = sorted(set_pos)
    prev = None
    if len(sorted_pos) > 0:
        for a in sorted_pos:
            if prev is not None:
                if a - prev > 1:
                    return prev + 1
            else:
                if a != 1:
                    return 1
            prev = a 
        return sorted_pos[-1] + 1
    else:
        return 1
    
if __name__ == '__main__':
    A = [1, 3, 6, 4, 1, 2]
    print("Smallest positive integer not in the array:", smallest_positive_integer(A))
    
    A = [1, 2, 3]
    print("Smallest positive integer not in the array:", smallest_positive_integer(A))
    
    A = [-1, -3]
    print("Smallest positive integer not in the array:", smallest_positive_integer(A))
    
    A = []
    print("Smallest positive integer not in the array:", smallest_positive_integer(A))  # Edge case: empty array

    A = [2, 3, 7, 6, 8, -1, -10, 15]
    print("Smallest positive integer not in the array:", smallest_positive_integer(A))
    
    A = [1, 1, 0, -1, -2]
    print("Smallest positive integer not in the array:", smallest_positive_integer(A))
    
    A = [1, 2, 0]
    print("Smallest positive integer not in the array:", smallest_positive_integer(A))
    
    A = [3, 4, -1, 1]
    print("Smallest positive integer not in the array:", smallest_positive_integer(A))
    
    A = [7, 8, 9, 11, 12]
    print("Smallest positive integer not in the array:", smallest_positive_integer(A))
    
    A = [1, 2, 3, 4, 5]
    print("Smallest positive integer not in the array:", smallest_positive_integer(A))
    
    A = [-1, -2, -3]
    print("Smallest positive integer not in the array:", smallest_positive_integer(A))
    
    A = [2]
    print("Smallest positive integer not in the array:", smallest_positive_integer(A))
    
    A = [1]
    print("Smallest positive integer not in the array:", smallest_positive_integer(A))
    
    A = [0]
    print("Smallest positive integer not in the array:", smallest_positive_integer(A))
    
    A = [1, 2, 2, 3, 3]
    print("Smallest positive integer not in the array:", smallest_positive_integer(A))
    
    A = [5, 3, 2, 1, 4]
    print("Smallest positive integer not in the array:", smallest_positive_integer(A))
    
    A = [10, 20, 30]
    print("Smallest positive integer not in the array:", smallest_positive_integer(A))
    
    A = [1, 3, 5, 7]
    print("Smallest positive integer not in the array:", smallest_positive_integer(A))
    
    A = [2, 4, 6, 8]
    print("Smallest positive integer not in the array:", smallest_positive_integer(A))