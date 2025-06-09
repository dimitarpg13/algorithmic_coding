def binary_search(arr, low, high, x):
    while low <= high:
        mid = low + (high - low) // 2
        
        if arr[mid] == x:
            return mid 
        
        elif arr[mid] < x:
            low = mid + 1
        
        else:
            high = mid - 1
            
    return -1

if __name__ == '__main__':
    arr = [-2, -1, 0, 1, 7, 11, 115, 223]
    x = -1
    
    result = binary_search(arr, 0, len(arr)-1, x)
    if result != -1:
        print("Element is present at index", result)
    else:
        print("Element is not present in array")
