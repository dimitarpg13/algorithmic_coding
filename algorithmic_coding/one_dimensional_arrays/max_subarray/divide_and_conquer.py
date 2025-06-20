# max subaray by using divide and conquer O(n*log(n)) time and O(n) space
from socket import fromfd

# divide the given array into two halves and return the maximum of following three:
# 1, Maximum subarray sum in left half.
# 2. Maximum subarray sum in the right half
# 3. Maximum subarray sum such that the subarray crosses the midpoint
# Maximum subarray in left and right halves can be found easily by two recursive calls.
# To find maximum subarray sum such that the subarray crosses the midpoint, find the maximum sum
# starting from mid point and ending at some point on the left of mid, then find the maximum sum
# starting from mid + 1 and ending with some point on right of mid + 1. Finally, combine the two and
# return the maximum among left, right and combination of both.

def max(a, b, c=None):
    if c is None:
        return a if a > b else b
    return max(max(a, b), c)


def max_crossing_sum(arr, l, m, h):
    # include elements on the left of mid
    left_sum = float('-inf')
    sum = 0
    for i in range(m, l - 1, -1):
        sum += arr[i]
        left_sum = max(left_sum, sum)
        
    right_sum = float('-inf')
    sum = 0
    for i in range(m, h + 1):
        sum += arr[i]
        right_sum = max(right_sum, sum)
        
    return max(left_sum + right_sum - arr[m], left_sum, right_sum)

def max_sum(arr, l, h):
    if l > h:
        return float('-inf')
    if l == h:
        return arr[l]
    
    m = l + (h - l) // 2
    
    return max(
        max_sum(arr, l, m),
        max_sum(arr, m + 1, h),
        max_crossing_sum(arr, l, m, h)
    )
    
def max_subarray_sum(arr):
    return max_sum(arr, 0, len(arr) - 1)
    
if __name__ == '__main__':
    arr = [-2, -3, 4, -1, -2, 1, 5, -3]
    print(max_subarray_sum(arr))
