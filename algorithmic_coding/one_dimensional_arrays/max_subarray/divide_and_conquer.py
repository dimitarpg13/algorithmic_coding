# max subaray by using divide and conquer O(n*log(n)) time and O(n) space

# divide the given array into two halves and return the maximum of following three:
# 1, Maximum subarray sum in left half.
# 2. Maximum subarray sum in the right half
# 3. Maximum subarray sum such that the subarray crosses the midpoint

def max(a, b, c=None):
    if c is None:
        return a if a > b else b
    return max(max(a, b), c)


def max_crossing_sum(arr, l, m, h):
    # include elements on the left of mid
    left_sum = float('inf')
    sum = 0
    for i in range(m, l - 1, -1):
        sum += arr[i]
