
# traverse the array from left to right and for each element find the
# maximum sum among all sub-arrays ending at that element.
def max_subarray_sum(arr):
    res = arr[0]
    max_ending = arr[0]

    for i in range(1, len(arr)):

        max_ending = max(max_ending + arr[i], arr[i])
        res = max(res, max_ending)

    return res


if __name__ == '__main__':
    arr = [-2, -3, 4, -1, -2, 1, 5, -3]
    print(max_subarray_sum(arr))