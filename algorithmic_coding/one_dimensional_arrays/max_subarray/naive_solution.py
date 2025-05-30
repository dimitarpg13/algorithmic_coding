
def max_subarray_sum(arr):
    res = arr[0]

    for i in range(1, len(arr)):
        curr_sum = 0

        for j in range(i, len(arr)):
            curr_sum += arr[j]

            res = max(curr_sum, res)

    return res


if __name__ == '__main__':
    arr = [-2, -3, 4, -1, -2, 1, 5, -3]
    print(max_subarray_sum(arr))