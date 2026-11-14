def sum_first_n(nums, n):
    """Return the sum of the first n numbers in the list."""
    total = 0
    for i in range(n):
        total += nums[i]
    return total
