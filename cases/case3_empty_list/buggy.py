def find_max(nums):
    """Return the largest number in the list, or None if the list is empty."""
    max_val = nums[0]
    for n in nums:
        if n > max_val:
            max_val = n
    return max_val
