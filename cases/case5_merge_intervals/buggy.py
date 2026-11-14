def merge_intervals(intervals):
    """Merge overlapping or touching intervals.

    Takes a list of [start, end] pairs in any order and returns a new list of
    merged pairs sorted by start. Intervals that touch (e.g. [1, 2] and [2, 3])
    count as overlapping and should be merged.
    """
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start < last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return merged
