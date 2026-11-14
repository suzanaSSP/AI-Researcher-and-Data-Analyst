from buggy import merge_intervals


def test_empty():
    assert merge_intervals([]) == []


def test_no_overlap():
    assert merge_intervals([[1, 2], [4, 5]]) == [[1, 2], [4, 5]]


def test_overlap():
    assert merge_intervals([[1, 4], [2, 6]]) == [[1, 6]]


def test_touching_intervals_merge():
    assert merge_intervals([[1, 2], [2, 3]]) == [[1, 3]]


def test_unsorted_input():
    assert merge_intervals([[8, 10], [1, 3], [2, 6]]) == [[1, 6], [8, 10]]


def test_nested():
    assert merge_intervals([[1, 10], [2, 3]]) == [[1, 10]]
