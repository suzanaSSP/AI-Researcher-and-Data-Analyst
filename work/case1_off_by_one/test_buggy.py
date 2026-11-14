from buggy import sum_first_n


def test_basic():
    assert sum_first_n([1, 2, 3, 4], 2) == 3


def test_single_item():
    assert sum_first_n([5], 1) == 5


def test_whole_list():
    assert sum_first_n([1, 2, 3], 3) == 6


def test_zero():
    assert sum_first_n([1, 2, 3], 0) == 0
