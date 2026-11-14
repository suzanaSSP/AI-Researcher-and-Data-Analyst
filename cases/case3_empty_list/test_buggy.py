from buggy import find_max


def test_basic():
    assert find_max([3, 9, 2]) == 9


def test_negatives():
    assert find_max([-5, -2, -9]) == -2


def test_single():
    assert find_max([7]) == 7


def test_empty_returns_none():
    assert find_max([]) is None
