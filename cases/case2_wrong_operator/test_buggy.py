from buggy import is_valid_password


def test_long_password():
    assert is_valid_password("abcdefghij") is True


def test_exactly_eight():
    assert is_valid_password("abcdefgh") is True


def test_too_short():
    assert is_valid_password("abc") is False


def test_empty():
    assert is_valid_password("") is False
