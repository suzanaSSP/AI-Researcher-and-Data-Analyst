from buggy import is_palindrome


def test_simple():
    assert is_palindrome("racecar") is True


def test_mixed_case():
    assert is_palindrome("Racecar") is True


def test_punctuation_and_spaces():
    assert is_palindrome("A man, a plan, a canal: Panama") is True


def test_not_palindrome():
    assert is_palindrome("hello") is False


def test_empty():
    assert is_palindrome("") is True
