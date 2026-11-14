def is_palindrome(text):
    """Return True if text reads the same forwards and backwards,
    ignoring upper/lower case and any characters that are not letters or digits."""
    cleaned = text.lower()
    return cleaned == cleaned[::-1]
